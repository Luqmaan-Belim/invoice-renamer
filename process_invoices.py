#!/usr/bin/env python3
"""
Invoice Renamer - changes-only Google Drive processor with optional Odoo attachment.

The first page is OCRed and renamed from an Odoo customer invoice number such as
INV/2026/042130 to 2026_042130.pdf. The extractor prioritises the Fresh Bake
invoice-number row and only accepts explicit invoice-number formats; company
registration numbers and other slash-separated values are ignored.
"""

import io
import json
import logging
import os
import pathlib
import re
from typing import Dict, List, Optional, Tuple

import cv2
import fitz  # PyMuPDF
import numpy as np
import pytesseract
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("invoice_renamer")


FOLDER_ID = os.environ["GDRIVE_FOLDER_ID"].strip()
SA_JSON = json.loads(os.environ["GDRIVE_SA_JSON"])
DEBUG = os.environ.get("DEBUG_LIST", "0") == "1"
ENABLE_ODOO = os.environ.get("ENABLE_ODOO", "0") == "1"

TOKEN_PATH = pathlib.Path(".drive_change_token")
TOKEN_APPDATA_NAME = "invoice-renamer-token"
DISABLE_APPDATA_TOKEN = os.environ.get("DISABLE_APPDATA_TOKEN", "0") == "1"

PDF_MT = "application/pdf"
FOLDER_MT = "application/vnd.google-apps.folder"
SHORTCUT_MT = "application/vnd.google-apps.shortcut"

IMG_PDF_RE = re.compile(r"^IMG.*\.pdf$", re.I)
RENAMED_RE = re.compile(r"^\d{4}(?:_\d{1,6})?(?:_\d+)?\.pdf$", re.I)

# OCR commonly substitutes these letters for digits. Translation is applied only
# inside a candidate invoice number, never to the rest of the page text.
OCR_DIGIT_TRANSLATION = str.maketrans(
    {
        "O": "0",
        "Q": "0",
        "D": "0",
        "I": "1",
        "L": "1",
        "Z": "2",
        "S": "5",
        "G": "6",
        "B": "8",
    }
)
OCR_DIGIT = r"[0-9OQDISBGLZ]"
OCR_YEAR = rf"(?P<year>(?:{OCR_DIGIT}\s*){{4}})"
OCR_SERIAL = rf"(?P<serial>(?:{OCR_DIGIT}\s*){{1,6}})(?!{OCR_DIGIT})"
OCR_SEPARATOR = r"\s*[/\\|_.-]\s*"

# Important: the negative look-ahead after INV/NV prevents matching the 'nv' in
# the word 'Invoice'. That old behaviour caused 'Tax Invoice Reg No: 2011/053816'
# to be interpreted as invoice 2011_053816.
EXPLICIT_INVOICE_RE = re.compile(
    rf"(?<![A-Z0-9])(?P<prefix>[I1L]\s*N\s*V|N\s*V)(?![A-Z0-9])"
    rf"[\s:#/\\|_.-]*{OCR_YEAR}{OCR_SEPARATOR}{OCR_SERIAL}",
    re.I,
)

# Safe fallback when OCR drops the INV token but still reads the specific field
# label from the Fresh Bake template.
LABELLED_INVOICE_RE = re.compile(
    rf"(?:TAX\s+)?INVOICE\s+"
    rf"(?:N[UO]M(?:BER|8ER)|NUMBER|NO\.?|NR\.?)"
    rf"\s*[:#-]?\s*{OCR_YEAR}{OCR_SEPARATOR}{OCR_SERIAL}",
    re.I,
)


def drive():
    creds = service_account.Credentials.from_service_account_info(
        SA_JSON,
        scopes=["https://www.googleapis.com/auth/drive"],
    )
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def resolve_root(d, folder_id: str) -> Tuple[str, Optional[str]]:
    """Return (real_root_id, drive_id), resolving a shortcut if needed."""
    meta = d.files().get(
        fileId=folder_id,
        fields="id,name,mimeType,driveId,shortcutDetails",
        supportsAllDrives=True,
    ).execute()
    if meta.get("mimeType") == SHORTCUT_MT:
        target_id = meta["shortcutDetails"]["targetId"]
        resolved = d.files().get(
            fileId=target_id,
            fields="id,name,mimeType,driveId",
            supportsAllDrives=True,
        ).execute()
        log.info("Resolved shortcut root: %s -> %s", folder_id, resolved["id"])
        return resolved["id"], resolved.get("driveId")
    return meta["id"], meta.get("driveId")


def get_start_token(d, drive_id: Optional[str]) -> str:
    if drive_id:
        response = d.changes().getStartPageToken(
            driveId=drive_id,
            supportsAllDrives=True,
        ).execute()
    else:
        response = d.changes().getStartPageToken().execute()
    return response["startPageToken"]


def load_token(d) -> Optional[str]:
    token: Optional[str] = None

    if not DISABLE_APPDATA_TOKEN:
        try:
            response = d.files().list(
                spaces="appDataFolder",
                q=f"name='{TOKEN_APPDATA_NAME}' and trashed=false",
                fields="files(id)",
                pageSize=1,
            ).execute()
            files = response.get("files") or []
            if files:
                data = d.files().get_media(fileId=files[0]["id"]).execute()
                token = (
                    data.decode("utf-8", errors="ignore").strip()
                    if isinstance(data, bytes)
                    else str(data).strip()
                )
        except Exception as exc:
            log.warning("Unable to read Drive token from appData: %s", exc)

    if token:
        try:
            TOKEN_PATH.write_text(token)
        except Exception:
            pass
        return token

    try:
        token = TOKEN_PATH.read_text().strip()
        if not token:
            return None
        if not DISABLE_APPDATA_TOKEN:
            log.info(
                "Using local Drive change token fallback; set "
                "DISABLE_APPDATA_TOKEN=1 to opt out of appData usage."
            )
        return token
    except FileNotFoundError:
        return None


def save_token(d, token: str):
    try:
        TOKEN_PATH.write_text(token)
    except Exception:
        pass

    if DISABLE_APPDATA_TOKEN:
        return

    media = MediaIoBaseUpload(io.BytesIO(token.encode("utf-8")), mimetype="text/plain")
    try:
        response = d.files().list(
            spaces="appDataFolder",
            q=f"name='{TOKEN_APPDATA_NAME}' and trashed=false",
            fields="files(id)",
            pageSize=1,
        ).execute()
        files = response.get("files") or []
        if files:
            d.files().update(fileId=files[0]["id"], media_body=media).execute()
        else:
            body = {"name": TOKEN_APPDATA_NAME, "parents": ["appDataFolder"]}
            d.files().create(body=body, media_body=media, fields="id").execute()
    except Exception as exc:
        log.warning("Unable to save Drive token to appData: %s", exc)


def file_meta(d, file_id: str, fields: str):
    return d.files().get(
        fileId=file_id,
        fields=fields,
        supportsAllDrives=True,
    ).execute()


def is_under_root(
    d,
    file_parents: List[str],
    root_id: str,
    parent_cache: Dict[str, Optional[List[str]]],
) -> bool:
    if not file_parents:
        return False

    stack = list(file_parents)
    while stack:
        parent_id = stack.pop()
        if parent_id == root_id:
            return True

        if parent_id in parent_cache:
            parents = parent_cache[parent_id]
        else:
            try:
                metadata = d.files().get(
                    fileId=parent_id,
                    fields="id,parents",
                    supportsAllDrives=True,
                ).execute()
                parents = metadata.get("parents") or []
            except Exception:
                parents = []
            parent_cache[parent_id] = parents if parents else None

        if parents:
            stack.extend(parents)

    return False


def download_pdf_bytes(d, file_id: str) -> Optional[bytes]:
    from googleapiclient.http import MediaIoBaseDownload

    request = d.files().get_media(fileId=file_id)
    buffer = io.BytesIO()
    downloader = MediaIoBaseDownload(buffer, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    data = buffer.getvalue()
    return data or None


def rasterize_first_page(pdf_bytes: bytes) -> Optional[np.ndarray]:
    matrix = fitz.Matrix(300.0 / 72.0, 300.0 / 72.0)
    with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
        if document.page_count == 0:
            return None
        page = document.load_page(0)
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        return np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
            pixmap.height,
            pixmap.width,
            3,
        )


def ocr_with_confidence(image: np.ndarray, psm: int = 6) -> Tuple[str, float]:
    data = pytesseract.image_to_data(
        image,
        config=f"--psm {psm} --oem 3",
        output_type=pytesseract.Output.DICT,
    )
    text = " ".join(str(value) for value in data.get("text", []) if str(value).strip())

    confidences: List[float] = []
    for value in data.get("conf", []):
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            continue
        if confidence >= 0:
            confidences.append(confidence)

    average = sum(confidences) / len(confidences) if confidences else 0.0
    return text.strip(), average


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Increase local contrast and binarise without amplifying background noise."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return cv2.threshold(
        enhanced,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )[1]


def _normalise_ocr_digits(value: str) -> str:
    translated = value.upper().translate(OCR_DIGIT_TRANSLATION)
    return re.sub(r"\D", "", translated)


def _invoice_from_match(match: re.Match) -> Optional[str]:
    year = _normalise_ocr_digits(match.group("year"))
    serial = _normalise_ocr_digits(match.group("serial"))

    if len(year) != 4 or not year.startswith("20"):
        return None
    if not 1 <= len(serial) <= 6:
        return None

    return f"{year}_{serial.zfill(6)}"


def extract_invoice_number(text: str) -> Optional[str]:
    """
    Extract an Odoo invoice number from OCR text.

    Accepted examples include INV/2026/042130 and, as a guarded fallback,
    'Tax Invoice Number 2026/042130'. A bare slash-separated number is never
    accepted, so Fresh Bake's registration number cannot be selected.
    """
    normalised_text = " ".join(text.upper().split())

    for pattern in (EXPLICIT_INVOICE_RE, LABELLED_INVOICE_RE):
        for match in pattern.finditer(normalised_text):
            invoice = _invoice_from_match(match)
            if invoice:
                return invoice

    return None


def _fresh_bake_header_crop(image: np.ndarray) -> np.ndarray:
    """Crop the new-template invoice-number zone while excluding the Reg No block."""
    height, width = image.shape[:2]
    y1 = max(0, int(height * 0.05))
    y2 = min(height, int(height * 0.25))
    x1 = max(0, int(width * 0.01))
    x2 = min(width, int(width * 0.70))
    return image[y1:y2, x1:x2]


def extract_invoice_number_from_image(image: np.ndarray) -> Tuple[Optional[str], float, str]:
    """OCR the most reliable regions first and return (number, confidence, source)."""
    header = _fresh_bake_header_crop(image)
    attempts = (
        ("fresh-bake-header", header, 6),
        ("fresh-bake-header-enhanced", preprocess_image(header), 6),
        ("full-page", image, 6),
        ("full-page-enhanced", preprocess_image(image), 6),
        ("full-page-sparse", image, 11),
    )

    best_confidence = 0.0
    for source, attempt_image, psm in attempts:
        text, confidence = ocr_with_confidence(attempt_image, psm=psm)
        best_confidence = max(best_confidence, confidence)
        invoice = extract_invoice_number(text)
        if invoice:
            if DEBUG:
                log.info(
                    "OCR MATCH invoice=%s source=%s confidence=%.1f",
                    invoice,
                    source,
                    confidence,
                )
            return invoice, confidence, source

    return None, best_confidence, "no-match"


def unique_name_in_folder(d, folder_id: str, base: str) -> str:
    name = base
    suffix = 1
    while True:
        query = f"name='{name}' and '{folder_id}' in parents and trashed=false"
        response = d.files().list(
            q=query,
            fields="files(id)",
            pageSize=1,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        ).execute()
        if not response.get("files"):
            return name
        root, extension = os.path.splitext(base)
        name = f"{root}_{suffix}{extension}"
        suffix += 1


def rename_in_drive(d, file_id: str, new_name: str):
    d.files().update(
        fileId=file_id,
        body={"name": new_name},
        supportsAllDrives=True,
    ).execute()


def main():
    d = drive()
    root_id, drive_id = resolve_root(d, FOLDER_ID)

    token = load_token(d)
    if not token:
        token = get_start_token(d, drive_id)
        save_token(d, token)
        log.info("Initialized change token; next run will process deltas.")
        return

    parent_cache: Dict[str, Optional[List[str]]] = {}
    processed = 0
    next_token = None

    odoo_client = None
    to_odoo_invoice_name = None
    if ENABLE_ODOO:
        try:
            from odoo_attach import OdooClient
            from odoo_attach import to_odoo_invoice_name as _to_odoo_invoice_name

            odoo_client = OdooClient()
            to_odoo_invoice_name = _to_odoo_invoice_name
            log.info("Odoo integration enabled.")
        except Exception as exc:
            log.warning("Odoo integration requested but failed to init: %s", exc)

    while True:
        kwargs = {
            "pageToken": token,
            "fields": "nextPageToken,newStartPageToken,changes(fileId,removed,file)",
            "includeItemsFromAllDrives": True,
            "supportsAllDrives": True,
        }
        if drive_id:
            kwargs["driveId"] = drive_id

        response = d.changes().list(**kwargs).execute()

        for change in response.get("changes", []):
            file_id = change.get("fileId")
            removed = change.get("removed", False)
            file_object = change.get("file") or {}

            if removed or not file_object or file_object.get("trashed"):
                continue
            if file_object.get("mimeType") != PDF_MT:
                continue

            name = file_object.get("name", "")
            parents = file_object.get("parents") or []

            if RENAMED_RE.match(name or ""):
                if DEBUG:
                    log.info("SKIP %s already looks renamed", name)
                continue
            if not IMG_PDF_RE.match(name or ""):
                if DEBUG:
                    log.info("SKIP %s not IMG*.pdf", name)
                continue

            if not parents:
                metadata = file_meta(d, file_id, "id,name,parents,trashed")
                if metadata.get("trashed"):
                    continue
                name = metadata.get("name", name)
                parents = metadata.get("parents") or []

            if not is_under_root(d, parents, root_id, parent_cache):
                if DEBUG:
                    log.info("SKIP %s outside target tree", name)
                continue

            pdf_bytes = download_pdf_bytes(d, file_id)
            if not pdf_bytes:
                log.warning("SKIP %s no PDF bytes", name)
                continue

            image = rasterize_first_page(pdf_bytes)
            if image is None:
                log.warning("SKIP %s could not rasterize first page", name)
                continue

            invoice, confidence, source = extract_invoice_number_from_image(image)
            if not invoice:
                # Do not turn an unreadable scan into UNKNOWN.pdf. Keeping the IMG
                # name makes the exception visible and prevents a false attachment.
                log.warning(
                    "SKIP %s invoice number not confidently found "
                    "(best OCR confidence=%.1f)",
                    name,
                    confidence,
                )
                continue

            parent_id = parents[0] if parents else root_id
            new_name = unique_name_in_folder(d, parent_id, f"{invoice}.pdf")

            if new_name == name:
                if DEBUG:
                    log.info("SKIP %s name unchanged", name)
                continue

            rename_in_drive(d, file_id, new_name)
            log.info(
                "RENAMED %s -> %s (source=%s confidence=%.1f)",
                name,
                new_name,
                source,
                confidence,
            )
            processed += 1

            if ENABLE_ODOO and odoo_client and to_odoo_invoice_name:
                try:
                    odoo_number = to_odoo_invoice_name(invoice)
                    matches = odoo_client.search_customer_invoice_by_number(odoo_number)

                    if len(matches) == 1:
                        move_id = matches[0]
                        attachment_id = odoo_client.attach_pdf_to_move(
                            move_id,
                            new_name,
                            pdf_bytes,
                        )
                        log.info(
                            "ODOO ATTACHED %s to invoice=%s move_id=%s attachment_id=%s",
                            new_name,
                            odoo_number,
                            move_id,
                            attachment_id,
                        )
                    elif not matches:
                        log.warning(
                            "ODOO NO MATCH for invoice=%s (scan=%s) file=%s",
                            odoo_number,
                            invoice,
                            new_name,
                        )
                    else:
                        log.warning(
                            "ODOO MULTIPLE MATCHES for invoice=%s -> %s (file=%s)",
                            odoo_number,
                            matches,
                            new_name,
                        )
                except Exception as exc:
                    log.warning(
                        "ODOO ATTACH FAILED for %s (invoice=%s): %s",
                        new_name,
                        invoice,
                        exc,
                    )

        token = response.get("nextPageToken")
        if not token:
            next_token = response.get("newStartPageToken")
            break

    if next_token:
        save_token(d, next_token)

    log.info("Processed %d file(s) this run.", processed)


if __name__ == "__main__":
    main()
