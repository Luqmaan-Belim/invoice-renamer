#!/usr/bin/env python3
"""
Invoice Renamer - recurring Google Drive sweep with optional Odoo attachment.

Every run recursively scans the configured Drive folder for outstanding IMG*.pdf
files. The first page is OCRed and renamed from an Odoo customer invoice number
such as INV/2026/042130 to 2026_042130.pdf. Files that cannot be processed remain
with their IMG name so the next run automatically retries them.
"""

import io
import json
import logging
import os
import re
from typing import Iterator, List, Optional, Tuple

import cv2
import fitz  # PyMuPDF
import numpy as np
import pytesseract
from google.oauth2 import service_account
from googleapiclient.discovery import build


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("invoice_renamer")


FOLDER_ID = os.environ["GDRIVE_FOLDER_ID"].strip()
SA_JSON = json.loads(os.environ["GDRIVE_SA_JSON"])
DEBUG = os.environ.get("DEBUG_LIST", "0") == "1"
ENABLE_ODOO = os.environ.get("ENABLE_ODOO", "0") == "1"

PDF_MT = "application/pdf"
FOLDER_MT = "application/vnd.google-apps.folder"
SHORTCUT_MT = "application/vnd.google-apps.shortcut"

IMG_PDF_RE = re.compile(r"^IMG.*\.pdf$", re.I)

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

EXPLICIT_INVOICE_RE = re.compile(
    rf"(?<![A-Z0-9])(?P<prefix>[I1L]\s*N\s*V|N\s*V)(?![A-Z0-9])"
    rf"[\s:#/\\|_.-]*{OCR_YEAR}{OCR_SEPARATOR}{OCR_SERIAL}",
    re.I,
)

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


def iter_outstanding_pdfs(d, root_id: str, drive_id: Optional[str]) -> Iterator[dict]:
    """Recursively yield IMG*.pdf files below root_id."""
    pending_folders = [root_id]
    visited_folders = set()

    while pending_folders:
        folder_id = pending_folders.pop()
        if folder_id in visited_folders:
            continue
        visited_folders.add(folder_id)

        page_token = None
        while True:
            kwargs = {
                "q": f"'{folder_id}' in parents and trashed=false",
                "fields": (
                    "nextPageToken,files("
                    "id,name,mimeType,parents,shortcutDetails(targetId,targetMimeType))"
                ),
                "pageSize": 1000,
                "supportsAllDrives": True,
                "includeItemsFromAllDrives": True,
            }
            if page_token:
                kwargs["pageToken"] = page_token
            if drive_id:
                kwargs["corpora"] = "drive"
                kwargs["driveId"] = drive_id

            response = d.files().list(**kwargs).execute()
            for item in response.get("files", []):
                mime_type = item.get("mimeType")
                name = item.get("name", "")

                if mime_type == FOLDER_MT:
                    pending_folders.append(item["id"])
                    continue

                if mime_type == SHORTCUT_MT:
                    if DEBUG:
                        log.info("SKIP shortcut %s", name)
                    continue

                if mime_type == PDF_MT and IMG_PDF_RE.match(name):
                    yield item

            page_token = response.get("nextPageToken")
            if not page_token:
                break


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


def process_file(d, file_object: dict, odoo_client=None, to_odoo_invoice_name=None) -> bool:
    """Process one outstanding IMG PDF. Return True only when it was renamed."""
    file_id = file_object["id"]
    name = file_object.get("name", "")
    parents = file_object.get("parents") or []

    try:
        pdf_bytes = download_pdf_bytes(d, file_id)
        if not pdf_bytes:
            log.warning("RETRY LATER %s: no PDF bytes", name)
            return False

        image = rasterize_first_page(pdf_bytes)
        if image is None:
            log.warning("RETRY LATER %s: could not rasterize first page", name)
            return False

        invoice, confidence, source = extract_invoice_number_from_image(image)
        if not invoice:
            log.warning(
                "RETRY LATER %s: invoice number not confidently found "
                "(best OCR confidence=%.1f)",
                name,
                confidence,
            )
            return False

        parent_id = parents[0] if parents else FOLDER_ID
        new_name = unique_name_in_folder(d, parent_id, f"{invoice}.pdf")
        rename_in_drive(d, file_id, new_name)
        log.info(
            "RENAMED %s -> %s (source=%s confidence=%.1f)",
            name,
            new_name,
            source,
            confidence,
        )

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

        return True

    except Exception as exc:
        log.exception("RETRY LATER %s: processing failed: %s", name, exc)
        return False


def main():
    d = drive()
    root_id, drive_id = resolve_root(d, FOLDER_ID)

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

    discovered = 0
    renamed = 0
    failed = 0

    for file_object in iter_outstanding_pdfs(d, root_id, drive_id):
        discovered += 1
        if process_file(d, file_object, odoo_client, to_odoo_invoice_name):
            renamed += 1
        else:
            failed += 1

    log.info(
        "Sweep complete: outstanding=%d renamed=%d retry_later=%d",
        discovered,
        renamed,
        failed,
    )


if __name__ == "__main__":
    main()
