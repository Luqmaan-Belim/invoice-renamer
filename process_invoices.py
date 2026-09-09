#!/usr/bin/env python3
"""
Invoice Renamer - recurring Google Drive sweep with Odoo attachment support.

Every run recursively scans the configured Drive folder for outstanding IMG*.pdf
files. It recognises Fresh Bake Tax Invoices and Credit Notes from the fixed
number row near the top-left of page 1. Files that cannot be safely identified or
attached remain with their IMG name so the next run retries them.
"""

import io
import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

import cv2
import fitz  # PyMuPDF
import numpy as np
import pytesseract
from google.oauth2 import service_account
from googleapiclient.discovery import build


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("invoice_renamer")

FOLDER_ID = os.environ["GDRIVE_FOLDER_ID"].strip()
SA_JSON = json.loads(os.environ["GDRIVE_SA_JSON"])
DEBUG = os.environ.get("DEBUG_LIST", "0") == "1"
ENABLE_ODOO = os.environ.get("ENABLE_ODOO", "0") == "1"

PDF_MT = "application/pdf"
FOLDER_MT = "application/vnd.google-apps.folder"
SHORTCUT_MT = "application/vnd.google-apps.shortcut"
IMG_PDF_RE = re.compile(r"^IMG.*\.pdf$", re.I)

OCR_DIGIT_TRANSLATION = str.maketrans({
    "O": "0", "Q": "0", "D": "0", "I": "1", "L": "1",
    "Z": "2", "S": "5", "G": "6", "B": "8",
})
OCR_DIGIT = r"[0-9OQDISBGLZ]"
OCR_YEAR = rf"(?P<year>(?:{OCR_DIGIT}\s*){{4}})"
OCR_SERIAL = rf"(?P<serial>(?:{OCR_DIGIT}\s*){{1,6}})(?!{OCR_DIGIT})"
OCR_SEPARATOR = r"\s*[/\\|_.-]\s*"
OCR_INVOICE_PREFIX = r"(?:[I1L]\s*N\s*V|N\s*V)"
OCR_CREDIT_PREFIX = r"R\s*(?:[I1L]\s*)?N\s*V"
OCR_NUMBER_LABEL = r"(?:N[UO0]M(?:BER|8ER)|NUMBER|NO\.?|NR\.?)"

CREDIT_LABELLED_RE = re.compile(
    rf"CREDIT\s+N[O0]TE\s+{OCR_NUMBER_LABEL}\s*[:#-]?\s*"
    rf"(?:{OCR_CREDIT_PREFIX}[\s:#/\\|_.-]*)?{OCR_YEAR}{OCR_SEPARATOR}{OCR_SERIAL}",
    re.I,
)
CREDIT_EXPLICIT_RE = re.compile(
    rf"(?<![A-Z0-9]){OCR_CREDIT_PREFIX}(?![A-Z0-9])"
    rf"[\s:#/\\|_.-]*{OCR_YEAR}{OCR_SEPARATOR}{OCR_SERIAL}",
    re.I,
)
INVOICE_LABELLED_RE = re.compile(
    rf"(?:TAX\s+)?INVOICE\s+{OCR_NUMBER_LABEL}\s*[:#-]?\s*"
    rf"(?:{OCR_INVOICE_PREFIX}[\s:#/\\|_.-]*)?{OCR_YEAR}{OCR_SEPARATOR}{OCR_SERIAL}",
    re.I,
)
INVOICE_EXPLICIT_RE = re.compile(
    rf"(?<![A-Z0-9]){OCR_INVOICE_PREFIX}(?![A-Z0-9])"
    rf"[\s:#/\\|_.-]*{OCR_YEAR}{OCR_SEPARATOR}{OCR_SERIAL}",
    re.I,
)
CREDIT_CONTEXT_RE = re.compile(r"CRED[I1]T\s+N[O0]TE|REVERSAL\s+OF", re.I)


@dataclass(frozen=True)
class DocumentNumber:
    kind: str
    year: str
    serial: str

    @property
    def prefix(self) -> str:
        return "RINV" if self.kind == "credit_note" else "INV"

    @property
    def odoo_number(self) -> str:
        return f"{self.prefix}/{self.year}/{self.serial}"

    @property
    def move_type(self) -> str:
        return "out_refund" if self.kind == "credit_note" else "out_invoice"

    @property
    def filename_stem(self) -> str:
        if self.kind == "credit_note":
            return f"RINV_{self.year}_{self.serial}"
        return f"{self.year}_{self.serial}"


def drive():
    creds = service_account.Credentials.from_service_account_info(
        SA_JSON, scopes=["https://www.googleapis.com/auth/drive"]
    )
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def resolve_root(d, folder_id: str) -> Tuple[str, Optional[str]]:
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
                elif mime_type == SHORTCUT_MT:
                    if DEBUG:
                        log.info("SKIP shortcut %s", name)
                elif mime_type == PDF_MT and IMG_PDF_RE.match(name):
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
    return buffer.getvalue() or None


def extract_first_page_text(pdf_bytes: bytes) -> str:
    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
            if document.page_count == 0:
                return ""
            return document.load_page(0).get_text("text") or ""
    except Exception:
        return ""


def rasterize_first_page(pdf_bytes: bytes) -> Optional[np.ndarray]:
    matrix = fitz.Matrix(300.0 / 72.0, 300.0 / 72.0)
    with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
        if document.page_count == 0:
            return None
        pixmap = document.load_page(0).get_pixmap(matrix=matrix, alpha=False)
        return np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
            pixmap.height, pixmap.width, 3
        )


def ocr_with_confidence(image: np.ndarray, psm: int = 6) -> Tuple[str, float]:
    data = pytesseract.image_to_data(
        image,
        config=f"--psm {psm} --oem 3",
        output_type=pytesseract.Output.DICT,
    )
    text = " ".join(str(v) for v in data.get("text", []) if str(v).strip())
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
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def adaptive_preprocess(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 12
    )


def rotate_for_ocr(image: np.ndarray, angle: float) -> np.ndarray:
    height, width = image.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        image, matrix, (width, height),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=255,
    )


def _normalise_ocr_digits(value: str) -> str:
    translated = value.upper().translate(OCR_DIGIT_TRANSLATION)
    return re.sub(r"\D", "", translated)


def _document_from_match(match: re.Match, kind: str) -> Optional[DocumentNumber]:
    year = _normalise_ocr_digits(match.group("year"))
    serial = _normalise_ocr_digits(match.group("serial"))
    if len(year) != 4 or not year.startswith("20"):
        return None
    if not 1 <= len(serial) <= 6:
        return None
    if kind == "credit_note":
        serial = serial[:5].zfill(5)
    else:
        serial = serial[:6].zfill(6)
    return DocumentNumber(kind=kind, year=year, serial=serial)


def extract_document_number(text: str) -> Optional[DocumentNumber]:
    normalised = " ".join(text.upper().split())

    # Credit note must win. Its page also contains the reversed INV/... reference.
    for pattern in (CREDIT_LABELLED_RE, CREDIT_EXPLICIT_RE):
        for match in pattern.finditer(normalised):
            document = _document_from_match(match, "credit_note")
            if document:
                return document

    # If the page clearly looks like a credit note but its RINV number is unreadable,
    # fail safely instead of attaching it to the INV/... shown in "Reversal of".
    if CREDIT_CONTEXT_RE.search(normalised):
        return None

    for pattern in (INVOICE_LABELLED_RE, INVOICE_EXPLICIT_RE):
        for match in pattern.finditer(normalised):
            document = _document_from_match(match, "invoice")
            if document:
                return document
    return None


def extract_invoice_number(text: str) -> Optional[str]:
    """Backward-compatible helper used by existing tests."""
    document = extract_document_number(text)
    return document.filename_stem if document else None


def _crop(image: np.ndarray, y1: float, y2: float, x1: float, x2: float) -> np.ndarray:
    height, width = image.shape[:2]
    return image[
        max(0, int(height * y1)):min(height, int(height * y2)),
        max(0, int(width * x1)):min(width, int(width * x2)),
    ]


def extract_document_number_from_image(
    image: np.ndarray,
) -> Tuple[Optional[DocumentNumber], float, str]:
    """Read the fixed Fresh Bake number row, with scan-friendly fallbacks."""
    number_row = _crop(image, 0.10, 0.18, 0.03, 0.53)
    header_block = _crop(image, 0.07, 0.22, 0.02, 0.60)
    upper_left_wide = _crop(image, 0.04, 0.27, 0.00, 0.72)

    attempts = (
        ("number-row", number_row, 6),
        ("number-row-enhanced", preprocess_image(number_row), 6),
        ("number-row-adaptive", adaptive_preprocess(number_row), 6),
        ("header-block", header_block, 6),
        ("header-block-enhanced", preprocess_image(header_block), 11),
        ("header-block-rotated-left", rotate_for_ocr(preprocess_image(header_block), -2.0), 11),
        ("header-block-rotated-right", rotate_for_ocr(preprocess_image(header_block), 2.0), 11),
        ("upper-left-wide", preprocess_image(upper_left_wide), 11),
    )

    best_confidence = 0.0
    for source, attempt_image, psm in attempts:
        text, confidence = ocr_with_confidence(attempt_image, psm=psm)
        best_confidence = max(best_confidence, confidence)
        document = extract_document_number(text)
        if document:
            if DEBUG:
                log.info(
                    "OCR MATCH document=%s kind=%s source=%s confidence=%.1f text=%r",
                    document.odoo_number, document.kind, source, confidence, text,
                )
            return document, confidence, source
    return None, best_confidence, "no-match"


def unique_name_in_folder(d, folder_id: str, base: str) -> str:
    name = base
    suffix = 1
    while True:
        response = d.files().list(
            q=f"name='{name}' and '{folder_id}' in parents and trashed=false",
            fields="files(id)", pageSize=1, supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        ).execute()
        if not response.get("files"):
            return name
        root, extension = os.path.splitext(base)
        name = f"{root}_{suffix}{extension}"
        suffix += 1


def rename_in_drive(d, file_id: str, new_name: str):
    d.files().update(
        fileId=file_id, body={"name": new_name}, supportsAllDrives=True
    ).execute()


def process_file(d, file_object: dict, odoo_client=None) -> bool:
    file_id = file_object["id"]
    name = file_object.get("name", "")
    parents = file_object.get("parents") or []

    try:
        pdf_bytes = download_pdf_bytes(d, file_id)
        if not pdf_bytes:
            log.warning("RETRY LATER %s: no PDF bytes", name)
            return False

        document = extract_document_number(extract_first_page_text(pdf_bytes))
        confidence = 100.0 if document else 0.0
        source = "pdf-text" if document else ""

        if not document:
            image = rasterize_first_page(pdf_bytes)
            if image is None:
                log.warning("RETRY LATER %s: could not rasterize first page", name)
                return False
            document, confidence, source = extract_document_number_from_image(image)

        if not document:
            log.warning(
                "RETRY LATER %s: invoice/credit-note number not confidently found "
                "(best OCR confidence=%.1f)", name, confidence,
            )
            return False

        parent_id = parents[0] if parents else FOLDER_ID
        new_name = unique_name_in_folder(d, parent_id, f"{document.filename_stem}.pdf")

        if ENABLE_ODOO and odoo_client:
            matches = odoo_client.search_customer_document_by_number(
                document.odoo_number, document.move_type
            )
            if not matches:
                log.warning(
                    "RETRY LATER %s: ODOO NO MATCH for %s (%s)",
                    name, document.odoo_number, document.move_type,
                )
                return False

            if len(matches) > 1:
                log.warning(
                    "ODOO MULTIPLE MATCHES for %s -> %s; attaching to all",
                    document.odoo_number, matches,
                )

            for move_id in matches:
                attachment_id, created = odoo_client.ensure_pdf_attachment(
                    move_id, new_name, pdf_bytes
                )
                if created:
                    log.info(
                        "ODOO ATTACHED %s to %s move_id=%s attachment_id=%s",
                        new_name, document.odoo_number, move_id, attachment_id,
                    )
                else:
                    log.info(
                        "ODOO ATTACHMENT ALREADY EXISTS %s on %s move_id=%s attachment_id=%s",
                        new_name, document.odoo_number, move_id, attachment_id,
                    )

        rename_in_drive(d, file_id, new_name)
        log.info(
            "RENAMED %s -> %s (document=%s source=%s confidence=%.1f)",
            name, new_name, document.odoo_number, source, confidence,
        )
        return True

    except Exception as exc:
        log.exception("RETRY LATER %s: processing failed: %s", name, exc)
        return False


def main():
    d = drive()
    root_id, drive_id = resolve_root(d, FOLDER_ID)

    odoo_client = None
    if ENABLE_ODOO:
        try:
            from odoo_attach import OdooClient
            odoo_client = OdooClient()
            log.info("Odoo integration enabled.")
        except Exception as exc:
            log.warning("Odoo integration requested but failed to init: %s", exc)

    discovered = renamed = failed = 0
    for file_object in iter_outstanding_pdfs(d, root_id, drive_id):
        discovered += 1
        if process_file(d, file_object, odoo_client):
            renamed += 1
        else:
            failed += 1

    log.info(
        "Sweep complete: outstanding=%d renamed=%d retry_later=%d",
        discovered, renamed, failed,
    )


if __name__ == "__main__":
    main()
