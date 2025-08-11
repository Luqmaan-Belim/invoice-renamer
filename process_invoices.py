#!/usr/bin/env python3
"""
Invoice Renamer — GitHub Actions version (Drive-based)

ENV (set as GitHub Actions secrets or env):
  GDRIVE_SA_JSON   : stringified JSON of a Google service-account key
  GDRIVE_FOLDER_ID : ID of the target folder (not a shortcut)
  DEBUG_LIST       : "1" to print a few discovered files (optional)

Requires (same libs you already use):
  google-api-python-client, google-auth, google-auth-httplib2, google-auth-oauthlib
  pytesseract, pdf2image, opencv-python-headless, numpy, Pillow
  (Plus system packages: tesseract-ocr, poppler-utils)

Notes:
- Add the service-account email to the folder (My Drive) OR to the whole Shared Drive
  as Content manager, so it can see/rename files.
"""

import os, io, re, json, time, logging
from typing import List, Dict, Tuple, Optional

import numpy as np
import cv2
import pytesseract
from pdf2image import convert_from_bytes

from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account

# ---------- Logging ----------
log = logging.getLogger("invoice_renamer")
log.setLevel(logging.INFO)
_fmt = logging.Formatter("%(asctime)s - %(message)s")

# Log to stdout (so you see it in Actions) AND to file (optional)
console = logging.StreamHandler()
console.setFormatter(_fmt)
log.addHandler(console)

file_handler = logging.FileHandler("invoice_renamer.log")
file_handler.setFormatter(_fmt)
log.addHandler(file_handler)

# No Windows tesseract path in Actions; system tesseract is used.
# pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"  # not required normally

# ---------- Config ----------
FOLDER_ID = os.environ["GDRIVE_FOLDER_ID"].strip()
SA_JSON = json.loads(os.environ["GDRIVE_SA_JSON"])
DEBUG_LIST = os.environ.get("DEBUG_LIST", "0") == "1"

PDF_MT = "application/pdf"
FOLDER_MT = "application/vnd.google-apps.folder"
SHORTCUT_MT = "application/vnd.google-apps.shortcut"

RENAMED = re.compile(r"^\d{4}_?\d{1,6}(?:_\d+)?\.pdf$", re.I)


# ---------- Drive helpers ----------
def drive():
    creds = service_account.Credentials.from_service_account_info(
        SA_JSON, scopes=["https://www.googleapis.com/auth/drive"]
    )
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def resolve_folder(d, folder_id: str) -> Tuple[str, bool, Optional[str]]:
    """Return (folder_id, is_shared_drive, drive_id). Resolve shortcuts to the real folder."""
    meta = d.files().get(
        fileId=folder_id,
        fields="id,name,mimeType,driveId,shortcutDetails",
        supportsAllDrives=True,
    ).execute()

    if meta.get("mimeType") == SHORTCUT_MT:
        target = meta["shortcutDetails"]["targetId"]
        log.info("Resolved shortcut: %s -> %s", folder_id, target)
        meta = d.files().get(
            fileId=target,
            fields="id,name,mimeType,driveId",
            supportsAllDrives=True,
        ).execute()
        folder_id = meta["id"]

    is_shared = bool(meta.get("driveId"))
    return folder_id, is_shared, meta.get("driveId")


def list_pdfs_recursive(d, root_id: str, is_shared: bool, drive_id: Optional[str]) -> List[Dict]:
    """
    BFS through root + subfolders (+ folder shortcuts). Return items:
    {"id", "name", "parent"} for PDFs that are NOT already renamed.
    """
    queue = [root_id]
    seen = set()
    out: List[Dict] = []

    base_kwargs = dict(
        fields="nextPageToken, files(id,name,mimeType,parents,shortcutDetails)",
        pageSize=1000,
        orderBy="createdTime desc",
        supportsAllDrives=True,
        includeItemsFromAllDrives=True,
    )
    if is_shared and drive_id:
        base_kwargs.update(corpora="drive", driveId=drive_id)

    while queue:
        folder_id = queue.pop(0)
        if folder_id in seen:
            continue
        seen.add(folder_id)

        q = f"'{folder_id}' in parents and trashed=false"
        token = None
        while True:
            kwargs = dict(base_kwargs, q=q)
            if token:
                kwargs["pageToken"] = token
            resp = d.files().list(**kwargs).execute()

            for f in resp.get("files", []):
                mt = f.get("mimeType", "")
                if mt == SHORTCUT_MT:
                    sd = f.get("shortcutDetails", {})
                    tgt_id = sd.get("targetId")
                    tgt_mt = sd.get("targetMimeType")
                    if tgt_mt == FOLDER_MT and tgt_id:
                        queue.append(tgt_id)
                    continue
                if mt == FOLDER_MT:
                    queue.append(f["id"])
                    continue
                if mt == PDF_MT and not RENAMED.match(f.get("name", "")):
                    parent = (f.get("parents") or [folder_id])[0]
                    out.append({"id": f["id"], "name": f["name"], "parent": parent})

            token = resp.get("nextPageToken")
            if not token:
                break

    if DEBUG_LIST:
        log.info("DEBUG: unrenamed PDFs across tree: %d", len(out))
        for f in out[:12]:
            log.info("DEBUG: %s (parent=%s)", f["name"], f["parent"])
    return out


def unique_name_in_folder(d, folder_id: str, base_name: str) -> str:
    """Return a unique name inside the given folder."""
    name, n = base_name, 1
    while True:
        q = f"name = '{name}' and '{folder_id}' in parents and trashed=false"
        res = d.files().list(
            q=q, fields="files(id)", pageSize=1,
            supportsAllDrives=True, includeItemsFromAllDrives=True,
        ).execute()
        if not res.get("files"):
            return name
        root, ext = os.path.splitext(base_name)
        name = f"{root}_{n}{ext}"
        n += 1


def rename_in_drive(d, file_id: str, new_name: str):
    d.files().update(
        fileId=file_id, body={"name": new_name}, supportsAllDrives=True
    ).execute()


def download_first_page(d, file_id: str) -> Optional[np.ndarray]:
    """Download the PDF and rasterize only the first page (300 DPI)."""
    buf = io.BytesIO()
    req = d.files().get_media(fileId=file_id)
    dl = MediaIoBaseDownload(buf, req)
    done = False
    while not done:
        _, done = dl.next_chunk()
    buf.seek(0)
    imgs = convert_from_bytes(buf.getvalue(), dpi=300, first_page=1, last_page=1)
    return np.array(imgs[0]) if imgs else None


# ---------- Your OCR pipeline (unchanged) ----------
class InvoiceRenamer:
    def __init__(self):
        self.processed_ids = set()

    def process_and_rename(self, d, file_id: str, parent_id: str, name: str):
        if file_id in self.processed_ids:
            return
        self.processed_ids.add(file_id)

        try:
            log.info("Processing: %s (%s)", name, file_id)

            image = download_first_page(d, file_id)
            if image is None:
                log.warning("No image extracted from %s", name)
                return

            text, avg_confidence = self.ocr_with_confidence(image)
            log.info("OCR confidence: %.2f", avg_confidence)
            log.info("Extracted Text:\n%s", text)

            if avg_confidence < 40:
                log.info("Low OCR confidence, preprocessing and retrying...")
                image = self.preprocess_image(image)
                text, avg_confidence = self.ocr_with_confidence(image)
                log.info("Retried OCR confidence: %.2f", avg_confidence)
                log.info("Retried Extracted Text:\n%s", text)

            invoice_number = self.extract_invoice_number(text) or "UNKNOWN"

            new_name = unique_name_in_folder(d, parent_id, f"{invoice_number}.pdf")
            if new_name == name:
                log.info("Already correctly named; skipping.")
                return

            rename_in_drive(d, file_id, new_name)
            log.info("Renamed %s -> %s", name, new_name)

        finally:
            # In a batch run, just discard the id
            self.processed_ids.discard(file_id)

    def ocr_with_confidence(self, image):
        data = pytesseract.image_to_data(
            image, config="--psm 6 --oem 3", output_type=pytesseract.Output.DICT
        )
        text = " ".join(data.get("text", [])).strip()
        confidences = [c for c in data.get("conf", []) if isinstance(c, (int, float)) and c != -1]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        return text, avg_confidence

    def preprocess_image(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray = cv2.equalizeHist(gray)
        processed = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        return processed

    def extract_invoice_number(self, text):
        regex = r"(?:INV|NV)[^\d]*(\d{4}[^\d]*\d{1,6})"
        match = re.search(regex, text)
        if match:
            raw_invoice_number = match.group(1).strip()
            cleaned = self.clean_invoice_number(raw_invoice_number)
            log.info("Raw Invoice: %s, Cleaned Invoice: %s", raw_invoice_number, cleaned)
            return cleaned
        return None

    def clean_invoice_number(self, invoice_number):
        corrections = {
            "A": "4",
            " ": "",
            "/": "_",
            "O": "0",
            "I": "1",
        }
        for k, v in corrections.items():
            invoice_number = invoice_number.replace(k, v)
        invoice_number = re.sub(r"[^\d_]", "", invoice_number)
        parts = invoice_number.split("_")
        if len(parts) == 2 and len(parts[1]) < 6:
            invoice_number = f"{parts[0]}_{parts[1]}{'0' * (6 - len(parts[1]))}"
        log.info("Final Cleaned Invoice: %s", invoice_number)
        return invoice_number


# ---------- Batch entry point (replaces filesystem watcher) ----------
def run_drive_batch():
    d = drive()
    root_id, is_shared, drive_id = resolve_folder(d, FOLDER_ID)
    renamer = InvoiceRenamer()

    # Recursively gather PDFs across root + subfolders
    targets = list_pdfs_recursive(d, root_id, is_shared, drive_id)
    if not targets:
        log.info("No PDFs to process.")
        return

    log.info("Found %d PDF(s) across folder + subfolders", len(targets))
    for f in targets:
        renamer.process_and_rename(d, f["id"], f["parent"], f["name"])


if __name__ == "__main__":
    run_drive_batch()
