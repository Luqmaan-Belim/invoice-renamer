#!/usr/bin/env python3
"""
Invoice Renamer for Google Drive (GitHub Actions)
-------------------------------------------------
- Recursively scans a Drive folder (and subfolders) for PDFs.
- OCRs the first page using Tesseract and extracts an invoice number.
- Renames each PDF in-place within its own folder.
- Supports My Drive and Shared Drives; resolves folder shortcuts.

ENV:
  - GDRIVE_SA_JSON   : JSON of a Google service account key (as a single string)
  - GDRIVE_FOLDER_ID : ID of the *actual* target folder (not a shortcut)
  - DEBUG_LIST       : "1" to print discovered files/folders (optional)

Permissions:
  - Share the folder (or entire Shared Drive) with the service account as
    "Content manager". For Shared Drives, adding at the drive level is safest.
"""

import os, io, re, json, logging
from typing import List, Dict, Tuple, Optional

from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account

from pdf2image import convert_from_bytes
import pytesseract
import numpy as np
import cv2

# ---------- Settings & Logging ----------

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

FOLDER_ID = os.environ["GDRIVE_FOLDER_ID"].strip()
SA_JSON = json.loads(os.environ["GDRIVE_SA_JSON"])
SCOPES = ["https://www.googleapis.com/auth/drive"]
DEBUG_LIST = os.environ.get("DEBUG_LIST", "0") == "1"

PDF_MT = "application/pdf"
FOLDER_MT = "application/vnd.google-apps.folder"
SHORTCUT_MT = "application/vnd.google-apps.shortcut"

RENAMED = re.compile(r"^\d{4}_?\d{1,6}(?:_\d+)?\.pdf$", re.I)


# ---------- Drive helpers ----------

def drive():
    creds = service_account.Credentials.from_service_account_info(SA_JSON, scopes=SCOPES)
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def resolve_folder(d, folder_id: str) -> Tuple[str, bool, Optional[str]]:
    """
    Returns (actual_folder_id, is_shared_drive, drive_id_or_None).
    Also resolves if the supplied ID is actually a *shortcut* to a folder.
    """
    meta = d.files().get(
        fileId=folder_id,
        fields="id,name,mimeType,driveId,shortcutDetails",
        supportsAllDrives=True,
    ).execute()

    if meta.get("mimeType") == SHORTCUT_MT:
        target = meta["shortcutDetails"]["targetId"]
        logging.info("Resolved shortcut: %s -> %s", folder_id, target)
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
    Breadth-first traversal starting at root_id.
    Collects PDFs that are not already renamed.
    Returns dicts: {"id", "name", "parent"}.
    Follows folder shortcuts.
    """
    queue = [root_id]
    seen = set()
    out: List[Dict] = []

    base_kwargs = dict(
        fields="nextPageToken, files(id,name,mimeType,parents,driveId,shortcutDetails)",
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
                # Follow *folder* shortcuts
                if mt == SHORTCUT_MT:
                    sd = f.get("shortcutDetails", {})
                    tgt_mt = sd.get("targetMimeType")
                    tgt_id = sd.get("targetId")
                    if tgt_mt == FOLDER_MT and tgt_id:
                        queue.append(tgt_id)
                    continue

                if mt == FOLDER_MT:
                    queue.append(f["id"])
                    continue

                if mt == PDF_MT:
                    name = f.get("name", "")
                    if not RENAMED.match(name):
                        parent = (f.get("parents") or [folder_id])[0]
                        out.append({"id": f["id"], "name": name, "parent": parent})

            token = resp.get("nextPageToken")
            if not token:
                break

    if DEBUG_LIST:
        logging.info("DEBUG: total unrenamed PDFs found across tree: %d", len(out))
        for f in out[:12]:
            logging.info("DEBUG: %s (parent=%s)", f["name"], f["parent"])
    return out


def unique_name_in_folder(d, folder_id: str, base_name: str) -> str:
    """
    Ensure filename is unique within *that parent folder*.
    """
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


def rename(d, file_id: str, new_name: str):
    d.files().update(fileId=file_id, body={"name": new_name}, supportsAllDrives=True).execute()


# ---------- OCR helpers ----------

def download_first_page(d, file_id: str) -> Optional[np.ndarray]:
    """
    Downloads the PDF and rasterizes only the first page at 300 DPI.
    """
    buf = io.BytesIO()
    req = d.files().get_media(fileId=file_id)
    dl = MediaIoBaseDownload(buf, req)
    done = False
    while not done:
        _, done = dl.next_chunk()
    buf.seek(0)
    images = convert_from_bytes(buf.getvalue(), dpi=300, first_page=1, last_page=1)
    return np.array(images[0]) if images else None


def ocr_with_conf(image: np.ndarray) -> Tuple[str, float]:
    data = pytesseract.image_to_data(
        image, config="--psm 6 --oem 3", output_type=pytesseract.Output.DICT
    )
    text = " ".join([t for t in data.get("text", []) if t]).strip()
    confs = [c for c in data.get("conf", []) if isinstance(c, (int, float)) and c != -1]
    avg = sum(confs) / len(confs) if confs else 0.0
    return text, avg


def preprocess(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 11, 2)


def extract_invoice(text: str) -> Optional[str]:
    m = re.search(r"(?:INV|NV)[^\d]*(\d{4}[^\d]*\d{1,6})", text, flags=re.I)
    if not m:
        return None
    s = m.group(1).strip()
    for k, v in {"A": "4", " ": "", "/": "_", "O": "0", "I": "1"}.items():
        s = s.replace(k, v)
    s = re.sub(r"[^\d_]", "", s)
    parts = s.split("_")
    if len(parts) == 2 and len(parts[1]) < 6:
        parts[1] = parts[1].ljust(6, "0")
        s = "_".join(parts)
    return s


# ---------- Main ----------

def main():
    d = drive()
    root_id, is_shared, drive_id = resolve_folder(d, FOLDER_ID)

    targets = list_pdfs_recursive(d, root_id, is_shared, drive_id)
    if not targets:
        logging.info("No PDFs to process.")
        return

    logging.info("Found %d PDF(s) across folder + subfolders", len(targets))

    for f in targets:
        fid, fname, parent = f["id"], f["name"], f["parent"]
        logging.info("Processing: %s", fname)

        img = download_first_page(d, fid)
        if img is None:
            logging.warning("No image from PDF; skipping.")
            continue

        text, conf = ocr_with_conf(img)
        if conf < 40:
            text, conf = ocr_with_conf(preprocess(img))
        logging.info("OCR avg confidence: %.2f", conf)

        inv = extract_invoice(text) or "UNKNOWN"
        new_name = unique_name_in_folder(d, parent, f"{inv}.pdf")
        if new_name == fname:
            logging.info("Already correct; skipping.")
            continue

        rename(d, fid, new_name)
        logging.info("Renamed to: %s", new_name)


if __name__ == "__main__":
    main()
