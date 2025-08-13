#!/usr/bin/env python3
"""
Invoice Renamer — GitHub Actions (Drive Changes API)

- Polls Google Drive "changes" using a page token so we only see NEW/CHANGED files.
- Filters to PDFs that live under your target folder (including subfolders).
- Runs Tesseract OCR on the first page and renames the file in its own folder.
- Works with My Drive and Shared Drives, and resolves folder shortcuts for the root.

ENV (GitHub Secrets/Env):
  GDRIVE_SA_JSON   : JSON of a Google service-account key (as one string)
  GDRIVE_FOLDER_ID : ID of the *actual* target folder (not a shortcut)
  DEBUG_LIST       : "1" to print debug info (optional)

Requires (pip):
  google-api-python-client, google-auth, google-auth-httplib2, google-auth-oauthlib
  pytesseract, pdf2image, opencv-python-headless, numpy, Pillow

System packages on runner:
  tesseract-ocr, poppler-utils
"""

import os, io, re, json, logging, pathlib
from typing import Optional, Tuple

import numpy as np
import cv2
import pytesseract
import fitz  # PyMuPDF

from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account

# ---------- Config & logging ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("invoice_renamer")

FOLDER_ID = os.environ["GDRIVE_FOLDER_ID"].strip()
SA_JSON = json.loads(os.environ["GDRIVE_SA_JSON"])
DEBUG = os.environ.get("DEBUG_LIST", "0") == "1"

TOKEN_PATH = pathlib.Path(".drive_change_token")  # persisted via Actions cache
PDF_MT = "application/pdf"
FOLDER_MT = "application/vnd.google-apps.folder"
SHORTCUT_MT = "application/vnd.google-apps.shortcut"
RENAMED_RE = re.compile(r"^\d{4}_?\d{1,6}(?:_\d+)?\.pdf$", re.I)


# ---------- Drive helpers ----------
def drive():
    creds = service_account.Credentials.from_service_account_info(
        SA_JSON, scopes=["https://www.googleapis.com/auth/drive"]
    )
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def resolve_root(d, folder_id: str) -> Tuple[str, bool, Optional[str]]:
    """Return (real_root_id, is_shared_drive, drive_id). Resolve if given ID is a shortcut."""
    meta = d.files().get(
        fileId=folder_id,
        fields="id,name,mimeType,driveId,shortcutDetails",
        supportsAllDrives=True,
    ).execute()
    if meta.get("mimeType") == SHORTCUT_MT:
        tgt = meta["shortcutDetails"]["targetId"]
        log.info("Resolved shortcut root: %s -> %s", folder_id, tgt)
        meta = d.files().get(
            fileId=tgt,
            fields="id,name,mimeType,driveId",
            supportsAllDrives=True,
        ).execute()
    return meta["id"], bool(meta.get("driveId")), meta.get("driveId")


def get_start_token(d, is_shared: bool, drive_id: Optional[str]) -> str:
    """Get a startPageToken for My Drive or a specific Shared Drive."""
    if is_shared and drive_id:
        resp = d.changes().getStartPageToken(driveId=drive_id, supportsAllDrives=True).execute()
    else:
        resp = d.changes().getStartPageToken().execute()
    return resp["startPageToken"]


def load_token() -> Optional[str]:
    try:
        return TOKEN_PATH.read_text().strip() or None
    except FileNotFoundError:
        return None


def save_token(token: str):
    TOKEN_PATH.write_text(token)


def file_meta(d, fid: str):
    """Fetch minimal metadata we need for filtering."""
    return d.files().get(
        fileId=fid,
        fields="id,name,mimeType,parents,driveId,trashed",
        supportsAllDrives=True,
    ).execute()


def is_under_root(d, file_parents, root_id: str, cache) -> bool:
    """
    Ascend parents until we hit root_id or top.
    Cache parent->parents lookups to reduce API calls when multiple files share ancestry.
    """
    # Some files may have multiple parents (rare). Check any chain.
    if not file_parents:
        return False
    stack = list(file_parents)
    while stack:
        pid = stack.pop()
        if pid == root_id:
            return True
        if pid in cache:
            # cached: None means top/not under root; list means its parents
            parents = cache[pid]
        else:
            try:
                md = d.files().get(
                    fileId=pid,
                    fields="id, parents",
                    supportsAllDrives=True,
                ).execute()
                parents = md.get("parents") or []
            except Exception:
                parents = []
            cache[pid] = parents if parents else None
        if parents:
            stack.extend(parents)
    return False


def download_first_page(drive, file_id):
    """
    Download PDF bytes and rasterize first page at 300 DPI using PyMuPDF.
    Returns an RGB NumPy array (H,W,3) suitable for OpenCV/Tesseract, or None.
    """
    import io
    from googleapiclient.http import MediaIoBaseDownload

    buf = io.BytesIO()
    req = drive.files().get_media(fileId=file_id)
    dl = MediaIoBaseDownload(buf, req)
    done = False
    while not done:
        _, done = dl.next_chunk()
    pdf_bytes = buf.getvalue()
    if not pdf_bytes:
        return None

    # DPI -> zoom: 72pt = 1.0 zoom. For ~300 DPI: 300/72 ≈ 4.1667
    zoom = 300 / 72.0
    mat = fitz.Matrix(zoom, zoom)

    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        if doc.page_count == 0:
            return None
        page = doc.load_page(0)
        pix = page.get_pixmap(matrix=mat, alpha=False)  # RGB
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        return img


def rename_in_drive(d, fid: str, new_name: str):
    d.files().update(fileId=fid, body={"name": new_name}, supportsAllDrives=True).execute()


# ---------- Your OCR logic (unchanged) ----------
def ocr_with_confidence(image: np.ndarray):
    data = pytesseract.image_to_data(
        image, config="--psm 6 --oem 3", output_type=pytesseract.Output.DICT
    )
    text = " ".join([t for t in data.get("text", []) if t]).strip()
    confs = [c for c in data.get("conf", []) if isinstance(c, (int, float)) and c != -1]
    avg = sum(confs) / len(confs) if confs else 0.0
    return text, avg


def preprocess_image(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 11, 2)


def extract_invoice_number(text: str) -> Optional[str]:
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


def unique_name_in_folder(d, folder_id: str, base: str) -> str:
    name, n = base, 1
    while True:
        q = f"name = '{name}' and '{folder_id}' in parents and trashed=false"
        res = d.files().list(
            q=q, fields="files(id)", pageSize=1,
            supportsAllDrives=True, includeItemsFromAllDrives=True,
        ).execute()
        if not res.get("files"):
            return name
        root, ext = os.path.splitext(base)
        name = f"{root}_{n}{ext}"
        n += 1


# ---------- Main (changes-driven) ----------
def main():
    d = drive()
    root_id, is_shared, drive_id = resolve_root(d, FOLDER_ID)

    token = load_token()
    if not token:
        token = get_start_token(d, is_shared, drive_id)
        save_token(token)
        log.info("Initialized change token.")
        return  # First run just initializes; next run will process deltas

    parent_cache = {}  # folder_id -> parents list (or None if top/not under root)
    processed = 0
    next_token = None

    # List changes since token; may be multiple pages
    while True:
        kwargs = dict(pageToken=token, fields="nextPageToken,newStartPageToken,changes(fileId,removed,file) ",
                      includeItemsFromAllDrives=True, supportsAllDrives=True)
        if is_shared and drive_id:
            kwargs.update(driveId=drive_id)
        resp = d.changes().list(**kwargs).execute()

        for ch in resp.get("changes", []):
            fid = ch.get("fileId")
            removed = ch.get("removed", False)
            fobj = ch.get("file") or {}

            # Skip deletions or items with no file object
            if removed or not fobj:
                continue

            # Quick filters
            if fobj.get("trashed"):
                continue
            if fobj.get("mimeType") != PDF_MT:
                continue
            name = fobj.get("name", "")
            if RENAMED_RE.match(name or ""):
                continue

            # Ensure file is under our root (ascend parents)
            # If file doesn't have parents in 'file', fetch fresh metadata
            parents = fobj.get("parents")
            if not parents:
                meta = file_meta(d, fid)
                if meta.get("trashed"):
                    continue
                parents = meta.get("parents") or []
                name = meta.get("name", name)

            if not is_under_root(d, parents, root_id, parent_cache):
                continue

            # Download first page and OCR
            img = download_first_page(d, fid)
            if img is None:
                log.warning("No image for %s; skipping", name)
                continue

            text, conf = ocr_with_confidence(img)
            if conf < 40:
                img2 = preprocess_image(img)
                text, conf = ocr_with_confidence(img2)
            inv = extract_invoice_number(text) or "UNKNOWN"

            # Use the nearest parent folder (first parent)
            parent_id = parents[0] if parents else root_id
            new_name = unique_name_in_folder(d, parent_id, f"{inv}.pdf")
            if new_name != name:
                rename_in_drive(d, fid, new_name)
                log.info("Renamed: %s -> %s", name, new_name)
                processed += 1

        token = resp.get("nextPageToken")
        if not token:
            next_token = resp.get("newStartPageToken")
            break

    # Save the token for the next run
    if next_token:
        save_token(next_token)

    if DEBUG:
        log.info("Done. Processed %d file(s) this run.", processed)


if __name__ == "__main__":
    main()
