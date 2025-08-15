#!/usr/bin/env python3
"""
Invoice Renamer — Rename ANY PDF whose name starts with 'IMG*' (case-insensitive)

• Scans your target Drive root (including all subfolders) for PDFs named 'IMG*.pdf'.
• Renames them based on OCR (first page). If no invoice number -> 'UNKNOWN_N.pdf'.
• Skips items already final-looking (2025_123456.pdf or UNKNOWN_7.pdf) and stamps
  appProperties.lb_renamed=1 so we don't touch them again.
• Processes up to MAX_FILES_PER_RUN per job, newest first.

Env (GitHub Actions → env/secrets):
  GDRIVE_SA_JSON, GDRIVE_FOLDER_ID  (required)
  MAX_FILES_PER_RUN  (default 100)  # how many IMG*.pdf to handle per run
  DEBUG_LIST         ("1" for verbose)

Requires (pip):
  google-api-python-client, google-auth, google-auth-httplib2, google-auth-oauthlib
  pytesseract, PyMuPDF, opencv-python-headless, numpy, Pillow
"""

import os, io, re, json, time, logging
from typing import Optional, Tuple, Dict, List
from datetime import datetime, timezone

import numpy as np
import cv2
import pytesseract
import fitz  # PyMuPDF

from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account

# ---------- config ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("invoice_renamer")

FOLDER_ID = os.environ["GDRIVE_FOLDER_ID"].strip()
SA_JSON = json.loads(os.environ["GDRIVE_SA_JSON"])
DEBUG = os.environ.get("DEBUG_LIST", "0") == "1"
MAX_FILES_PER_RUN = int(os.environ.get("MAX_FILES_PER_RUN", "100"))

PDF_MT = "application/pdf"
FOLDER_MT = "application/vnd.google-apps.folder"
SHORTCUT_MT = "application/vnd.google-apps.shortcut"

RENAMED_RE = re.compile(r"^\d{4}_?\d{1,6}(?:_\d+)?\.pdf$", re.I)
UNKNOWN_RE = re.compile(r"^UNKNOWN(?:_\d+)?\.pdf$", re.I)

# ---------- helpers ----------
def drive():
    creds = service_account.Credentials.from_service_account_info(
        SA_JSON, scopes=["https://www.googleapis.com/auth/drive"]
    )
    return build("drive", "v3", credentials=creds, cache_discovery=False)

def resolve_root(d, folder_id: str) -> Tuple[str, bool, Optional[str]]:
    """Resolve if given ID is a shortcut; return (real_root_id, is_shared_drive, drive_id)."""
    meta = d.files().get(
        fileId=folder_id,
        fields="id,name,mimeType,driveId,shortcutDetails",
        supportsAllDrives=True,
    ).execute()
    if meta.get("mimeType") == SHORTCUT_MT:
        tgt = meta["shortcutDetails"]["targetId"]
        log.info("Resolved shortcut root: %s -> %s", folder_id, tgt)
        meta = d.files().get(
            fileId=tgt, fields="id,name,mimeType,driveId", supportsAllDrives=True
        ).execute()
    if DEBUG:
        log.info("Root: %s (%s) SharedDrive=%s driveId=%s",
                 meta.get("name"), meta.get("id"), bool(meta.get("driveId")), meta.get("driveId"))
    return meta["id"], bool(meta.get("driveId")), meta.get("driveId")

def file_meta(d, fid: str) -> Dict:
    return d.files().get(
        fileId=fid,
        fields=("id,name,mimeType,parents,driveId,trashed,appProperties,"
                "createdTime,modifiedTime,capabilities(canRename,canEdit)"),
        supportsAllDrives=True,
    ).execute()

def is_under_root(d, parents: List[str], root_id: str, cache: Dict[str, Optional[List[str]]]) -> bool:
    """Ascend parents until we hit root_id; cache parent->parents lookups."""
    if not parents:
        return False
    stack = list(parents)
    while stack:
        pid = stack.pop()
        if pid == root_id:
            return True
        if pid in cache:
            p2 = cache[pid]
        else:
            try:
                md = d.files().get(fileId=pid, fields="id,parents", supportsAllDrives=True).execute()
                p2 = md.get("parents") or []
            except Exception:
                p2 = []
            cache[pid] = p2 if p2 else None
        if p2:
            stack.extend(p2)
    return False

def download_first_page(d, file_id: str) -> Optional[np.ndarray]:
    """Return RGB image (H,W,3) of first page at ~300 DPI using PyMuPDF."""
    buf = io.BytesIO()
    req = d.files().get_media(fileId=file_id)
    dl = MediaIoBaseDownload(buf, req)
    done = False
    while not done:
        _, done = dl.next_chunk()
    pdf_bytes = buf.getvalue()
    if not pdf_bytes:
        return None
    zoom = 300 / 72.0
    mat = fitz.Matrix(zoom, zoom)
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        if doc.page_count == 0:
            return None
        page = doc.load_page(0)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        return img

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
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

def extract_invoice_number(text: str) -> Optional[str]:
    m = re.search(r"(?:INV|NV)[^\d]*(\d{4}[^\d]*\d{1,6})", text, flags=re.I)
    if not m:
        return None
    s = m.group(1).strip()
    # common OCR normalizations
    for k, v in {"A": "4", " ": "", "/": "_", "O": "0", "I": "1"}.items():
        s = s.replace(k, v)
    s = re.sub(r"[^\d_]", "", s)
    parts = s.split("_")
    if len(parts) == 2 and len(parts[1]) < 6:
        parts[1] = parts[1].ljust(6, "0")
        s = "_".join(parts)
    return s

def ensure_marked(d, fid):
    d.files().update(
        fileId=fid, body={"appProperties": {"lb_renamed": "1"}}, supportsAllDrives=True
    ).execute()

def unique_name_in_folder(d, folder_id: str, base: str) -> str:
    name, n = base, 1
    while True:
        q = f"name = '{name}' and '{folder_id}' in parents and trashed=false"
        res = d.files().list(
            q=q, fields="files(id)", pageSize=1,
            supportsAllDrives=True, includeItemsFromAllDrives=True
        ).execute()
        if not res.get("files"):
            return name
        root, ext = os.path.splitext(base)
        name = f"{root}_{n}{ext}"
        n += 1

def safe_rename(d, fid: str, target_name: str) -> bool:
    for _ in range(3):
        d.files().update(
            fileId=fid,
            body={"name": target_name, "appProperties": {"lb_renamed": "1"}},
            supportsAllDrives=True,
        ).execute()
        cur = d.files().get(fileId=fid, fields="name,appProperties", supportsAllDrives=True).execute()
        if cur.get("name") == target_name:
            return True
        time.sleep(2)
    return False

def log_skip(name: str, reason: str):
    if DEBUG:
        log.info("SKIP %-28s %s", name or "(no-name)", reason)

# ---------- main ----------
def main():
    d = drive()
    root_id, is_shared, drive_id = resolve_root(d, FOLDER_ID)

    # We list by "name contains 'IMG'" then post-filter startswith('IMG')
    # and verify the file lives under our root (including subfolders).
    processed = 0
    parent_cache: Dict[str, Optional[List[str]]] = {}

    page_token = None
    while True:
        q = "mimeType='application/pdf' and trashed=false and name contains 'IMG'"
        res = d.files().list(
            q=q,
            fields=("nextPageToken, files("
                    "id,name,mimeType,parents,trashed,appProperties,"
                    "createdTime,modifiedTime,capabilities(canRename))"),
            pageSize=200,
            pageToken=page_token,
            orderBy="createdTime desc",
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        ).execute()

        for f in res.get("files", []):
            if processed >= MAX_FILES_PER_RUN:
                break

            if f.get("trashed"):
                continue
            if f.get("mimeType") != PDF_MT:
                continue

            name = f.get("name", "")
            if not (name or "").upper().startswith("IMG"):
                continue  # only IMG*.pdf

            # Already final-looking? mark once & skip
            if RENAMED_RE.match(name or "") or UNKNOWN_RE.match(name or ""):
                if (f.get("appProperties") or {}).get("lb_renamed") != "1":
                    ensure_marked(d, f["id"])
                log_skip(name, "final-looking")
                continue

            # don't reprocess marked items
            if (f.get("appProperties") or {}).get("lb_renamed") == "1":
                continue

            # Must live under our configured root
            parents = f.get("parents") or []
            if not is_under_root(d, parents, root_id, parent_cache):
                continue

            # Permission check
            caps = f.get("capabilities") or {}
            if caps.get("canRename") is False:
                log_skip(name, "no rename permission")
                continue

            # OCR first page
            img = download_first_page(d, f["id"])
            if img is None:
                log_skip(name, "no image")
                continue

            text, conf = ocr_with_confidence(img)
            if conf < 40:
                text, conf = ocr_with_confidence(preprocess_image(img))

            inv = extract_invoice_number(text) or "UNKNOWN"

            parent_id = parents[0] if parents else root_id
            new_name = unique_name_in_folder(d, parent_id, f"{inv}.pdf")
            if new_name == name:
                ensure_marked(d, f["id"])
                log_skip(name, "already correct")
                continue

            if safe_rename(d, f["id"], new_name):
                processed += 1
                log.info("RENAMED %s -> %s", name, new_name)
            else:
                log.warning("FAILED RENAME %s -> %s", name, new_name)

        if processed >= MAX_FILES_PER_RUN:
            break

        page_token = res.get("nextPageToken")
        if not page_token:
            break

    log.info("Processed %d file(s) this run.", processed)

if __name__ == "__main__":
    main()
