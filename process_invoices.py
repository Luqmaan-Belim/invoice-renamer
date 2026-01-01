#!/usr/bin/env python3
"""
Invoice Renamer — Changes-only (fast) version.

- Uses Drive Changes API with a persisted startPageToken: only processes NEW/CHANGED files.
- Resolves the provided folder ID if it’s a shortcut, supports My Drive & Shared Drives.
- Filters to PDFs whose names start with "IMG" (case-insensitive), within the folder tree.
- OCRs the first page with Tesseract (rasterized via PyMuPDF) and renames the file in place.
- Skips items already in final "NNNN[_mmmmmm][_n].pdf" form.

ENV
  GDRIVE_SA_JSON   : service account JSON (one line)
  GDRIVE_FOLDER_ID : target folder ID (not shortcut; will be resolved if it is)
  DEBUG_LIST       : "1" for verbose logging (optional)

Python requirements
  google-api-python-client, google-auth
  pytesseract, opencv-python-headless, numpy, Pillow, PyMuPDF

System requirement
  tesseract-ocr
"""

import base64
import io
import os
import re
import json
import logging
import pathlib
from typing import Optional, Tuple, Dict, List

import numpy as np
import cv2
import pytesseract
import fitz  # PyMuPDF

from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
from google.oauth2 import service_account

# ---------- Logging ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("invoice_renamer")

# ---------- Config from env ----------
FOLDER_ID = os.environ["GDRIVE_FOLDER_ID"].strip()
SA_JSON = json.loads(os.environ["GDRIVE_SA_JSON"])
DEBUG = os.environ.get("DEBUG_LIST", "0") == "1"

# Where we persist the Drive startPageToken between runs.
# The authoritative copy lives in Drive appDataFolder; we mirror it locally for compatibility.
TOKEN_PATH = pathlib.Path(".drive_change_token")
TOKEN_APPDATA_NAME = "invoice-renamer-token"
DISABLE_APPDATA_TOKEN = os.environ.get("DISABLE_APPDATA_TOKEN", "0") == "1"

# MIME constants
PDF_MT = "application/pdf"
FOLDER_MT = "application/vnd.google-apps.folder"
SHORTCUT_MT = "application/vnd.google-apps.shortcut"

# File name filters/patterns
IMG_PDF_RE = re.compile(r"^IMG.*\.pdf$", re.I)
RENAMED_RE = re.compile(r"^\d{4}(?:_\d{1,6})?(?:_\d+)?\.pdf$", re.I)

# ---------- Drive helpers ----------
def drive():
    creds = service_account.Credentials.from_service_account_info(
        SA_JSON, scopes=["https://www.googleapis.com/auth/drive"]
    )
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def resolve_root(d, folder_id: str) -> Tuple[str, Optional[str]]:
    """
    Return (real_root_id, drive_id).
    If the provided ID is a shortcut, resolve to its target.
    """
    meta = d.files().get(
        fileId=folder_id,
        fields="id,name,mimeType,driveId,shortcutDetails",
        supportsAllDrives=True,
    ).execute()
    if meta.get("mimeType") == SHORTCUT_MT:
        tgt = meta["shortcutDetails"]["targetId"]
        resolved = d.files().get(
            fileId=tgt,
            fields="id,name,mimeType,driveId",
            supportsAllDrives=True,
        ).execute()
        log.info("Resolved shortcut root: %s -> %s", folder_id, resolved["id"])
        return resolved["id"], resolved.get("driveId")
    return meta["id"], meta.get("driveId")


def get_start_token(d, drive_id: Optional[str]) -> str:
    """Fetch a fresh startPageToken."""
    if drive_id:
        resp = d.changes().getStartPageToken(
            driveId=drive_id, supportsAllDrives=True
        ).execute()
    else:
        resp = d.changes().getStartPageToken().execute()
    return resp["startPageToken"]


def load_token(d) -> Optional[str]:
    token: Optional[str] = None

    if not DISABLE_APPDATA_TOKEN:
        try:
            resp = d.files().list(
                spaces="appDataFolder",
                q=f"name='{TOKEN_APPDATA_NAME}' and trashed=false",
                fields="files(id)",
                pageSize=1,
            ).execute()
            files = resp.get("files") or []
            if files:
                data = d.files().get_media(fileId=files[0]["id"]).execute()
                if isinstance(data, bytes):
                    token = data.decode("utf-8", errors="ignore").strip()
                else:
                    token = str(data).strip()
        except Exception as exc:
            log.warning("Unable to read Drive token from appData: %s", exc)

    if token:
        try:
            TOKEN_PATH.write_text(token)
        except Exception:
            pass
        return token or None

    try:
        token = TOKEN_PATH.read_text().strip()
        if not token:
            return None
        if not DISABLE_APPDATA_TOKEN:
            log.info("Using local Drive change token fallback; enable DISABLE_APPDATA_TOKEN=1 to opt out of appData usage if desired.")
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

    data = token.encode("utf-8")
    media = MediaIoBaseUpload(io.BytesIO(data), mimetype="text/plain")
    try:
        resp = d.files().list(
            spaces="appDataFolder",
            q=f"name='{TOKEN_APPDATA_NAME}' and trashed=false",
            fields="files(id)",
            pageSize=1,
        ).execute()
        files = resp.get("files") or []
        if files:
            d.files().update(fileId=files[0]["id"], media_body=media).execute()
        else:
            body = {"name": TOKEN_APPDATA_NAME, "parents": ["appDataFolder"]}
            d.files().create(body=body, media_body=media, fields="id").execute()
    except Exception as exc:
        log.warning("Unable to save Drive token to appData: %s", exc)


def file_meta(d, fid: str, fields: str):
    return d.files().get(fileId=fid, fields=fields, supportsAllDrives=True).execute()


def is_under_root(
    d,
    file_parents: List[str],
    root_id: str,
    parent_cache: Dict[str, Optional[List[str]]],
) -> bool:
    """
    Ascend parents until we hit root_id or the top.
    parent_cache memoizes parent->parents lookups to reduce API calls.
    """
    if not file_parents:
        return False
    stack = list(file_parents)
    while stack:
        pid = stack.pop()
        if pid == root_id:
            return True
        if pid in parent_cache:
            parents = parent_cache[pid]
        else:
            try:
                md = d.files().get(fileId=pid, fields="id,parents", supportsAllDrives=True).execute()
                parents = md.get("parents") or []
            except Exception:
                parents = []
            parent_cache[pid] = parents if parents else None
        if parents:
            stack.extend(parents)
    return False


# ---------- Download & OCR ----------
def download_first_page(d, file_id: str) -> Optional[np.ndarray]:
    """
    Download the PDF and rasterize the first page ~300 DPI using PyMuPDF.
    Returns an RGB uint8 HxWx3 array or None.
    """
    req = d.files().get_media(fileId=file_id)
    buf = io.BytesIO()
    downloader = next_chunk = None
    from googleapiclient.http import MediaIoBaseDownload
    downloader = MediaIoBaseDownload(buf, req)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    data = buf.getvalue()
    if not data:
        return None

    zoom = 300.0 / 72.0  # 72pt baseline -> ~300 DPI
    mat = fitz.Matrix(zoom, zoom)
    with fitz.open(stream=data, filetype="pdf") as doc:
        if doc.page_count == 0:
            return None
        page = doc.load_page(0)
        pix = page.get_pixmap(matrix=mat, alpha=False)  # RGB
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
    """
    Extract invoice number. Pattern: (INV|NV) ... 4+digits ... up to 6 trailing digits.
    Produces 'NNNN[_mmmmmm]' style after cleaning.
    """
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
    """
    Ensure (base) is unique in the folder; if not, suffix _n.
    """
    name, n = base, 1
    while True:
        q = f"name='{name}' and '{folder_id}' in parents and trashed=false"
        res = d.files().list(
            q=q, fields="files(id)", pageSize=1,
            supportsAllDrives=True, includeItemsFromAllDrives=True
        ).execute()
        if not res.get("files"):
            return name
        root, ext = os.path.splitext(base)
        name = f"{root}_{n}{ext}"
        n += 1


def rename_in_drive(d, fid: str, new_name: str):
    d.files().update(fileId=fid, body={"name": new_name}, supportsAllDrives=True).execute()


# ---------- Main (changes-driven) ----------
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

    while True:
        kwargs = dict(
            pageToken=token,
            fields="nextPageToken,newStartPageToken,changes(fileId,removed,file)",
            includeItemsFromAllDrives=True,
            supportsAllDrives=True,
        )
        if drive_id:
            kwargs["driveId"] = drive_id

        resp = d.changes().list(**kwargs).execute()

        for ch in resp.get("changes", []):
            fid = ch.get("fileId")
            removed = ch.get("removed", False)
            fobj = ch.get("file") or {}

            if removed or not fobj:
                continue
            if fobj.get("trashed"):
                continue
            if fobj.get("mimeType") != PDF_MT:
                continue

            # name & parents might not be present in change
            name = fobj.get("name", "")
            parents = fobj.get("parents") or []

            # Quick file-name filters:
            if RENAMED_RE.match(name or ""):
                if DEBUG:
                    log.info("SKIP %s  already looks renamed", name)
                continue
            if not IMG_PDF_RE.match(name or ""):
                if DEBUG:
                    log.info("SKIP %s  not IMG*.pdf", name)
                continue

            if not parents:
                meta = file_meta(d, fid, "id,name,parents,trashed")
                if meta.get("trashed"):
                    continue
                name = meta.get("name", name)
                parents = meta.get("parents") or []

            if not is_under_root(d, parents, root_id, parent_cache):
                if DEBUG:
                    log.info("SKIP %s  outside target tree", name)
                continue

            # Download first page & OCR
            img = download_first_page(d, fid)
            if img is None:
                if DEBUG:
                    log.info("No image for %s; skipping", name)
                continue

            text, conf = ocr_with_confidence(img)
            if conf < 40:
                img2 = preprocess_image(img)
                text, conf = ocr_with_confidence(img2)

            inv = extract_invoice_number(text) or "UNKNOWN"

            parent_id = parents[0] if parents else root_id
            new_name = unique_name_in_folder(d, parent_id, f"{inv}.pdf")
            if new_name != name:
                rename_in_drive(d, fid, new_name)
                log.info("RENAMED  %s -> %s", name, new_name)
                processed += 1
            elif DEBUG:
                log.info("SKIP %s  name unchanged", name)

        token = resp.get("nextPageToken")
        if not token:
            next_token = resp.get("newStartPageToken")
            break

    if next_token:
        save_token(d, next_token)

    log.info("Processed %d file(s) this run.", processed)


if __name__ == "__main__":
    main()
