#!/usr/bin/env python3
"""
Invoice Renamer — Changes API + PyMuPDF + sticky renames + DEBUG tracing
"""

import os, io, re, json, time, logging, pathlib
from typing import Optional, Tuple, Dict, List
from datetime import datetime, timezone

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
GRACE_SECONDS = int(os.environ.get("GRACE_SECONDS", "90"))
MAX_FILES_PER_RUN = int(os.environ.get("MAX_FILES_PER_RUN", "0"))  # 0 = unlimited

TOKEN_PATH = pathlib.Path(".drive_change_token")

PDF_MT = "application/pdf"
SHORTCUT_MT = "application/vnd.google-apps.shortcut"
RENAMED_RE = re.compile(r"^\d{4}_?\d{1,6}(?:_\d+)?\.pdf$", re.I)

def iso_to_dt(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))

# ---------- Drive helpers ----------
def drive():
    creds = service_account.Credentials.from_service_account_info(
        SA_JSON, scopes=["https://www.googleapis.com/auth/drive"]
    )
    return build("drive", "v3", credentials=creds, cache_discovery=False)

def resolve_root(d, folder_id: str) -> Tuple[str, bool, Optional[str]]:
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
    if DEBUG:
        log.info("Root folder: name=%s id=%s sharedDrive=%s driveId=%s",
                 meta.get("name"), meta.get("id"), bool(meta.get("driveId")), meta.get("driveId"))
    return meta["id"], bool(meta.get("driveId")), meta.get("driveId")

def get_start_token(d, is_shared: bool, drive_id: Optional[str]) -> str:
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

def file_meta(d, fid: str) -> Dict:
    return d.files().get(
        fileId=fid,
        fields=("id,name,mimeType,parents,driveId,trashed,appProperties,"
                "createdTime,modifiedTime,capabilities(canRename,canEdit)"),
        supportsAllDrives=True,
    ).execute()

def is_under_root(d, file_parents: List[str], root_id: str, cache: Dict[str, Optional[List[str]]]) -> bool:
    if not file_parents:
        return False
    stack = list(file_parents)
    while stack:
        pid = stack.pop()
        if pid == root_id:
            return True
        if pid in cache:
            parents = cache[pid]
        else:
            try:
                md = d.files().get(fileId=pid, fields="id,parents", supportsAllDrives=True).execute()
                parents = md.get("parents") or []
            except Exception:
                parents = []
            cache[pid] = parents if parents else None
        if parents:
            stack.extend(parents)
    return False

# ---------- PDF → image ----------
def download_first_page(d, file_id: str) -> Optional[np.ndarray]:
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

# ---------- Rename helpers ----------
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

def safe_rename(d, fid: str, target_name: str) -> bool:
    for attempt in range(1, 4):
        d.files().update(
            fileId=fid,
            body={"name": target_name, "appProperties": {"lb_renamed": "1"}},
            supportsAllDrives=True
        ).execute()
        cur = d.files().get(fileId=fid, fields="name,appProperties", supportsAllDrives=True).execute()
        if cur.get("name") == target_name:
            return True
        log.warning("Rename bounce (attempt %d). Current=%s, Wanted=%s",
                    attempt, cur.get("name"), target_name)
        time.sleep(3)
    return False

# ---------- OCR ----------
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

# ---- debug helper
def log_skip(prefix: str, reason: str):
    if DEBUG:
        log.info("SKIP %-20s %s", prefix, reason)

# ---------- Main ----------
def main():
    d = drive()
    root_id, is_shared, drive_id = resolve_root(d, FOLDER_ID)

    token = load_token()
    if not token:
        token = get_start_token(d, is_shared, drive_id)
        save_token(token)
        log.info("Initialized change token.")
        return

    parent_cache: Dict[str, Optional[List[str]]] = {}
    processed = 0
    next_token = None
    total_seen = 0

    while True:
        fields = (
            "nextPageToken,newStartPageToken,"
            "changes(fileId,removed,"
            "file(id,name,mimeType,parents,driveId,trashed,appProperties,"
            "createdTime,modifiedTime,capabilities))"
        )
        kwargs = dict(
            pageToken=token,
            fields=fields,
            includeItemsFromAllDrives=True,
            supportsAllDrives=True,
        )
        if is_shared and drive_id:
            kwargs.update(driveId=drive_id)

        resp = d.changes().list(**kwargs).execute()
        changes = resp.get("changes", [])
        total_seen += len(changes)
        if DEBUG:
            log.info("Fetched %d change(s) at token=%s (driveId=%s)", len(changes), token, drive_id or "mydrive")

        for ch in changes:
            if ch.get("removed", False):
                log_skip("removed", ch.get("fileId", ""))
                continue
            fobj = ch.get("file") or {}
            fid = ch.get("fileId")
            if not fid or not fobj:
                log_skip("no-file", str(ch))
                continue

            name = fobj.get("name", "")
            mt = fobj.get("mimeType")
            parents = fobj.get("parents")
            if DEBUG:
                log.info("Change fid=%s name=%s mime=%s parents=%s", fid, name, mt, parents)

            if fobj.get("trashed"):
                log_skip(name, "trashed")
                continue
            if mt != PDF_MT:
                log_skip(name, f"mime {mt}")
                continue
            if RENAMED_RE.match(name or ""):
                log_skip(name, "already looks renamed")
                continue

            props = fobj.get("appProperties") or {}
            if props.get("lb_renamed") == "1":
                log_skip(name, "appProperty lb_renamed=1")
                continue

            caps = fobj.get("capabilities") or {}
            if caps and not caps.get("canRename", True):
                log_skip(name, "cannot rename (permissions)")
                continue

            created = fobj.get("createdTime") or fobj.get("modifiedTime")
            if not parents or not created:
                meta = file_meta(d, fid)
                if meta.get("trashed"):
                    log_skip(name, "trashed (meta)")
                    continue
                parents = parents or (meta.get("parents") or [])
                created = created or (meta.get("createdTime") or meta.get("modifiedTime"))
                name = name or meta.get("name", "")
                if DEBUG:
                    log.info("Refreshed meta for %s: parents=%s created=%s", fid, parents, created)

            if created:
                age = (datetime.now(timezone.utc) - iso_to_dt(created)).total_seconds()
                if age < GRACE_SECONDS:
                    log_skip(name, f"too new {int(age)}s < {GRACE_SECONDS}s")
                    continue

            if not is_under_root(d, parents, root_id, parent_cache):
                log_skip(name, "outside root tree")
                continue

            img = download_first_page(d, fid)
            if img is None:
                log_skip(name, "no image from pdf")
                continue

            text, conf = ocr_with_confidence(img)
            if conf < 40:
                img2 = preprocess_image(img)
                text, conf = ocr_with_confidence(img2)

            inv = extract_invoice_number(text) or "UNKNOWN"
            parent_id = parents[0] if parents else root_id
            new_name = unique_name_in_folder(d, parent_id, f"{inv}.pdf")

            if new_name == name:
                log_skip(name, "target name equals current")
                continue

            if safe_rename(d, fid, new_name):
                processed += 1
                log.info("RENAMED %s -> %s", name, new_name)
            else:
                log.error("FAILED RENAME %s -> %s", name, new_name)

            if MAX_FILES_PER_RUN and processed >= MAX_FILES_PER_RUN:
                log.info("Hit MAX_FILES_PER_RUN=%d; stopping early.", MAX_FILES_PER_RUN)
                break

        token = resp.get("nextPageToken")
        if not token:
            next_token = resp.get("newStartPageToken")
            break
        if MAX_FILES_PER_RUN and processed >= MAX_FILES_PER_RUN:
            next_token = resp.get("newStartPageToken") or token
            break

    if next_token:
        save_token(next_token)

    if DEBUG:
        log.info("SUMMARY: seen=%d processed=%d savedToken=%s", total_seen, processed, next_token)
    else:
        log.info("Processed %d file(s).", processed)

if __name__ == "__main__":
    main()
