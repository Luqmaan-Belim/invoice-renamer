# process_invoices.py (drop-in replacement)

import os, io, re, json, logging
from googleapiclient.discovery import build
from google.oauth2 import service_account
from googleapiclient.http import MediaIoBaseDownload
from pdf2image import convert_from_bytes
import pytesseract, numpy as np, cv2

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

FOLDER_ID = os.environ["GDRIVE_FOLDER_ID"].strip()
SA_JSON = json.loads(os.environ["GDRIVE_SA_JSON"])
SCOPES = ["https://www.googleapis.com/auth/drive"]
DEBUG_LIST = os.environ.get("DEBUG_LIST", "0") == "1"   # set to 1 in a run if you want extra logs

RENAMED = re.compile(r'^\d{4}_?\d{1,6}(?:_\d+)?\.pdf$', re.I)

def drive():
    creds = service_account.Credentials.from_service_account_info(SA_JSON, scopes=SCOPES)
    return build("drive", "v3", credentials=creds, cache_discovery=False)

def resolve_folder(d, folder_id):
    """Return (actual_folder_id, is_shared_drive, drive_id_or_None). Also resolves shortcuts."""
    meta = d.files().get(fileId=folder_id, fields="id, name, mimeType, driveId, shortcutDetails").execute()
    if meta.get("mimeType") == "application/vnd.google-apps.shortcut":
        target = meta["shortcutDetails"]["targetId"]
        logging.info("Resolved shortcut: %s -> %s", folder_id, target)
        meta = d.files().get(fileId=target, fields="id, name, mimeType, driveId").execute()
        folder_id = meta["id"]
    is_shared = bool(meta.get("driveId"))
    return folder_id, is_shared, meta.get("driveId")

def list_pdfs(d, folder_id, is_shared, drive_id):
    # Build a query that works for both My Drive and Shared Drives
    q = f"'{folder_id}' in parents and mimeType='application/pdf' and trashed=false"
    fields = "nextPageToken, files(id,name,createdTime)"
    kwargs = {"q": q, "fields": fields, "pageSize": 1000, "orderBy": "createdTime desc",
              "supportsAllDrives": True, "includeItemsFromAllDrives": True}
    if is_shared and drive_id:
        kwargs["corpora"] = "drive"
        kwargs["driveId"] = drive_id
    files, tok = [], None
    while True:
        if tok: kwargs["pageToken"] = tok
        resp = d.files().list(**kwargs).execute()
        files += resp.get("files", [])
        tok = resp.get("nextPageToken")
        if not tok: break
    if DEBUG_LIST:
        logging.info("DEBUG: folder %s (shared=%s) visible files: %d", folder_id, is_shared, len(files))
        for f in files[:10]:
            logging.info("DEBUG: %s", f["name"])
    return [f for f in files if not RENAMED.match(f["name"])]

def download_first_page(d, file_id):
    buf = io.BytesIO()
    req = d.files().get_media(fileId=file_id)
    dl = MediaIoBaseDownload(buf, req)
    done = False
    while not done:
        _, done = dl.next_chunk()
    buf.seek(0)
    imgs = convert_from_bytes(buf.getvalue(), dpi=300, first_page=1, last_page=1)
    return np.array(imgs[0]) if imgs else None

def ocr_with_conf(img):
    data = pytesseract.image_to_data(img, config="--psm 6 --oem 3", output_type=pytesseract.Output.DICT)
    text = " ".join([t for t in data.get("text", []) if t]).strip()
    confs = [c for c in data.get("conf", []) if isinstance(c,(int,float)) and c != -1]
    avg = sum(confs)/len(confs) if confs else 0
    return text, avg

def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 11, 2)

def extract_invoice(text):
    m = re.search(r'(?:INV|NV)[^\d]*(\d{4}[^\d]*\d{1,6})', text, flags=re.I)
    if not m: return None
    s = m.group(1).strip()
    for k,v in {'A':'4',' ':'','/':'_','O':'0','I':'1'}.items(): s = s.replace(k,v)
    s = re.sub(r'[^\d_]', '', s)
    parts = s.split('_')
    if len(parts) == 2 and len(parts[1]) < 6:
        parts[1] = parts[1].ljust(6,'0'); s = '_'.join(parts)
    return s

def unique_name(d, folder_id, base):
    name, n = base, 1
    while True:
        q = f"name = '{name}' and '{folder_id}' in parents and trashed=false"
        res = d.files().list(q=q, fields="files(id)", pageSize=1,
                              supportsAllDrives=True, includeItemsFromAllDrives=True).execute()
        if not res.get("files"): return name
        root, ext = os.path.splitext(base); name = f"{root}_{n}{ext}"; n += 1

def rename(d, file_id, new):
    d.files().update(fileId=file_id, body={"name": new}, supportsAllDrives=True).execute()

def main():
    d = drive()
    folder_id, is_shared, drive_id = resolve_folder(d, FOLDER_ID)
    targets = list_pdfs(d, folder_id, is_shared, drive_id)
    if not targets:
        logging.info("No PDFs to process."); return
    logging.info("Found %d PDF(s)", len(targets))

    for f in targets:
        fid, fname = f["id"], f["name"]
        logging.info("Processing: %s", fname)
        img = download_first_page(d, fid)
        if img is None: logging.warning("No image; skipping."); continue
        text, conf = ocr_with_conf(img)
        if conf < 40: text, conf = ocr_with_conf(preprocess(img))
        logging.info("OCR avg conf: %.2f", conf)
        inv = extract_invoice(text) or "UNKNOWN"
        new = unique_name(d, folder_id, f"{inv}.pdf")
        if new == fname: logging.info("Already correct"); continue
        rename(d, fid, new)
        logging.info("Renamed to: %s", new)

if __name__ == "__main__":
    main()
