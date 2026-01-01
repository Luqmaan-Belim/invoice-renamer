#!/usr/bin/env python3
"""
Renew a Google Drive Changes watch channel SAFELY:
- Stops the previous channel (if we have its id + resourceId)
- Creates a new channel pointing at the SAME stable webhook URL (no ?t= spam)
- Persists the new channel info to .drive_watch_channel.json (cached by GitHub Actions)

Env:
  GDRIVE_SA_JSON    : service account json (one line / json)
  GDRIVE_FOLDER_ID  : folder id (can be shortcut; we resolve)
  WEBHOOK_URL       : stable webhook base or full /hook URL
"""

import json
import os
import time
import uuid
import pathlib
from urllib.parse import urlsplit

from googleapiclient.discovery import build
from google.oauth2 import service_account

CHANNEL_PATH = pathlib.Path(".drive_watch_channel.json")
SHORTCUT_MT = "application/vnd.google-apps.shortcut"


def drive():
    sa = json.loads(os.environ["GDRIVE_SA_JSON"])
    creds = service_account.Credentials.from_service_account_info(
        sa, scopes=["https://www.googleapis.com/auth/drive"]
    )
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def normalize_hook(url: str) -> str:
    url = url.strip()
    if not url.endswith("/hook"):
        url = url.rstrip("/") + "/hook"
    # IMPORTANT: DO NOT append ?t=...  (that creates a “new address” every hour)
    parts = urlsplit(url)
    print("WEBHOOK HOST:", parts.hostname, "PATH:", parts.path)
    return url


def resolve_root_and_drive_id(d, folder_id: str):
    meta = d.files().get(
        fileId=folder_id,
        fields="id,mimeType,shortcutDetails,driveId",
        supportsAllDrives=True,
    ).execute()

    if meta.get("mimeType") == SHORTCUT_MT:
        meta = d.files().get(
            fileId=meta["shortcutDetails"]["targetId"],
            fields="id,driveId",
            supportsAllDrives=True,
        ).execute()

    return meta["id"], meta.get("driveId")


def get_start_page_token(d, drive_id: str | None) -> str:
    if drive_id:
        resp = d.changes().getStartPageToken(driveId=drive_id, supportsAllDrives=True).execute()
    else:
        resp = d.changes().getStartPageToken().execute()
    return resp["startPageToken"]


def stop_old_channel(d):
    if not CHANNEL_PATH.exists():
        print("No previous channel file; nothing to stop.")
        return

    try:
        old = json.loads(CHANNEL_PATH.read_text())
    except Exception as e:
        print("Could not read old channel file; skipping stop:", e)
        return

    cid = old.get("id")
    rid = old.get("resourceId")
    if not cid or not rid:
        print("Old channel file missing id/resourceId; skipping stop.")
        return

    try:
        d.channels().stop(body={"id": cid, "resourceId": rid}).execute()
        print(f"Stopped old channel id={cid}")
    except Exception as e:
        # If it expired already, Drive may error. That's fine.
        print(f"Warning: channels.stop failed (may already be expired): {e}")


def create_new_channel(d, hook: str, drive_id: str | None):
    start = get_start_page_token(d, drive_id)

    body = {
        "id": str(uuid.uuid4()),
        "type": "web_hook",
        "address": hook,
    }

    if drive_id:
        resp = d.changes().watch(
            pageToken=start,
            body=body,
            driveId=drive_id,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        ).execute()
    else:
        resp = d.changes().watch(
            pageToken=start,
            body=body,
            supportsAllDrives=True,
        ).execute()

    # Persist minimal fields needed to stop next time
    persist = {
        "id": resp.get("id"),
        "resourceId": resp.get("resourceId"),
        "expiration": resp.get("expiration"),
        "savedAtEpoch": int(time.time()),
        "hook": hook,
        "driveId": drive_id,
    }
    CHANNEL_PATH.write_text(json.dumps(persist, indent=2))
    print("Created new watch channel:")
    print("  id:", resp.get("id"))
    print("  resourceId:", resp.get("resourceId"))
    print("  expiration(ms):", resp.get("expiration"))


def main():
    folder = os.environ["GDRIVE_FOLDER_ID"].strip()
    hook = normalize_hook(os.environ["WEBHOOK_URL"])

    d = drive()
    _, drive_id = resolve_root_and_drive_id(d, folder)

    stop_old_channel(d)
    create_new_channel(d, hook, drive_id)


if __name__ == "__main__":
    main()
