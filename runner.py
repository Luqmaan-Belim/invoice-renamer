#!/usr/bin/env python3
"""Fresh Bake invoice / delivery-note / customer-claim renamer.

Policy:
- always process genuinely new IMG*.pdf files before backlog/retries;
- permanently delete Drive files whose size is 0 bytes;
- OCR an unchanged unrecognised PDF once, then leave it in manual_review;
- retry recognised documents only when the missing piece is in Odoo, without
  repeating OCR, by caching the recognised target in Drive appProperties;
- retry manual-review files only when their bytes change or PARSER_VERSION changes.
"""

import logging
import os
import re
import time
from collections import defaultdict
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

import process_invoices as core


log = logging.getLogger("invoice_renamer")

MAX_FILES_PER_RUN = int(os.environ.get("MAX_FILES_PER_RUN", "40"))
CLAIM_LOOKBACK_DAYS = int(os.environ.get("CLAIM_LOOKBACK_DAYS", "365"))
ODOO_RETRY_MINUTES = int(os.environ.get("ODOO_RETRY_MINUTES", "30"))
PARSER_VERSION = "3"

STATUS_KEY = "invoice_renamer_status"
MD5_KEY = "invoice_renamer_md5"
VERSION_KEY = "invoice_renamer_parser_version"
REASON_KEY = "invoice_renamer_reason"
RETRY_AFTER_KEY = "invoice_renamer_retry_after"
TARGET_TYPE_KEY = "invoice_renamer_target_type"
TARGET_VALUE_KEY = "invoice_renamer_target_value"
TARGET_MOVE_TYPE_KEY = "invoice_renamer_target_move_type"

ORDER_TRANSLATION = str.maketrans({
    "O": "0", "Q": "0", "D": "0", "I": "1", "L": "1",
    "Z": "2", "S": "5", "G": "6", "B": "8",
})
ORDER_TOKEN_RE = re.compile(
    r"(?<![A-Z0-9])([S$53][\s._'’:/\\-]*(?:[0-9OQDISBGLZI1][\s._'’:/\\-]*){5,8})(?![A-Z0-9])",
    re.I,
)
DELIVERY_CONTEXT_RE = re.compile(
    r"(?:D|B|O)?ELIVERY\s+[FN]?OTE|DELIVERY\s+NOTE", re.I
)


def _int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalise(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", (value or "").upper())


def _partner_tokens(name: str) -> List[str]:
    stop = {"PTY", "LTD", "CC", "THE", "AND", "SOUTH", "AFRICA", "STORE", "SUPER", "FOODS"}
    return [
        token for token in re.findall(r"[A-Z0-9]+", (name or "").upper())
        if len(token) >= 4 and token not in stop
    ]


def _props(item: dict) -> dict:
    return item.get("appProperties") or {}


def _set_properties(d, item: dict, **updates):
    props = dict(_props(item))
    for key, value in updates.items():
        if value is None:
            props.pop(key, None)
        else:
            props[key] = str(value)
    d.files().update(
        fileId=item["id"], body={"appProperties": props}, supportsAllDrives=True
    ).execute()
    item["appProperties"] = props


def _clear_state(d, item: dict):
    keys = (
        STATUS_KEY, MD5_KEY, VERSION_KEY, REASON_KEY, RETRY_AFTER_KEY,
        TARGET_TYPE_KEY, TARGET_VALUE_KEY, TARGET_MOVE_TYPE_KEY,
    )
    if any(key in _props(item) for key in keys):
        _set_properties(d, item, **{key: None for key in keys})


def _mark_manual_review(d, item: dict, reason: str):
    _set_properties(
        d,
        item,
        **{
            STATUS_KEY: "manual_review",
            MD5_KEY: item.get("md5Checksum") or "",
            VERSION_KEY: PARSER_VERSION,
            REASON_KEY: reason[:100],
            RETRY_AFTER_KEY: None,
            TARGET_TYPE_KEY: None,
            TARGET_VALUE_KEY: None,
            TARGET_MOVE_TYPE_KEY: None,
        },
    )
    log.warning("MANUAL REVIEW %s: %s", item.get("name"), reason)


def _mark_awaiting_odoo(
    d, item: dict, target_type: str, target_value: str, reason: str,
    move_type: Optional[str] = None,
):
    retry_after = int(time.time() + ODOO_RETRY_MINUTES * 60)
    _set_properties(
        d,
        item,
        **{
            STATUS_KEY: "awaiting_odoo",
            MD5_KEY: item.get("md5Checksum") or "",
            VERSION_KEY: PARSER_VERSION,
            REASON_KEY: reason[:100],
            RETRY_AFTER_KEY: retry_after,
            TARGET_TYPE_KEY: target_type,
            TARGET_VALUE_KEY: target_value,
            TARGET_MOVE_TYPE_KEY: move_type,
        },
    )
    log.warning(
        "AWAITING ODOO %s: target=%s:%s retry_in=%dm reason=%s",
        item.get("name"), target_type, target_value, ODOO_RETRY_MINUTES, reason,
    )


def _mark_transient(d, item: dict, reason: str):
    retry_after = int(time.time() + 5 * 60)
    _set_properties(
        d,
        item,
        **{
            STATUS_KEY: "transient",
            MD5_KEY: item.get("md5Checksum") or "",
            VERSION_KEY: PARSER_VERSION,
            REASON_KEY: reason[:100],
            RETRY_AFTER_KEY: retry_after,
        },
    )
    log.warning("TRANSIENT %s: retry later (%s)", item.get("name"), reason)


def _eligibility(item: dict) -> Tuple[bool, str]:
    props = _props(item)
    status = props.get(STATUS_KEY, "")
    current_md5 = item.get("md5Checksum") or ""
    stored_md5 = props.get(MD5_KEY, "")
    parser_version = props.get(VERSION_KEY, "")

    if not status:
        return True, "new"

    if current_md5 and stored_md5 and current_md5 != stored_md5:
        return True, "file-changed"

    if status == "manual_review":
        if parser_version != PARSER_VERSION:
            return True, "parser-upgraded"
        return False, "manual-review"

    if status in {"awaiting_odoo", "transient"}:
        if _int(props.get(RETRY_AFTER_KEY), 0) <= int(time.time()):
            return True, status
        return False, status

    if parser_version != PARSER_VERSION:
        return True, "parser-upgraded"
    return True, "backlog"


def iter_outstanding_pdfs(d, root_id: str, drive_id: Optional[str]) -> Iterator[dict]:
    pending_folders = [root_id]
    visited = set()
    while pending_folders:
        folder_id = pending_folders.pop()
        if folder_id in visited:
            continue
        visited.add(folder_id)
        page_token = None
        while True:
            kwargs = {
                "q": f"'{folder_id}' in parents and trashed=false",
                "fields": (
                    "nextPageToken,files(id,name,mimeType,parents,size,modifiedTime,"
                    "md5Checksum,appProperties,shortcutDetails(targetId,targetMimeType))"
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
                mime = item.get("mimeType")
                name = item.get("name", "")
                if mime == core.FOLDER_MT:
                    pending_folders.append(item["id"])
                elif mime == core.PDF_MT and core.IMG_PDF_RE.match(name):
                    yield item
            page_token = response.get("nextPageToken")
            if not page_token:
                break


def _delete_zero_byte(d, item: dict) -> bool:
    try:
        d.files().delete(fileId=item["id"], supportsAllDrives=True).execute()
        log.warning("DELETED ZERO BYTE %s", item.get("name"))
        return True
    except Exception as exc:
        log.exception("ZERO BYTE DELETE FAILED %s: %s", item.get("name"), exc)
        return False


def _load_claim_index(odoo_client) -> Dict[str, List[dict]]:
    index: Dict[str, List[dict]] = defaultdict(list)
    if not odoo_client:
        return index
    rows = odoo_client.list_credit_note_claims(CLAIM_LOOKBACK_DAYS)
    for row in rows:
        claim = str(row.get("claim_no") or "").strip()
        norm = _normalise(claim)
        if not norm:
            continue
        partner = row.get("partner_id") or False
        partner_name = partner[1] if isinstance(partner, (list, tuple)) and len(partner) > 1 else ""
        enriched = dict(row)
        enriched["claim_text"] = claim
        enriched["claim_norm"] = norm
        enriched["partner_name"] = partner_name
        index[norm].append(enriched)
    log.info(
        "Loaded %d distinct Odoo claim numbers (%d credit notes)",
        len(index), sum(map(len, index.values())),
    )
    return index


def _ocr_full_page_rotations(image: np.ndarray) -> Tuple[List[str], float]:
    texts: List[str] = []
    best_conf = 0.0
    for turns, angle in ((0, 0), (1, 90), (2, 180), (3, 270)):
        rotated = np.rot90(image, turns).copy() if turns else image
        for label, prepared, psm in (
            ("raw", rotated, 6),
            ("enhanced", core.preprocess_image(rotated), 11),
        ):
            text, confidence = core.ocr_with_confidence(prepared, psm=psm)
            best_conf = max(best_conf, confidence)
            if text:
                texts.append(text)
                if core.DEBUG:
                    log.info(
                        "CLAIM OCR angle=%s mode=%s confidence=%.1f text=%r",
                        angle, label, confidence, text,
                    )
    return texts, best_conf


def _match_claim(texts: List[str], claim_index: Dict[str, List[dict]]) -> Tuple[Optional[str], List[dict]]:
    if not texts or not claim_index:
        return None, []
    normalised_texts = [(_normalise(text), text.upper()) for text in texts]
    matched: Dict[str, List[dict]] = {}

    for claim_norm in sorted(claim_index, key=len, reverse=True):
        rows = claim_index[claim_norm]
        for compact, raw_upper in normalised_texts:
            if claim_norm not in compact:
                continue
            if len(claim_norm) <= 4:
                if not any(
                    any(token in raw_upper or token in compact for token in _partner_tokens(row.get("partner_name", "")))
                    for row in rows
                ):
                    continue
            matched[claim_norm] = rows
            break

    if not matched:
        return None, []
    longest = max(len(key) for key in matched)
    winners = [key for key in matched if len(key) == longest]
    if len(winners) != 1:
        return None, []
    winner = winners[0]
    return winner, matched[winner]


def _safe_name(value: str) -> str:
    return (value or "").replace("/", "_").replace("\\", "_").strip("_") or "DOCUMENT"


def _claim_filename(claim_norm: str, rows: List[dict]) -> str:
    move_names = sorted({str(row.get("name") or "").strip() for row in rows if row.get("name")})
    if len(move_names) == 1:
        return f"CLAIM_{claim_norm}_{_safe_name(move_names[0])}.pdf"
    return f"CLAIM_{claim_norm}.pdf"


def _attach_claim_rows(odoo_client, rows: List[dict], filename: str, pdf_bytes: bytes):
    for row in rows:
        attachment_id, created = odoo_client.ensure_pdf_attachment(row["id"], filename, pdf_bytes)
        log.info(
            "%s %s move_id=%s attachment_id=%s",
            "ODOO ATTACHED" if created else "ODOO ATTACHMENT ALREADY EXISTS",
            filename, row["id"], attachment_id,
        )


def _normalise_order_token(raw: str) -> Optional[str]:
    value = re.sub(r"[\s._'’:/\\-]+", "", (raw or "").upper())
    if not value:
        return None
    if value[0] in "53$":
        value = "S" + value[1:]
    if not value.startswith("S"):
        return None
    digits = re.sub(r"[^A-Z0-9]", "", value[1:]).translate(ORDER_TRANSLATION)
    digits = re.sub(r"\D", "", digits)
    if not 5 <= len(digits) <= 8:
        return None
    return "S" + digits


def _delivery_order_candidates(text: str) -> List[str]:
    candidates: List[str] = []
    for line in (text or "").splitlines():
        upper = line.upper()
        if not any(word in upper for word in ("DELIVERY", "ELIVERY", "ORDER", "RDER", "REFERENCE", "EFERENCE")):
            continue
        for match in ORDER_TOKEN_RE.finditer(upper):
            candidate = _normalise_order_token(match.group(1))
            if candidate and candidate not in candidates:
                candidates.append(candidate)
    return candidates


def _delivery_header_ocr(image: np.ndarray) -> Tuple[str, float]:
    height, width = image.shape[:2]
    crop = image[int(height * 0.035):int(height * 0.27), int(width * 0.01):int(width * 0.92)]
    best_text, best_conf = "", 0.0
    for prepared, psm in (
        (crop, 6),
        (core.preprocess_image(crop), 6),
        (core.adaptive_preprocess(crop), 6),
    ):
        text, confidence = core.ocr_with_confidence(prepared, psm=psm)
        if confidence > best_conf:
            best_text, best_conf = text, confidence
        if _delivery_order_candidates(text):
            return text, confidence
    return best_text, best_conf


def _looks_like_delivery_note(text: str) -> bool:
    upper = (text or "").upper()
    return bool(DELIVERY_CONTEXT_RE.search(upper)) or (
        "FRESH BAKE" in upper and ("DELIVERY" in upper or "ELIVERY" in upper)
    )


def _delivery_filename(order_name: str, moves: List[dict]) -> str:
    invoice_names = sorted({str(move.get("name") or "").strip() for move in moves if move.get("name")})
    if len(invoice_names) == 1:
        return f"DN_{order_name}_{_safe_name(invoice_names[0])}.pdf"
    return f"DN_{order_name}.pdf"


def _retry_cached_odoo_target(d, item: dict, odoo_client, pdf_bytes: bytes) -> Optional[bool]:
    """Retry an Odoo-side dependency without repeating OCR. None = no cached target."""
    props = _props(item)
    if props.get(STATUS_KEY) != "awaiting_odoo":
        return None
    if (props.get(MD5_KEY) or "") != (item.get("md5Checksum") or ""):
        return None
    if props.get(VERSION_KEY) != PARSER_VERSION:
        return None

    target_type = props.get(TARGET_TYPE_KEY, "")
    target_value = props.get(TARGET_VALUE_KEY, "")
    if not target_type or not target_value:
        return None
    if not odoo_client:
        _mark_awaiting_odoo(d, item, target_type, target_value, "Odoo unavailable", props.get(TARGET_MOVE_TYPE_KEY))
        return False

    parent_id = (item.get("parents") or [core.FOLDER_ID])[0]
    if target_type == "document":
        move_type = props.get(TARGET_MOVE_TYPE_KEY, "out_invoice")
        matches = odoo_client.search_customer_document_by_number(target_value, move_type)
        if not matches:
            _mark_awaiting_odoo(d, item, target_type, target_value, f"no Odoo match for {target_value}", move_type)
            return False
        stem = target_value.replace("INV/", "").replace("RINV/", "RINV/").replace("/", "_")
        if target_value.startswith("RINV/"):
            stem = target_value.replace("/", "_")
        filename = core.unique_name_in_folder(d, parent_id, f"{stem}.pdf")
        for move_id in matches:
            odoo_client.ensure_pdf_attachment(move_id, filename, pdf_bytes)
        _clear_state(d, item)
        core.rename_in_drive(d, item["id"], filename)
        log.info("ODOO RETRY MATCH %s -> %s target=%s matches=%d", item.get("name"), filename, target_value, len(matches))
        return True

    if target_type == "delivery":
        resolved = odoo_client.find_sale_orders_with_customer_invoices([target_value])
        viable = [row for row in resolved if row.get("invoice_moves")]
        if len(viable) != 1:
            _mark_awaiting_odoo(d, item, target_type, target_value, f"sale order {target_value} still has no unique related invoice")
            return False
        order = viable[0]
        moves = order["invoice_moves"]
        filename = core.unique_name_in_folder(d, parent_id, _delivery_filename(order["name"], moves))
        for move in moves:
            odoo_client.ensure_pdf_attachment(move["id"], filename, pdf_bytes)
        _clear_state(d, item)
        core.rename_in_drive(d, item["id"], filename)
        log.info("ODOO RETRY DELIVERY %s -> %s order=%s", item.get("name"), filename, target_value)
        return True

    return None


def process_one(d, item: dict, odoo_client, claim_index: Dict[str, List[dict]]) -> bool:
    name = item.get("name", "")
    pdf_bytes = core.download_pdf_bytes(d, item)
    if not pdf_bytes:
        _mark_transient(d, item, "download failed or incomplete")
        return False

    cached = _retry_cached_odoo_target(d, item, odoo_client, pdf_bytes)
    if cached is not None:
        return cached

    pdf_text = core.extract_first_page_text(pdf_bytes)

    document = core.extract_document_number(pdf_text)
    image = None
    confidence = 100.0 if document else 0.0
    source = "pdf-text" if document else ""
    if not document:
        image = core.rasterize_first_page(pdf_bytes)
        if image is None:
            _mark_manual_review(d, item, "could not rasterize first page")
            return False
        document, confidence, source = core.extract_document_number_from_image(image)

    if document:
        if not odoo_client:
            _mark_awaiting_odoo(d, item, "document", document.odoo_number, "Odoo unavailable", document.move_type)
            return False
        matches = odoo_client.search_customer_document_by_number(document.odoo_number, document.move_type)
        if not matches:
            _mark_awaiting_odoo(
                d, item, "document", document.odoo_number,
                f"no Odoo match for {document.odoo_number}", document.move_type,
            )
            return False
        parent_id = (item.get("parents") or [core.FOLDER_ID])[0]
        filename = core.unique_name_in_folder(d, parent_id, f"{document.filename_stem}.pdf")
        for move_id in matches:
            odoo_client.ensure_pdf_attachment(move_id, filename, pdf_bytes)
        _clear_state(d, item)
        core.rename_in_drive(d, item["id"], filename)
        log.info(
            "RENAMED %s -> %s (document=%s source=%s confidence=%.1f matches=%d)",
            name, filename, document.odoo_number, source, confidence, len(matches),
        )
        return True

    delivery_candidates = _delivery_order_candidates(pdf_text)
    looks_delivery = _looks_like_delivery_note(pdf_text)
    if image is None:
        image = core.rasterize_first_page(pdf_bytes)
    if image is not None and (looks_delivery or not delivery_candidates):
        header_text, header_confidence = _delivery_header_ocr(image)
        for candidate in _delivery_order_candidates(header_text):
            if candidate not in delivery_candidates:
                delivery_candidates.append(candidate)
        looks_delivery = looks_delivery or _looks_like_delivery_note(header_text)
        if core.DEBUG and header_text:
            log.info(
                "DELIVERY HEADER OCR %s candidates=%s confidence=%.1f text=%r",
                name, delivery_candidates, header_confidence, header_text,
            )

    if looks_delivery:
        if not delivery_candidates:
            _mark_manual_review(d, item, "Fresh Bake delivery note number could not be read")
            return False
        if not odoo_client:
            _mark_awaiting_odoo(d, item, "delivery", delivery_candidates[0], "Odoo unavailable for delivery note")
            return False

        resolved = odoo_client.find_sale_orders_with_customer_invoices(delivery_candidates)
        viable = [row for row in resolved if row.get("invoice_moves")]
        if len(viable) == 1:
            order = viable[0]
            moves = order["invoice_moves"]
            parent_id = (item.get("parents") or [core.FOLDER_ID])[0]
            filename = core.unique_name_in_folder(d, parent_id, _delivery_filename(order["name"], moves))
            for move in moves:
                attachment_id, created = odoo_client.ensure_pdf_attachment(move["id"], filename, pdf_bytes)
                log.info(
                    "%s %s order=%s invoice=%s move_id=%s attachment_id=%s",
                    "ODOO ATTACHED DELIVERY NOTE" if created else "DELIVERY ATTACHMENT ALREADY EXISTS",
                    filename, order["name"], move.get("name"), move["id"], attachment_id,
                )
            _clear_state(d, item)
            core.rename_in_drive(d, item["id"], filename)
            log.info("DELIVERY MATCH %s -> %s order=%s", name, filename, order["name"])
            return True
        if len(viable) > 1:
            _mark_manual_review(d, item, f"delivery note ambiguous; candidates {[row['name'] for row in viable]}")
            return False

        _mark_awaiting_odoo(
            d, item, "delivery", delivery_candidates[0],
            f"delivery order {delivery_candidates[0]} has no related customer invoice",
        )
        return False

    if image is None:
        _mark_manual_review(d, item, "could not rasterize customer claim")
        return False
    texts, claim_confidence = _ocr_full_page_rotations(image)
    claim_norm, rows = _match_claim(texts, claim_index)
    if not claim_norm or not rows:
        _mark_manual_review(
            d, item,
            f"unsupported/unmatched document; no unique Odoo claim_no; OCR confidence {claim_confidence:.1f}",
        )
        return False

    parent_id = (item.get("parents") or [core.FOLDER_ID])[0]
    filename = core.unique_name_in_folder(d, parent_id, _claim_filename(claim_norm, rows))
    _attach_claim_rows(odoo_client, rows, filename, pdf_bytes)
    _clear_state(d, item)
    core.rename_in_drive(d, item["id"], filename)
    log.info(
        "CLAIM MATCH %s -> %s claim=%s attached_to=%s confidence=%.1f",
        name, filename, claim_norm, [row["id"] for row in rows], claim_confidence,
    )
    return True


def main():
    d = core.drive()
    root_id, drive_id = core.resolve_root(d, core.FOLDER_ID)

    odoo_client = None
    if core.ENABLE_ODOO:
        try:
            from odoo_attach import OdooClient
            odoo_client = OdooClient()
            log.info("Odoo integration enabled.")
        except Exception as exc:
            log.warning("Odoo integration failed to init: %s", exc)

    claim_index = _load_claim_index(odoo_client)
    all_items = list(iter_outstanding_pdfs(d, root_id, drive_id))

    deleted_zero = delete_failed = manual_skipped = waiting_skipped = 0
    new_items: List[dict] = []
    retry_items: List[dict] = []

    for item in all_items:
        if _int(item.get("size"), -1) == 0:
            if _delete_zero_byte(d, item):
                deleted_zero += 1
            else:
                delete_failed += 1
            continue

        eligible, reason = _eligibility(item)
        if not eligible:
            if reason == "manual-review":
                manual_skipped += 1
            else:
                waiting_skipped += 1
            continue

        if reason == "new":
            new_items.append(item)
        else:
            retry_items.append(item)

    newest_first = lambda item: item.get("modifiedTime") or ""
    new_items.sort(key=newest_first, reverse=True)
    retry_items.sort(key=newest_first, reverse=True)
    queue = (new_items + retry_items)[:MAX_FILES_PER_RUN]

    processed = renamed = failed = 0
    for item in queue:
        processed += 1
        try:
            if process_one(d, item, odoo_client, claim_index):
                renamed += 1
            else:
                failed += 1
        except Exception as exc:
            failed += 1
            log.exception("PROCESS FAILED %s: %s", item.get("name"), exc)
            try:
                _mark_transient(d, item, f"unexpected error: {exc}")
            except Exception:
                log.exception("Could not save transient state for %s", item.get("name"))

    log.info(
        "Sweep complete: outstanding=%d new=%d retry_ready=%d processed=%d renamed=%d failed=%d "
        "manual_skipped=%d waiting_skipped=%d zero_deleted=%d zero_delete_failed=%d cap=%d parser=%s",
        len(all_items), len(new_items), len(retry_items), processed, renamed, failed,
        manual_skipped, waiting_skipped, deleted_zero, delete_failed,
        MAX_FILES_PER_RUN, PARSER_VERSION,
    )


if __name__ == "__main__":
    main()
