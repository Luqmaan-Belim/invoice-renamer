import base64
import os
import re
import xmlrpc.client
from datetime import date, timedelta
from typing import Any, Dict, List, Tuple


def to_odoo_invoice_name(scan_inv: str) -> str:
    """Backward-compatible conversion for legacy callers."""
    match = re.fullmatch(r"(\d{4})_(\d{1,6})", scan_inv.strip())
    if not match:
        return scan_inv.strip()
    return f"INV/{match.group(1)}/{match.group(2).zfill(6)}"


class OdooClient:
    def __init__(self):
        self.url = os.environ["ODOO_URL"].rstrip("/")
        self.db = os.environ["ODOO_DB"]
        self.user = os.environ["ODOO_USER"]
        self.api_key = os.environ["ODOO_API_KEY"]

        common = xmlrpc.client.ServerProxy(f"{self.url}/xmlrpc/2/common")
        self.uid = common.authenticate(self.db, self.user, self.api_key, {})
        if not self.uid:
            raise RuntimeError("Odoo authentication failed (check ODOO_DB/USER/API_KEY).")
        self.models = xmlrpc.client.ServerProxy(f"{self.url}/xmlrpc/2/object")

    def search_customer_document_by_number(
        self, odoo_number: str, move_type: str
    ) -> List[int]:
        """Return every customer invoice/credit-note match, regardless of state."""
        domain = [
            ("move_type", "=", move_type),
            ("name", "=", odoo_number),
        ]
        return self.models.execute_kw(
            self.db, self.uid, self.api_key,
            "account.move", "search", [domain]
        )

    def search_customer_invoice_by_number(self, odoo_number: str) -> List[int]:
        """Backward-compatible invoice-only helper."""
        return self.search_customer_document_by_number(odoo_number, "out_invoice")

    def list_credit_note_claims(self, lookback_days: int = 120) -> List[Dict[str, Any]]:
        """
        Return customer credit notes that have claim_no populated.

        No state filter is applied: draft and posted credit notes are both included.
        The lookback keeps the every-minute lookup small while still covering the
        normal scanned-claim backlog. Set lookback_days <= 0 to query all dates.
        """
        domain: List[Any] = [
            ("move_type", "=", "out_refund"),
            ("claim_no", "!=", False),
        ]
        if lookback_days > 0:
            cutoff = (date.today() - timedelta(days=lookback_days)).isoformat()
            domain.append(("invoice_date", ">=", cutoff))

        return self.models.execute_kw(
            self.db, self.uid, self.api_key,
            "account.move", "search_read", [domain],
            {
                "fields": ["id", "name", "claim_no", "partner_id", "invoice_date"],
                "order": "invoice_date desc, id desc",
                "limit": 50000,
            },
        )

    def find_sale_orders_with_customer_invoices(
        self, order_names: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Resolve scanned Fresh Bake Delivery Note / Sale Order numbers to invoices.

        The Delivery Note Number printed by Fresh Bake is the sale.order name
        (for example S113982). We deliberately do not filter invoice state: draft
        and posted customer invoices are both valid attachment targets.
        """
        names = sorted({name.strip() for name in order_names if name and name.strip()})
        if not names:
            return []

        orders = self.models.execute_kw(
            self.db, self.uid, self.api_key,
            "sale.order", "search_read", [[("name", "in", names)]],
            {"fields": ["id", "name", "invoice_ids"], "limit": 100},
        )
        if not orders:
            return []

        invoice_ids = sorted({
            move_id
            for order in orders
            for move_id in (order.get("invoice_ids") or [])
        })

        moves_by_id: Dict[int, Dict[str, Any]] = {}
        if invoice_ids:
            moves = self.models.execute_kw(
                self.db, self.uid, self.api_key,
                "account.move", "search_read",
                [[("id", "in", invoice_ids), ("move_type", "=", "out_invoice")]],
                {"fields": ["id", "name", "state", "invoice_origin"], "limit": 1000},
            )
            moves_by_id = {move["id"]: move for move in moves}

        results: List[Dict[str, Any]] = []
        for order in orders:
            move_rows = [
                moves_by_id[move_id]
                for move_id in (order.get("invoice_ids") or [])
                if move_id in moves_by_id
            ]

            # Fallback for custom invoice creation paths where invoice_ids has not
            # been populated but invoice_origin still carries the SO number.
            if not move_rows:
                fallback = self.models.execute_kw(
                    self.db, self.uid, self.api_key,
                    "account.move", "search_read",
                    [[
                        ("move_type", "=", "out_invoice"),
                        ("invoice_origin", "ilike", order["name"]),
                    ]],
                    {"fields": ["id", "name", "state", "invoice_origin"], "limit": 100},
                )
                move_rows = fallback

            enriched = dict(order)
            enriched["invoice_moves"] = move_rows
            results.append(enriched)

        return results

    def ensure_pdf_attachment(
        self, move_id: int, filename: str, pdf_bytes: bytes
    ) -> Tuple[int, bool]:
        """Return (attachment_id, created). Safe to call again after a retry."""
        domain = [
            ("res_model", "=", "account.move"),
            ("res_id", "=", move_id),
            ("name", "=", filename),
            ("mimetype", "=", "application/pdf"),
        ]
        existing = self.models.execute_kw(
            self.db, self.uid, self.api_key,
            "ir.attachment", "search", [domain], {"limit": 1}
        )
        if existing:
            return existing[0], False
        return self.attach_pdf_to_move(move_id, filename, pdf_bytes), True

    def attach_pdf_to_move(self, move_id: int, filename: str, pdf_bytes: bytes) -> int:
        vals: Dict[str, Any] = {
            "name": filename,
            "type": "binary",
            "datas": base64.b64encode(pdf_bytes).decode("ascii"),
            "res_model": "account.move",
            "res_id": move_id,
            "mimetype": "application/pdf",
        }
        return self.models.execute_kw(
            self.db, self.uid, self.api_key,
            "ir.attachment", "create", [vals]
        )
