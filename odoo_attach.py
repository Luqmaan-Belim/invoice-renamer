import os
import base64
import re
import xmlrpc.client
from typing import List, Dict, Any


def to_odoo_invoice_name(scan_inv: str) -> str:
    """
    Convert '2026_000073' (or '2026_73') -> 'INV/2026/000073'
    Keeps it strict and predictable.
    """
    m = re.fullmatch(r"(\d{4})_(\d{1,6})", scan_inv.strip())
    if not m:
        # If it doesn't match expected pattern, just return as-is
        # (caller can decide whether to search differently)
        return scan_inv.strip()

    year = m.group(1)
    seq = m.group(2).zfill(6)
    return f"INV/{year}/{seq}"


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

    def search_customer_invoice_by_number(self, odoo_number: str) -> List[int]:
        """
        Customer invoices are account.move with move_type = out_invoice.
        Match by the posted invoice number in 'name' (e.g. 'INV/2026/000073').
        """
        domain = [
            ("move_type", "=", "out_invoice"),
            ("name", "=", odoo_number),
        ]
        return self.models.execute_kw(
            self.db, self.uid, self.api_key,
            "account.move", "search",
            [domain],
            {"limit": 5}
        )

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
            "ir.attachment", "create",
            [vals]
        )
