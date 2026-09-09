import os
import unittest

os.environ.setdefault("GDRIVE_FOLDER_ID", "test-folder")
os.environ.setdefault("GDRIVE_SA_JSON", "{}")

from process_invoices import extract_document_number, extract_invoice_number


class InvoiceNumberExtractionTests(unittest.TestCase):
    def test_new_template_ignores_registration_number(self):
        text = (
            "Tax Invoice Reg No: 2011/053816/23 "
            "Tax Invoice Number INV/2026/042130"
        )
        self.assertEqual(extract_invoice_number(text), "2026_042130")

    def test_registration_number_alone_is_not_an_invoice(self):
        self.assertIsNone(
            extract_invoice_number("Tax Invoice Reg No: 2011/053816/23")
        )

    def test_nv_legacy_prefix_must_be_a_separate_token(self):
        self.assertEqual(
            extract_invoice_number("Invoice Number NV/2026/000073"),
            "2026_000073",
        )
        self.assertIsNone(
            extract_invoice_number(
                "Tax Invoice Reg No: 2011/053816/23 Invoice Date 26/07/2026"
            )
        )

    def test_labelled_fallback_zero_pads_on_the_left(self):
        self.assertEqual(
            extract_invoice_number("Tax Invoice Number 2026/42130"),
            "2026_042130",
        )

    def test_common_ocr_digit_substitutions(self):
        self.assertEqual(
            extract_invoice_number("Tax Invoice Number I N V / 2O26 / O4213O"),
            "2026_042130",
        )

    def test_credit_note_number_is_recognised(self):
        doc = extract_document_number("Credit Note Number RINV/2026/08281")
        self.assertIsNotNone(doc)
        self.assertEqual(doc.kind, "credit_note")
        self.assertEqual(doc.odoo_number, "RINV/2026/08281")
        self.assertEqual(doc.filename_stem, "RINV_2026_08281")
        self.assertEqual(doc.move_type, "out_refund")

    def test_credit_note_never_uses_reversal_invoice_reference(self):
        text = (
            "Credit Note Number RINV/2026/08281 Credit Note Date 09/09/2026 "
            "Reference Reversal of: INV/2026/052913"
        )
        doc = extract_document_number(text)
        self.assertEqual(doc.odoo_number, "RINV/2026/08281")

    def test_unreadable_credit_note_does_not_use_reversal_invoice(self):
        text = "Credit Note Number unreadable Reference Reversal of: INV/2026/052913"
        self.assertIsNone(extract_document_number(text))

    def test_credit_note_label_fallback_without_prefix(self):
        doc = extract_document_number("Credit Note Number 2026/8281")
        self.assertEqual(doc.odoo_number, "RINV/2026/08281")


if __name__ == "__main__":
    unittest.main()
