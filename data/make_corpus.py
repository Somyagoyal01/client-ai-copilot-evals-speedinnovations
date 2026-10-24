"""Generate a small SYNTHETIC client knowledge base (ERP + banking ops docs) and ERP records.

Everything here is invented for the demo. No real company or customer data.
Run: python -m data.make_corpus  (writes data/docs/*.md and data/erp.json)
"""
import json
import random
from pathlib import Path

ROOT = Path(__file__).parent
DOCS = {
    "ap_invoice_policy.md": """# Accounts Payable: Invoice Approval Policy (v3.2)
## Approval limits
Invoices up to INR 50,000 can be approved by the department manager.
Invoices between INR 50,001 and INR 5,00,000 need approval from the Finance Controller.
Invoices above INR 5,00,000 need approval from the CFO and a signed purchase order.
## Three-way match
Every invoice must match the purchase order and the goods receipt note (GRN) before payment.
A price variance of up to 2% is tolerated; above 2% the invoice is put on hold.
## Payment terms
Standard vendor payment terms are Net 45 days from the invoice date.
MSME-registered vendors must be paid within 45 days as required by the MSMED Act.
""",
    "vendor_onboarding_sop.md": """# Vendor Onboarding SOP
## Documents required
A new vendor must submit a GST certificate, PAN, a cancelled cheque and a signed code of conduct.
## Verification
The procurement team verifies GSTIN on the GST portal within 2 working days.
Bank details are verified by a penny-drop test before the first payment.
## Blocking
A vendor is blocked automatically if their GST status becomes inactive.
""",
    "loan_product_note.md": """# Retail Loan Product Note: Personal Loan Plus
## Eligibility
Applicants must be between 23 and 58 years old with a minimum net monthly income of INR 30,000.
A CIBIL score of 700 or above is required.
## Pricing
Interest rates range from 11.5% to 16% per annum depending on the risk band.
The processing fee is 1.5% of the loan amount, capped at INR 10,000.
## Prepayment
Prepayment is allowed after 6 EMIs with a 3% foreclosure charge.
""",
    "kyc_refresh_guideline.md": """# KYC Periodic Refresh Guideline
## Frequency
High-risk customers are re-verified every 2 years, medium-risk every 8 years and low-risk every 10 years.
## Channels
Customers can submit a self-declaration through net banking if there is no change in details.
## Escalation
If KYC is not refreshed within 30 days of the due date, the account is moved to partial freeze.
""",
    "erp_month_end_close.md": """# ERP Month-End Close Checklist
## Timeline
The general ledger is closed on the 3rd working day of the following month.
Accruals must be posted by the 2nd working day.
## Reconciliations
Bank reconciliation is done daily; intercompany reconciliation is done monthly.
## Sign-off
The Finance Controller signs off the trial balance before the books are locked.
""",
}


def make_erp(seed: int = 7):
    rnd = random.Random(seed)
    vendors = ["Apex Polymers", "Sahyadri Logistics", "Nimbus Office Supplies", "Kaveri Steel", "BluePeak IT"]
    invoices = []
    for i in range(1, 41):
        amt = rnd.choice([18_500, 42_000, 75_000, 1_20_000, 3_40_000, 6_80_000, 9_95_000])
        invoices.append({
            "invoice_id": f"INV-{1000 + i}",
            "vendor": rnd.choice(vendors),
            "amount_inr": amt,
            "status": rnd.choice(["paid", "on_hold", "pending_approval"]),
            "po_number": f"PO-{5000 + rnd.randint(1, 60)}",
        })
    terms = {v: {"payment_terms_days": rnd.choice([30, 45, 60]), "msme": rnd.random() < 0.4} for v in vendors}
    return {"invoices": invoices, "vendor_terms": terms}


def main():
    (ROOT / "docs").mkdir(exist_ok=True)
    for name, text in DOCS.items():
        (ROOT / "docs" / name).write_text(text, encoding="utf-8")
    (ROOT / "erp.json").write_text(json.dumps(make_erp(), indent=1), encoding="utf-8")
    print(f"wrote {len(DOCS)} docs and erp.json")


if __name__ == "__main__":
    main()
