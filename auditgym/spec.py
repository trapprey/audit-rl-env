SCHEMA_VERSION = "1.0"
GENERATOR_VERSION = "0.4.1"
ENV_VERSION = "0.4.1"

FINDING_TYPES = ("duplicate_invoice", "payment_allocation", "fx_rate", "period_cutoff")
INTERACTIONS = ("fx_partial_boundary", "duplicate_fx")
TABLES = ("ledger", "invoices", "payments", "fx")

MAX_STEPS = 100
PAGE_LIMIT = 50
MAX_FINDINGS = 30
MAX_EVIDENCE = 10
MAX_AFFECTED = 10
MAX_EXPLANATION = 1000
MAX_AMOUNT = 10**15
AMOUNT_TOLERANCE_MINOR = 1
STEP_COST = 0.002
MAX_ISSUES = 12

BANK = "1010"
AR = "1200"
FX_CLEARING = "1950"
UNAPPLIED = "2400"
REVENUE = "4000"

ACCOUNTS = {
    BANK: "Bank",
    AR: "Accounts receivable",
    FX_CLEARING: "Currency exchange clearing",
    UNAPPLIED: "Unapplied customer receipts",
    REVENUE: "Revenue",
}

LEDGER_COLUMNS = (
    "row_id",
    "timestamp",
    "account",
    "debit_minor",
    "credit_minor",
    "currency",
    "entity_id",
    "invoice_id",
    "payment_id",
)
FX_COLUMNS = ("date", "source_currency", "target_currency", "rate")
SUBMISSION_KEYS = ("episode_id", "findings")
FINDING_KEYS = ("type", "affected_ids", "claimed_amount_minor", "currency", "evidence_ids", "explanation")

FINDING_RULES = {
    "duplicate_invoice": {
        "meaning": "The same invoice is issued and booked twice under two invoice numbers.",
        "affected_ids": "invoice_id of the original and of the duplicate",
        "amount": "total_minor of the duplicate, in invoice currency",
        "evidence": "doc_id of both invoice documents",
    },
    "payment_allocation": {
        "meaning": "A receipt references an invoice but the ledger applied less than the referenced amount.",
        "affected_ids": "payment_id and invoice_id",
        "amount": "referenced amount that was not applied, in payment currency",
        "evidence": "payment doc_id, its 2400 ledger row and every ledger row that applied this payment in payment currency",
    },
    "fx_rate": {
        "meaning": "A cross-currency receipt was converted with a rate other than the one the policy requires.",
        "affected_ids": "payment_id and invoice_id",
        "amount": "absolute difference between the booked invoice-currency amount and the policy conversion, in invoice currency",
        "evidence": "payment doc_id, the 1950 payment-currency row, the 1200 or 1950 invoice-currency row, and fx_id of every non-base rate used",
    },
    "period_cutoff": {
        "meaning": "An invoice issued outside the reporting period was booked inside it.",
        "affected_ids": "invoice_id",
        "amount": "total_minor of the invoice, in invoice currency",
        "evidence": "invoice doc_id and the 1200 or 4000 ledger row that booked it",
    },
}
