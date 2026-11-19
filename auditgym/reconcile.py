from collections import defaultdict
from datetime import timedelta
from decimal import Decimal

from . import spec
from .money import convert, fx_id
from .timeutil import parse_ts


def finding(kind, affected, amount, currency, evidence, explanation):
    return {
        "type": kind,
        "affected_ids": sorted(set(affected)),
        "claimed_amount_minor": int(amount),
        "currency": currency,
        "evidence_ids": sorted(set(evidence))[:10],
        "explanation": explanation[:1000],
    }


class Books:
    def __init__(self, manifest, policy, data):
        self.manifest = manifest
        self.policy = policy
        self.base = manifest["base_currency"]
        self.start = parse_ts(manifest["period_start"])
        self.end = parse_ts(manifest["period_end"])
        self.tz = self.start.tzinfo
        self.ledger = data.get("ledger", [])
        self.invoices = data.get("invoices", [])
        self.payments = data.get("payments", [])
        self.fx = {}
        for row in data.get("fx", []):
            self.fx[row.get("fx_id") or fx_id(row["date"], row["source_currency"], row["target_currency"])] = row
        self.invoice_by_id = {d["invoice_id"]: d for d in self.invoices}

    def in_period(self, text):
        return self.start <= parse_ts(text) < self.end

    def rate_day(self, text):
        day = parse_ts(text).astimezone(self.tz).date()
        if self.policy["fx"]["rate_date"] == "previous_local_date":
            day -= timedelta(days=1)
        return day

    def rates(self, day, currencies):
        table = {self.base: Decimal(1)}
        keys = []
        for cur in currencies:
            if cur == self.base:
                continue
            key = fx_id(day.isoformat(), cur, self.base)
            if key not in self.fx:
                return None, []
            table[cur] = Decimal(self.fx[key]["rate"])
            keys.append(key)
        return table, keys


def audit(books):
    rows_by_payment = defaultdict(list)
    rows_by_invoice = defaultdict(list)
    for row in books.ledger:
        if row["payment_id"]:
            rows_by_payment[row["payment_id"]].append(row)
        if row["invoice_id"]:
            rows_by_invoice[row["invoice_id"]].append(row)
    out = []
    out += _duplicates(books, rows_by_invoice)
    out += _allocations(books, rows_by_payment)
    out += _fx(books, rows_by_payment)
    out += _cutoffs(books, rows_by_invoice)
    return out


def _booked_rows(rows_by_invoice, invoice_id):
    return [
        r for r in rows_by_invoice.get(invoice_id, [])
        if r["account"] == spec.AR and r["debit_minor"] > 0 and r["payment_id"] is None
    ]


def _duplicates(books, rows_by_invoice):
    groups = defaultdict(list)
    for doc in books.invoices:
        if not _booked_rows(rows_by_invoice, doc["invoice_id"]):
            continue
        key = (doc["entity_id"], doc["currency"], doc["total_minor"], repr(doc["line_items"]))
        groups[key].append(doc)
    out = []
    for docs in groups.values():
        if len(docs) < 2:
            continue
        docs.sort(key=lambda d: (d["issued_at"], d["invoice_id"]))
        first = docs[0]
        for other in docs[1:]:
            out.append(finding(
                "duplicate_invoice",
                [first["invoice_id"], other["invoice_id"]],
                other["total_minor"],
                other["currency"],
                [first["doc_id"], other["doc_id"]],
                "Same customer, currency, total and line items including service periods were booked twice.",
            ))
    return out


def _application_rows(payment, rows):
    return [
        r for r in rows
        if r["account"] in (spec.AR, spec.FX_CLEARING) and r["credit_minor"] > 0
        and r["currency"] == payment["currency"] and r["invoice_id"] is not None
    ]


def _allocations(books, rows_by_payment):
    out = []
    for payment in books.payments:
        if payment["kind"] != "receipt" or not payment["invoice_refs"]:
            continue
        rows = rows_by_payment.get(payment["payment_id"], [])
        if not rows:
            continue
        applied_rows = _application_rows(payment, rows)
        applied = defaultdict(int)
        for row in applied_rows:
            applied[row["invoice_id"]] += row["credit_minor"]
        unapplied = [r for r in rows if r["account"] == spec.UNAPPLIED and r["credit_minor"] > 0]
        for ref in payment["invoice_refs"]:
            shortfall = ref["amount_minor"] - applied.get(ref["invoice_id"], 0)
            if shortfall <= 0 or not unapplied:
                continue
            evidence = [payment["doc_id"]] + [r["row_id"] for r in unapplied] + [r["row_id"] for r in applied_rows]
            out.append(finding(
                "payment_allocation",
                [payment["payment_id"], ref["invoice_id"]],
                shortfall,
                payment["currency"],
                evidence,
                f"Remittance applies {ref['amount_minor']} to the invoice but only {applied.get(ref['invoice_id'], 0)} was applied; the rest sits in 2400.",
            ))
    return out


def _fx(books, rows_by_payment):
    out = []
    for payment in books.payments:
        if payment["kind"] != "receipt" or not payment["invoice_refs"]:
            continue
        rows = rows_by_payment.get(payment["payment_id"], [])
        for ref in payment["invoice_refs"]:
            iid = ref["invoice_id"]
            outgoing = [
                r for r in rows
                if r["invoice_id"] == iid and r["account"] == spec.FX_CLEARING
                and r["credit_minor"] > 0 and r["currency"] == payment["currency"]
            ]
            incoming = [
                r for r in rows
                if r["invoice_id"] == iid and r["account"] == spec.AR
                and r["credit_minor"] > 0 and r["currency"] != payment["currency"]
            ]
            if len(outgoing) != 1 or len(incoming) != 1:
                continue
            target = incoming[0]["currency"]
            rates, keys = books.rates(books.rate_day(payment["timestamp"]), [payment["currency"], target])
            if rates is None:
                continue
            expected = convert(outgoing[0]["credit_minor"], payment["currency"], target, rates)
            difference = abs(incoming[0]["credit_minor"] - expected)
            if difference <= spec.AMOUNT_TOLERANCE_MINOR:
                continue
            out.append(finding(
                "fx_rate",
                [payment["payment_id"], iid],
                difference,
                target,
                [payment["doc_id"], outgoing[0]["row_id"], incoming[0]["row_id"], *keys],
                f"Policy rate gives {expected} {target}; ledger booked {incoming[0]['credit_minor']}.",
            ))
    return out


def _cutoffs(books, rows_by_invoice):
    out = []
    for doc in books.invoices:
        if books.in_period(doc["issued_at"]):
            continue
        booked = [r for r in _booked_rows(rows_by_invoice, doc["invoice_id"]) if books.in_period(r["timestamp"])]
        if not booked:
            continue
        out.append(finding(
            "period_cutoff",
            [doc["invoice_id"]],
            doc["total_minor"],
            doc["currency"],
            [doc["doc_id"], booked[0]["row_id"]],
            "Invoice is issued outside the period but its receivable is booked inside it.",
        ))
    return out
