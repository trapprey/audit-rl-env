from datetime import timedelta
from decimal import Decimal

from . import spec
from .money import convert, fx_id
from .timeutil import parse_ts


class PublicIndex:
    def __init__(self, public):
        self.manifest = public["manifest"]
        self.policy = public["policy"]
        self.base = self.manifest["base_currency"]
        self.rows = {r["row_id"]: r for r in public["ledger"]}
        self.docs = {}
        self.invoices = {}
        self.payments = {}
        for doc in public["invoices"]:
            self.docs[doc["doc_id"]] = doc
            self.invoices[doc["invoice_id"]] = doc
        for doc in public["payments"]:
            self.docs[doc["doc_id"]] = doc
            self.payments[doc["payment_id"]] = doc
        self.fx = {fx_id(r["date"], r["source_currency"], r["target_currency"]): r for r in public["fx"]}
        self.start = parse_ts(self.manifest["period_start"])
        self.end = parse_ts(self.manifest["period_end"])
        self.tz = self.start.tzinfo

    def known_ids(self):
        return set(self.rows) | set(self.docs) | set(self.fx) | set(self.invoices) | set(self.payments)

    def in_period(self, text):
        moment = parse_ts(text)
        return self.start <= moment < self.end

    def rate_day(self, text):
        day = parse_ts(text).astimezone(self.tz).date()
        if self.policy["fx"]["rate_date"] == "previous_local_date":
            day -= timedelta(days=1)
        return day

    def rates_from(self, ids, day, currencies):
        table = {self.base: Decimal(1)}
        for cur in currencies:
            if cur == self.base:
                continue
            key = fx_id(day.isoformat(), cur, self.base)
            if key not in ids or key not in self.fx:
                return None
            table[cur] = Decimal(self.fx[key]["rate"])
        return table


def _one(items):
    return items[0] if len(items) == 1 else None


def _duplicate(idx, issue, ids):
    docs = [idx.docs[i] for i in ids if i in idx.docs and "invoice_id" in idx.docs[i]]
    if len(docs) != 2:
        return None
    first, second = docs
    if sorted([first["invoice_id"], second["invoice_id"]]) != issue["affected_ids"]:
        return None
    for key in ("entity_id", "currency", "total_minor", "line_items"):
        if first[key] != second[key]:
            return None
    if first["currency"] != issue["currency"] or first["issued_at"] == second["issued_at"]:
        return None
    return first["total_minor"]


def _payment_and_invoice(idx, issue, ids):
    payment = _one([idx.docs[i] for i in ids if i in idx.docs and "payment_id" in idx.docs[i]])
    if payment is None or payment["payment_id"] not in issue["affected_ids"]:
        return None, None
    rest = [a for a in issue["affected_ids"] if a != payment["payment_id"]]
    if len(rest) != 1:
        return None, None
    return payment, rest[0]


def _allocation(idx, issue, ids):
    payment, invoice_id = _payment_and_invoice(idx, issue, ids)
    if payment is None or not payment["invoice_refs"]:
        return None
    pid, cur = payment["payment_id"], payment["currency"]
    rows = [idx.rows[i] for i in ids if i in idx.rows and idx.rows[i]["payment_id"] == pid]
    unapplied = _one([r for r in rows if r["account"] == spec.UNAPPLIED and r["credit_minor"] > 0])
    if unapplied is None or unapplied["currency"] != cur:
        return None
    applied = {}
    for row in rows:
        if row["account"] in (spec.AR, spec.FX_CLEARING) and row["credit_minor"] > 0 and row["currency"] == cur:
            applied[row["invoice_id"]] = applied.get(row["invoice_id"], 0) + row["credit_minor"]
    target = None
    for ref in payment["invoice_refs"]:
        if ref["invoice_id"] == invoice_id:
            target = ref["amount_minor"]
        elif applied.get(ref["invoice_id"], 0) != ref["amount_minor"]:
            return None
    if target is None:
        return None
    shortfall = target - applied.get(invoice_id, 0)
    if shortfall <= 0 or shortfall != unapplied["credit_minor"] or cur != issue["currency"]:
        return None
    return shortfall


def _fx(idx, issue, ids):
    payment, invoice_id = _payment_and_invoice(idx, issue, ids)
    if payment is None:
        return None
    pid, cur = payment["payment_id"], payment["currency"]
    rows = [
        idx.rows[i] for i in ids
        if i in idx.rows and idx.rows[i]["payment_id"] == pid and idx.rows[i]["invoice_id"] == invoice_id
    ]
    out = _one([r for r in rows if r["account"] == spec.FX_CLEARING and r["credit_minor"] > 0 and r["currency"] == cur])
    inside = _one([
        r for r in rows
        if r["currency"] != cur and (
            (r["account"] == spec.AR and r["credit_minor"] > 0)
            or (r["account"] == spec.FX_CLEARING and r["debit_minor"] > 0)
        )
    ])
    if out is None or inside is None:
        return None
    target_cur = inside["currency"]
    booked = inside["credit_minor"] or inside["debit_minor"]
    rates = idx.rates_from(ids, idx.rate_day(payment["timestamp"]), [cur, target_cur])
    if rates is None or target_cur != issue["currency"]:
        return None
    difference = abs(booked - convert(out["credit_minor"], cur, target_cur, rates))
    if difference <= spec.AMOUNT_TOLERANCE_MINOR:
        return None
    return difference


def _cutoff(idx, issue, ids):
    invoice = _one([idx.docs[i] for i in ids if i in idx.docs and "invoice_id" in idx.docs[i]])
    if invoice is None or [invoice["invoice_id"]] != issue["affected_ids"]:
        return None
    if idx.in_period(invoice["issued_at"]):
        return None
    rows = [
        idx.rows[i] for i in ids
        if i in idx.rows and idx.rows[i]["invoice_id"] == invoice["invoice_id"] and (
            (idx.rows[i]["account"] == spec.AR and idx.rows[i]["debit_minor"] > 0)
            or (idx.rows[i]["account"] == spec.REVENUE and idx.rows[i]["credit_minor"] > 0)
        )
    ]
    if not rows:
        return None
    for row in rows:
        if not idx.in_period(row["timestamp"]):
            return None
        if (row["debit_minor"] or row["credit_minor"]) != invoice["total_minor"]:
            return None
    if invoice["currency"] != issue["currency"]:
        return None
    return invoice["total_minor"]


CHECKS = {
    "duplicate_invoice": _duplicate,
    "payment_allocation": _allocation,
    "fx_rate": _fx,
    "period_cutoff": _cutoff,
}


def verify_truth(public, truth):
    idx = PublicIndex(public)
    problems = []
    keys = set()
    for issue in truth["issues"]:
        label = issue["issue_id"]
        key = (issue["type"], tuple(issue["affected_ids"]))
        if key in keys:
            problems.append(f"{label}: canonical key repeats")
        keys.add(key)
        if not issue["evidence_sets"]:
            problems.append(f"{label}: no evidence set")
        for group in issue["evidence_sets"]:
            if len(group) > spec.MAX_EVIDENCE:
                problems.append(f"{label}: evidence set too large")
                continue
            missing = [i for i in group if i not in idx.known_ids()]
            if missing:
                problems.append(f"{label}: evidence {missing} is not visible")
                continue
            amount = CHECKS[issue["type"]](idx, issue, set(group))
            if amount is None:
                problems.append(f"{label}: evidence {group} does not establish the issue")
            elif amount != issue["expected_amount_minor"]:
                problems.append(f"{label}: evidence gives {amount}, truth says {issue['expected_amount_minor']}")
    return problems
