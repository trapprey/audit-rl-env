from collections import Counter, defaultdict

from . import spec
from .reconcile import Books, audit
from .schema import validate
from .timeutil import parse_ts
from .verify import PublicIndex, verify_truth

OPS = {
    "insert_duplicate_invoice": "duplicate_invoice",
    "reduce_allocation": "payment_allocation",
    "wrong_rate_date": "fx_rate",
    "early_posting": "period_cutoff",
}


def _schemas(public, truth):
    errors = validate(public["manifest"], "manifest") + validate(public["policy"], "policy") + validate(truth, "truth")
    for table, name in (("ledger", "ledger_row"), ("invoices", "invoice"), ("payments", "payment"), ("fx", "fx_row")):
        for record in public[table]:
            errors += validate(record, name)
    return errors


def _counts(public, truth):
    manifest = public["manifest"]
    errors = [
        f"{table}: manifest says {manifest['record_counts'][table]}, file has {len(public[table])}"
        for table in spec.TABLES
        if manifest["record_counts"][table] != len(public[table])
    ]
    if manifest["document_count"] != len(public["invoices"]) + len(public["payments"]):
        errors.append("document_count does not match")
    if truth["episode_id"] != manifest["episode_id"]:
        errors.append("truth belongs to another episode")
    return errors


def _identifiers(public):
    errors = []
    ids = Counter(
        [r["row_id"] for r in public["ledger"]]
        + [d["doc_id"] for d in public["invoices"] + public["payments"]]
        + [d["invoice_id"] for d in public["invoices"]]
        + [d["payment_id"] for d in public["payments"]]
    )
    errors += [f"id {i} is used {n} times" for i, n in ids.items() if n > 1]
    invoices = {d["invoice_id"] for d in public["invoices"]}
    payments = {d["payment_id"] for d in public["payments"]}
    entities = {d["entity_id"] for d in public["invoices"] + public["payments"]}
    for row in public["ledger"]:
        if row["invoice_id"] is not None and row["invoice_id"] not in invoices:
            errors.append(f"{row['row_id']} points to a missing invoice")
        if row["payment_id"] is not None and row["payment_id"] not in payments:
            errors.append(f"{row['row_id']} points to a missing payment")
        if row["entity_id"] not in entities:
            errors.append(f"{row['row_id']} points to a missing entity")
    for doc in public["payments"]:
        for ref in doc["invoice_refs"] or []:
            if ref["invoice_id"] not in invoices:
                errors.append(f"{doc['payment_id']} references a missing invoice")
        if doc["refund_of"] is not None and doc["refund_of"] not in payments:
            errors.append(f"{doc['payment_id']} refunds a missing payment")
    return errors


def _equalities(public):
    errors = []
    start = parse_ts(public["manifest"]["period_start"])
    end = parse_ts(public["manifest"]["period_end"])
    totals = defaultdict(lambda: [0, 0])
    by_payment = defaultdict(list)
    by_invoice = defaultdict(list)
    for row in public["ledger"]:
        debit, credit = row["debit_minor"], row["credit_minor"]
        if (debit > 0) == (credit > 0):
            errors.append(f"{row['row_id']} must have exactly one positive side")
        if not start <= parse_ts(row["timestamp"]) < end:
            errors.append(f"{row['row_id']} is outside the period")
        totals[row["currency"]][0] += debit
        totals[row["currency"]][1] += credit
        if row["payment_id"]:
            by_payment[row["payment_id"]].append(row)
        elif row["invoice_id"]:
            by_invoice[row["invoice_id"]].append(row)
    for cur, (debit, credit) in sorted(totals.items()):
        if debit != credit:
            errors.append(f"{cur}: debits {debit} != credits {credit}")
    docs = {d["invoice_id"]: d for d in public["invoices"]}
    for invoice_id, rows in by_invoice.items():
        ar = sum(r["debit_minor"] for r in rows if r["account"] == spec.AR)
        revenue = sum(r["credit_minor"] for r in rows if r["account"] == spec.REVENUE)
        if not ar == revenue == docs[invoice_id]["total_minor"]:
            errors.append(f"invoice {invoice_id} is booked for a different amount than its document")
    for doc in public["payments"]:
        rows = by_payment.get(doc["payment_id"], [])
        if not rows:
            if start <= parse_ts(doc["timestamp"]) < end:
                errors.append(f"payment {doc['payment_id']} in the period is not booked")
            continue
        bank = [r for r in rows if r["account"] == spec.BANK]
        signed = sum(r["debit_minor"] - r["credit_minor"] for r in bank)
        if signed != doc["amount_minor"]:
            errors.append(f"payment {doc['payment_id']} bank amount differs from its document")
        if doc["kind"] == "receipt":
            outgoing = sum(
                r["credit_minor"] for r in rows
                if r["currency"] == doc["currency"] and r["account"] in (spec.AR, spec.FX_CLEARING, spec.UNAPPLIED)
            )
            if outgoing != doc["amount_minor"]:
                errors.append(f"receipt {doc['payment_id']} is not fully accounted for")
    return errors


def _canonical(items, amount_key):
    return Counter((i["type"], tuple(sorted(i["affected_ids"])), i[amount_key], i["currency"].upper()) for i in items)


def _reconciliation(public, truth):
    found = _canonical(audit(Books(public["manifest"], public["policy"], public)), "claimed_amount_minor")
    declared = _canonical(truth["issues"], "expected_amount_minor")
    errors = [f"undeclared discrepancy {k}" for k in found - declared]
    errors += [f"declared discrepancy not present in the books {k}" for k in declared - found]
    return errors


def _provenance(public, truth, provenance):
    mutations = provenance.get("mutations")
    if mutations is None:
        return ["provenance has no mutations"]
    errors = []
    declared = Counter(i["type"] for i in truth["issues"])
    recorded = Counter(OPS.get(m.get("op"), "unknown") for m in mutations)
    if declared != recorded:
        errors.append(f"mutations {dict(recorded)} do not match issues {dict(declared)}")
    rows = {r["row_id"] for r in public["ledger"]}
    docs = {d["doc_id"] for d in public["invoices"] + public["payments"]}
    affected = {i for issue in truth["issues"] for i in issue["affected_ids"]}
    for mutation in mutations:
        for row in mutation.get("ledger_rows", []):
            if row not in rows:
                errors.append(f"{mutation['op']} names a missing ledger row")
        for doc in mutation.get("documents", []):
            if doc not in docs:
                errors.append(f"{mutation['op']} names a missing document")
        for key in ("original", "duplicate", "payment", "invoice"):
            if key in mutation and mutation[key] not in affected:
                errors.append(f"{mutation['op']} touches {key} outside the declared issues")
    return errors


CHECKS = [
    ("schemas", lambda p, t, v: _schemas(p, t)),
    ("record counts and episode id", lambda p, t, v: _counts(p, t)),
    ("identifiers and references", lambda p, t, v: _identifiers(p)),
    ("bookkeeping equalities", lambda p, t, v: _equalities(p)),
    ("evidence is visible and sufficient", lambda p, t, v: verify_truth(p, t)),
    ("books differ from policy exactly by declared issues", lambda p, t, v: _reconciliation(p, t)),
    ("provenance matches truth", lambda p, t, v: _provenance(p, t, v)),
]


def check_episode(episode):
    results = []
    for name, check in CHECKS:
        try:
            problems = check(episode.public, episode.truth, episode.provenance)
        except Exception as exc:
            problems = [f"check crashed: {type(exc).__name__}: {exc}"]
        results.append((name, problems))
    return results


__all__ = ["check_episode", "PublicIndex"]
