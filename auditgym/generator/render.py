from datetime import date, datetime

from .. import spec
from ..money import EXPONENTS, fx_id
from ..timeutil import fmt_ts, offset_text
from .world import postings

REF_TAGS = {
    "invoice", "payment", "inv_doc", "pay_doc", "fx", "inv_ar", "inv_rev", "pay_bank", "alloc_ar",
    "alloc_clear_out", "alloc_clear_in", "unapplied", "refund_unapplied", "refund_bank",
}
HEX = "0123456789abcdef"
DIGITS = "0123456789"


class IdFactory:
    def __init__(self, rng, year):
        self.rng = rng
        self.year = year
        self.used = set()
        self.entity_style = rng.randrange(3)
        self.invoice_style = rng.randrange(3)
        self.payment_style = rng.randrange(3)
        self.doc_style = rng.randrange(2)
        self.row_style = rng.randrange(2)
        self.entity_seq = rng.randint(10, 400)
        self.invoice_seq = rng.randint(100, 8000)
        self.row_seq = rng.randint(1, 90000)
        self.letters = "".join(rng.choice("ABCDEFGHJKLMNPRSTUVWXZ") for _ in range(2))

    def _pick(self, alphabet, size):
        return "".join(self.rng.choice(alphabet) for _ in range(size))

    def _unique(self, make):
        for _ in range(1000):
            value = make()
            if value not in self.used:
                self.used.add(value)
                return value
        raise RuntimeError("id space exhausted")

    def entity(self):
        if self.entity_style == 0:
            return self._unique(lambda: f"C-{self._pick(DIGITS, 5)}")
        if self.entity_style == 1:
            self.entity_seq += 1
            return self._unique(lambda: f"CUST{self.entity_seq:04d}")
        return self._unique(lambda: self._pick("BCDFGHKMNPRSTVZ", 3) + self._pick(DIGITS, 2))

    def invoice(self):
        self.invoice_seq += 1
        n = self.invoice_seq
        if self.invoice_style == 0:
            return self._unique(lambda: f"INV-{self.year}-{n:05d}")
        if self.invoice_style == 1:
            return self._unique(lambda: f"SI{n:07d}")
        return self._unique(lambda: f"{self.letters}/{self.year % 100:02d}/{n:05d}")

    def payment(self):
        if self.payment_style == 0:
            return self._unique(lambda: f"PMT-{self._pick(HEX, 6).upper()}")
        if self.payment_style == 1:
            return self._unique(lambda: f"RC{self._pick(DIGITS, 7)}")
        return self._unique(lambda: f"BNK-{self._pick(DIGITS, 4)}-{self._pick(DIGITS, 4)}")

    def doc(self):
        if self.doc_style == 0:
            return self._unique(lambda: f"D{self._pick(HEX, 9)}")
        return self._unique(lambda: f"DOC-{self._pick(HEX, 4)}-{self._pick(HEX, 4)}")

    def row(self):
        if self.row_style == 0:
            self.row_seq += 1
            return self._unique(lambda: f"GL{self.row_seq:06d}")
        return self._unique(lambda: f"JE-{self._pick(HEX, 7)}")


def build_policy(world):
    rule_text = {
        "transaction_local_date": "Use the rate dated on the receipt's calendar date in the company timezone.",
        "previous_local_date": "Use the rate dated on the calendar day before the receipt's date in the company timezone.",
    }[world.rate_rule]
    return {
        "period": {
            "start": fmt_ts(world.period_start, world.offset),
            "end": fmt_ts(world.period_end, world.offset),
            "end_exclusive": True,
            "timezone": offset_text(world.offset),
            "rule": "A record belongs to the period when its instant is on or after start and before end. "
                    "Compare instants, not the calendar date printed in another offset.",
        },
        "fx": {
            "table": "fx",
            "quote": "1 unit of source_currency = rate units of target_currency; target_currency is the base currency.",
            "rate_date": world.rate_rule,
            "rate_date_rule": rule_text,
            "cross_rate": "Convert through the base currency using both rates of the same date.",
            "formula": "round_half_even(amount_minor / 10^exp_source * rate_source / rate_target * 10^exp_target); "
                       "rate of the base currency is 1.",
            "rounding": "ROUND_HALF_EVEN",
        },
        "bookkeeping": {
            "invoice": "On issue: debit 1200 and credit 4000 for total_minor in invoice currency, timestamped at issue.",
            "receipt": "Debit 1010 in payment currency. Each invoice_refs entry is applied in full: credit 1200 in the same "
                       "currency, or, when currencies differ, credit 1950 in payment currency, then debit 1950 and credit "
                       "1200 in invoice currency for the converted amount.",
            "unreferenced_receipt": "A receipt with invoice_refs null is credited to 2400 until it is identified.",
            "refund": "A refund of an unapplied receipt debits 2400 and credits 1010.",
            "open_items": "Invoices may stay partly or fully unpaid; that alone is not an error.",
        },
        "chart_of_accounts": dict(spec.ACCOUNTS),
        "findings": spec.FINDING_RULES,
        "amount_tolerance_minor": spec.AMOUNT_TOLERANCE_MINOR,
    }


def _jsonable(value, ext):
    if isinstance(value, tuple) and value and value[0] in REF_TAGS:
        return ext(value)
    if isinstance(value, dict):
        return {k: _jsonable(v, ext) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v, ext) for v in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return value


def render(world, issues, mutations, prng, cfg, episode_id, meta):
    rows = postings(world)
    ties = [prng.random() for _ in rows]
    order = sorted(range(len(rows)), key=lambda i: (rows[i]["ts"], ties[i]))
    if cfg["ledger_order"] == "shuffled":
        prng.shuffle(order)
    ids = IdFactory(prng, world.period_start.year)
    zulu = prng.random() < 0.5

    entity_keys = sorted(world.entities)
    prng.shuffle(entity_keys)
    entity_id = {k: ids.entity() for k in entity_keys}
    invoice_keys = sorted(world.invoices, key=lambda k: (world.invoices[k].issued_at, k))
    invoice_id = {k: ids.invoice() for k in invoice_keys}
    payment_keys = sorted(world.payments, key=lambda k: (world.payments[k].paid_at, k))
    payment_id = {k: ids.payment() for k in payment_keys}
    doc_refs = [("inv_doc", k) for k in invoice_keys] + [("pay_doc", k) for k in payment_keys]
    prng.shuffle(doc_refs)
    doc_id = {ref: ids.doc() for ref in doc_refs}

    ledger = []
    row_id = {}
    for index in order:
        row = rows[index]
        if row["ref"] in row_id:
            raise RuntimeError(f"posting reference {row['ref']} is not unique")
        rid = ids.row()
        row_id[row["ref"]] = rid
        ledger.append({
            "row_id": rid,
            "timestamp": fmt_ts(row["ts"], world.offset, zulu),
            "account": row["account"],
            "debit_minor": row["debit"],
            "credit_minor": row["credit"],
            "currency": row["currency"],
            "entity_id": entity_id[row["entity"]],
            "invoice_id": invoice_id[row["invoice"]] if row["invoice"] is not None else None,
            "payment_id": payment_id[row["payment"]] if row["payment"] is not None else None,
        })

    def ext(ref):
        tag = ref[0]
        if tag == "invoice":
            return invoice_id[ref[1]]
        if tag == "payment":
            return payment_id[ref[1]]
        if tag in ("inv_doc", "pay_doc"):
            return doc_id[ref]
        if tag == "fx":
            return fx_id(ref[1].isoformat(), ref[2], world.base)
        return row_id[ref]

    invoices = []
    shuffled = list(invoice_keys)
    prng.shuffle(shuffled)
    for key in shuffled:
        inv = world.invoices[key]
        invoices.append({
            "doc_id": doc_id[("inv_doc", key)],
            "invoice_id": invoice_id[key],
            "entity_id": entity_id[inv.entity],
            "customer_name": world.entities[inv.entity].name,
            "issued_at": fmt_ts(inv.issued_at, inv.offset, zulu),
            "due_at": fmt_ts(inv.due_at, inv.offset, zulu),
            "currency": inv.currency,
            "total_minor": inv.total,
            "line_items": [dict(line) for line in inv.lines],
        })

    payments = []
    shuffled = list(payment_keys)
    prng.shuffle(shuffled)
    for key in shuffled:
        pay = world.payments[key]
        refs = None
        if pay.refs is not None:
            refs = [{"invoice_id": invoice_id[ik], "amount_minor": amount} for ik, amount in pay.refs]
        payments.append({
            "doc_id": doc_id[("pay_doc", key)],
            "payment_id": payment_id[key],
            "entity_id": entity_id[pay.entity],
            "payer_name": world.entities[pay.entity].name,
            "timestamp": fmt_ts(pay.paid_at, pay.offset, zulu),
            "currency": pay.currency,
            "amount_minor": pay.amount,
            "kind": pay.kind,
            "refund_of": payment_id[pay.refund_of] if pay.refund_of is not None else None,
            "invoice_refs": refs,
        })

    fx = [
        {"date": day.isoformat(), "source_currency": cur, "target_currency": world.base, "rate": text}
        for (day, cur), text in sorted(world.fx.items())
    ]

    manifest = {
        "schema_version": spec.SCHEMA_VERSION,
        "episode_id": episode_id,
        "base_currency": world.base,
        "period_start": fmt_ts(world.period_start, world.offset),
        "period_end": fmt_ts(world.period_end, world.offset),
        "period_end_exclusive": True,
        "timezone": offset_text(world.offset),
        "currency_exponents": {cur: EXPONENTS[cur] for cur in sorted(world.currencies)},
        "record_counts": {"ledger": len(ledger), "invoices": len(invoices), "payments": len(payments), "fx": len(fx)},
        "document_count": len(invoices) + len(payments),
        "rounding": {
            "money": "integer minor units",
            "fx": "ROUND_HALF_EVEN",
            "amount_tolerance_minor": spec.AMOUNT_TOLERANCE_MINOR,
        },
        "generator_version": spec.GENERATOR_VERSION,
        "config_version": cfg["config_version"],
    }
    public = {
        "manifest": manifest,
        "policy": build_policy(world),
        "ledger": ledger,
        "invoices": invoices,
        "payments": payments,
        "fx": fx,
    }

    items = []
    for issue in issues:
        items.append({
            "type": issue.type,
            "affected_ids": sorted(ext(a) for a in issue.affected),
            "expected_amount_minor": issue.amount,
            "currency": issue.currency,
            "evidence_sets": [sorted(ext(r) for r in group) for group in issue.evidence],
            "scenario": issue.scenario,
        })
    items.sort(key=lambda item: (item["type"], item["affected_ids"]))
    truth = {
        "episode_id": episode_id,
        "issues": [{"issue_id": f"ISS-{n:02d}", **item} for n, item in enumerate(items, 1)],
    }
    provenance = {
        **meta,
        "world": {
            "base_currency": world.base,
            "foreign_currencies": list(world.foreign),
            "timezone": offset_text(world.offset),
            "rate_rule": world.rate_rule,
        },
        "mutations": [_jsonable(m, ext) for m in mutations],
    }
    return public, truth, provenance
