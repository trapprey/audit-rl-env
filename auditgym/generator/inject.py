from copy import deepcopy
from dataclasses import dataclass
from datetime import timedelta, timezone

from .. import spec
from ..money import convert
from .world import (
    Allocation,
    GenerationError,
    Invoice,
    add_invoice,
    add_receipt,
    display_offset,
    other_currency,
    random_instant,
)


@dataclass
class Issue:
    type: str
    affected: list
    amount: int
    currency: str
    evidence: list
    scenario: str


def _draw(rng, value):
    if isinstance(value, int):
        return value
    return rng.randint(value[0], value[1])


def draw_plan(rng, cfg):
    singles = []
    for kind in spec.FINDING_TYPES:
        singles += [kind] * _draw(rng, cfg["errors"][kind])
    interactions = []
    for name in spec.INTERACTIONS:
        interactions += [name] * _draw(rng, cfg["interactions"][name])
    rng.shuffle(singles)
    while interactions and 2 * len(interactions) > spec.MAX_ISSUES:
        interactions.pop()
    while singles and len(singles) + 2 * len(interactions) > spec.MAX_ISSUES:
        singles.pop()
    return interactions, singles


class Injector:
    def __init__(self, world, rng):
        self.world = world
        self.rng = rng
        self.used_invoices = set()
        self.used_payments = set()
        self.issues = []
        self.mutations = []

    def run(self, interactions, singles):
        for name in interactions:
            getattr(self, name)()
        for kind in singles:
            getattr(self, kind)("single")
        return self.issues, self.mutations

    def _claim(self, invoices=(), payments=()):
        self.used_invoices.update(invoices)
        self.used_payments.update(payments)

    def _posted_invoices(self):
        w = self.world
        out = []
        for key in sorted(w.invoice_posting):
            invoice = w.invoices[key]
            if not w.in_period(invoice.issued_at):
                continue
            if key in self.used_invoices or w.has_tag("invoice", key, "distractor"):
                continue
            out.append(invoice)
        return out

    def _receipts(self):
        w = self.world
        out = []
        for key in sorted(w.payment_booking):
            payment = w.payments[key]
            if payment.kind != "receipt" or payment.refs is None:
                continue
            if key in self.used_payments or w.has_tag("payment", key, "distractor"):
                continue
            if any(ik in self.used_invoices for ik, _ in payment.refs):
                continue
            out.append(payment)
        return out

    def _application_ref(self, payment, alloc):
        invoice = self.world.invoices[alloc.invoice]
        tag = "alloc_ar" if invoice.currency == payment.currency else "alloc_clear_out"
        return (tag, payment.key, alloc.invoice)

    def _fx_refs(self, payment, invoice, day):
        currencies = sorted({payment.currency, invoice.currency} - {self.world.base})
        return [("fx", day, cur) for cur in currencies]

    def _paid_pair(self, cross, earliest, latest, issue_before=None):
        w, rng = self.world, self.rng
        entities = w.sorted_entities()
        for _ in range(30):
            entity = rng.choice(entities)
            paid_at = random_instant(rng, earliest, latest)
            limit = paid_at - timedelta(days=1)
            if issue_before is not None:
                limit = min(limit, issue_before)
            if limit <= w.period_start + timedelta(hours=1):
                continue
            invoice = add_invoice(w, rng, entity, random_instant(rng, w.period_start, limit))
            currency = other_currency(w, rng, invoice.currency) if cross else invoice.currency
            payment = add_receipt(w, rng, invoice, invoice.total, currency, paid_at)
            if payment is not None:
                return invoice, payment
        raise GenerationError("could not create a paid invoice")

    def _wrong_day(self, payment, alloc, boundary):
        w = self.world
        invoice = w.invoices[alloc.invoice]
        correct = alloc.rate_day
        options = [
            payment.paid_at.astimezone(timezone.utc).date(),
            w.local_day(payment.paid_at),
            correct + timedelta(days=1),
            correct - timedelta(days=1),
            w.local_day(invoice.issued_at),
            correct + timedelta(days=2),
            correct - timedelta(days=3),
        ]
        if not boundary:
            options = sorted(set(options))
            self.rng.shuffle(options)
        target = convert(alloc.applied, payment.currency, invoice.currency, w.rates(correct))
        seen = set()
        for day in options:
            if day == correct or day in seen or not w.has_rates(day):
                continue
            seen.add(day)
            value = convert(alloc.applied, payment.currency, invoice.currency, w.rates(day))
            if value > 0 and abs(value - target) >= 2:
                return day
        return None

    def duplicate_invoice(self, scenario, original=None):
        w, rng = self.world, self.rng
        limit = w.period_end - timedelta(days=4)
        if original is None:
            candidates = [i for i in self._posted_invoices() if i.issued_at <= limit]
            if candidates:
                original = rng.choice(candidates)
            else:
                entity = rng.choice(w.sorted_entities())
                original = add_invoice(w, rng, entity, random_instant(rng, w.period_start, limit))
        issued = original.issued_at + timedelta(minutes=rng.randint(90, 72 * 60))
        key = w.next_key()
        duplicate = Invoice(
            key, original.entity, original.currency, issued, issued + (original.due_at - original.issued_at),
            deepcopy(original.lines), display_offset(w, rng, w.entities[original.entity]),
        )
        w.invoices[key] = duplicate
        w.invoice_posting[key] = issued
        self._claim(invoices=[original.key, key])
        self.issues.append(Issue(
            "duplicate_invoice",
            [("invoice", original.key), ("invoice", key)],
            original.total,
            original.currency,
            [[("inv_doc", original.key), ("inv_doc", key)]],
            scenario,
        ))
        self.mutations.append({
            "op": "insert_duplicate_invoice",
            "scenario": scenario,
            "original": ("invoice", original.key),
            "duplicate": ("invoice", key),
            "posted_at": issued,
            "documents": [("inv_doc", key)],
            "ledger_rows": [("inv_ar", key), ("inv_rev", key)],
        })

    def payment_allocation(self, scenario, payment=None, alloc=None, ratio=None):
        w, rng = self.world, self.rng
        if payment is None:
            candidates = self._receipts()
            if candidates:
                payment = rng.choice(candidates)
            else:
                _, payment = self._paid_pair(
                    rng.random() < 0.3, w.period_start + timedelta(days=2), w.period_end - timedelta(hours=1)
                )
        booking = w.payment_booking[payment.key]
        if alloc is None:
            alloc = rng.choice(booking.allocations)
        invoice = w.invoices[alloc.invoice]
        if ratio is None:
            ratio = rng.choice([0.0, rng.uniform(0.35, 0.8)])
        before = alloc.applied
        kept = int(before * ratio)
        rates = w.rates(alloc.rate_day)
        if kept > 0 and convert(kept, payment.currency, invoice.currency, rates) <= 0:
            kept = 0
        shortfall = before - kept
        if kept == 0:
            booking.allocations.remove(alloc)
        else:
            alloc.applied = kept
            alloc.converted = convert(kept, payment.currency, invoice.currency, rates)
        booking.unapplied += shortfall
        self._claim(invoices=[invoice.key], payments=[payment.key])
        evidence = [("pay_doc", payment.key), ("unapplied", payment.key)]
        evidence += [self._application_ref(payment, a) for a in booking.allocations]
        self.issues.append(Issue(
            "payment_allocation",
            [("payment", payment.key), ("invoice", invoice.key)],
            shortfall,
            payment.currency,
            [evidence],
            scenario,
        ))
        self.mutations.append({
            "op": "reduce_allocation",
            "scenario": scenario,
            "payment": ("payment", payment.key),
            "invoice": ("invoice", invoice.key),
            "applied_before": before,
            "applied_after": kept,
            "ledger_rows": [("unapplied", payment.key)] + (
                [self._application_ref(payment, alloc)] if kept else []
            ),
        })

    def fx_rate(self, scenario, payment=None, alloc=None, boundary=False):
        w, rng = self.world, self.rng
        if payment is None:
            found = None
            candidates = [
                (p, a)
                for p in self._receipts()
                for a in w.payment_booking[p.key].allocations
                if w.invoices[a.invoice].currency != p.currency
            ]
            rng.shuffle(candidates)
            for p, a in candidates:
                day = self._wrong_day(p, a, False)
                if day is not None:
                    found = (p, a, day)
                    break
            attempts = 0
            while found is None:
                attempts += 1
                if attempts > 15:
                    raise GenerationError("no cross-currency receipt for an fx error")
                _, p = self._paid_pair(True, w.period_start + timedelta(days=2), w.period_end - timedelta(hours=1))
                a = w.payment_booking[p.key].allocations[0]
                day = self._wrong_day(p, a, False)
                if day is not None:
                    found = (p, a, day)
            payment, alloc, wrong = found
        else:
            wrong = self._wrong_day(payment, alloc, boundary)
            if wrong is None:
                raise GenerationError("no usable wrong rate date")
        invoice = w.invoices[alloc.invoice]
        correct_day = alloc.rate_day
        correct = convert(alloc.applied, payment.currency, invoice.currency, w.rates(correct_day))
        booked = convert(alloc.applied, payment.currency, invoice.currency, w.rates(wrong))
        alloc.converted = booked
        alloc.rate_day = wrong
        self._claim(invoices=[invoice.key], payments=[payment.key])
        fx = self._fx_refs(payment, invoice, correct_day)
        out_ref = ("alloc_clear_out", payment.key, invoice.key)
        self.issues.append(Issue(
            "fx_rate",
            [("payment", payment.key), ("invoice", invoice.key)],
            abs(booked - correct),
            invoice.currency,
            [
                [("pay_doc", payment.key), out_ref, ("alloc_ar", payment.key, invoice.key), *fx],
                [("pay_doc", payment.key), out_ref, ("alloc_clear_in", payment.key, invoice.key), *fx],
            ],
            scenario,
        ))
        self.mutations.append({
            "op": "wrong_rate_date",
            "scenario": scenario,
            "payment": ("payment", payment.key),
            "invoice": ("invoice", invoice.key),
            "policy_rate_date": correct_day,
            "used_rate_date": wrong,
            "converted_before": correct,
            "converted_after": booked,
            "ledger_rows": [("alloc_clear_in", payment.key, invoice.key), ("alloc_ar", payment.key, invoice.key)],
        })

    def period_cutoff(self, scenario):
        w, rng = self.world, self.rng
        candidates = [
            w.invoices[k]
            for k in sorted(w.invoices)
            if k not in w.invoice_posting and k not in self.used_invoices and w.invoices[k].issued_at >= w.period_end
        ]
        if candidates:
            invoice = rng.choice(candidates)
        else:
            entity = rng.choice(w.sorted_entities())
            issued = random_instant(rng, w.period_end, w.period_end + timedelta(days=3))
            invoice = add_invoice(w, rng, entity, issued, post=False)
        posted_at = w.period_end - timedelta(minutes=rng.randint(20, 6 * 24 * 60))
        w.invoice_posting[invoice.key] = posted_at
        self._claim(invoices=[invoice.key])
        self.issues.append(Issue(
            "period_cutoff",
            [("invoice", invoice.key)],
            invoice.total,
            invoice.currency,
            [[("inv_doc", invoice.key), ("inv_ar", invoice.key)], [("inv_doc", invoice.key), ("inv_rev", invoice.key)]],
            scenario,
        ))
        self.mutations.append({
            "op": "early_posting",
            "scenario": scenario,
            "invoice": ("invoice", invoice.key),
            "issued_at": invoice.issued_at,
            "posted_at": posted_at,
            "ledger_rows": [("inv_ar", invoice.key), ("inv_rev", invoice.key)],
        })

    def fx_partial_boundary(self):
        w, rng = self.world, self.rng
        for _ in range(20):
            _, payment = self._paid_pair(True, w.period_end - timedelta(hours=30), w.period_end - timedelta(minutes=10))
            alloc = w.payment_booking[payment.key].allocations[0]
            ratio = rng.uniform(0.4, 0.75)
            kept = int(alloc.applied * ratio)
            if kept <= 0:
                continue
            probe = Allocation(alloc.invoice, kept, 0, alloc.rate_day)
            if self._wrong_day(payment, probe, True) is None:
                continue
            self.payment_allocation("fx_partial_boundary", payment, alloc, ratio)
            self.fx_rate("fx_partial_boundary", payment, alloc, boundary=True)
            return
        raise GenerationError("could not build fx_partial_boundary")

    def duplicate_fx(self):
        w = self.world
        for _ in range(20):
            invoice, payment = self._paid_pair(
                True,
                w.period_start + timedelta(days=3),
                w.period_end - timedelta(hours=2),
                issue_before=w.period_end - timedelta(days=4),
            )
            alloc = w.payment_booking[payment.key].allocations[0]
            if self._wrong_day(payment, alloc, False) is None:
                continue
            self.duplicate_invoice("duplicate_fx", original=invoice)
            self.fx_rate("duplicate_fx", payment, alloc)
            return
        raise GenerationError("could not build duplicate_fx")
