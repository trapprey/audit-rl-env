import json
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

from .. import spec
from ..money import EXPONENTS, USD_VALUE, convert, rate_text

BASE_CHOICES = ["USD", "EUR", "KZT", "GBP"]
FOREIGN_CHOICES = ["USD", "EUR", "GBP", "KZT", "CHF", "AED", "JPY", "KWD", "TRY"]
OFFSET_CHOICES = [-300, -180, 0, 60, 180, 300, 330, 360, 540]
PAYMENT_TERMS = [7, 14, 30, 30, 45, 60]

NAME_HEADS = [
    "Alt", "Bor", "Cel", "Dar", "Eri", "Fal", "Gor", "Hel", "Ister", "Jun", "Kel", "Lum",
    "Mar", "Nor", "Orb", "Pel", "Quor", "Ros", "Sar", "Tal", "Ulm", "Ver", "Wes", "Zan",
]
NAME_TAILS = ["ana", "ex", "ion", "ora", "um", "ic", "eta", "on", "ar", "is", "ova", "ent"]
NAME_KINDS = [
    "LLC", "Ltd", "Group", "Trading", "Systems", "Partners", "Labs", "Logistics",
    "Foods", "Studio", "Holdings", "Works",
]

CATALOG = [
    ("CONS-H", "Consulting hours", 120, (2, 40)),
    ("SUP-M", "Support plan", 900, (1, 1)),
    ("LIC-U", "Software licence seats", 45, (5, 60)),
    ("TRN-D", "Training day", 1500, (1, 3)),
    ("DEV-S", "Development sprint", 6400, (1, 2)),
    ("HOST-M", "Cloud hosting", 380, (1, 4)),
    ("AUD-R", "Data review report", 2200, (1, 2)),
    ("SHIP", "Freight handling", 260, (1, 12)),
]


class GenerationError(RuntimeError):
    pass


@dataclass
class Entity:
    key: int
    name: str
    currency: str
    offset: int
    price_factor: float


@dataclass
class Invoice:
    key: int
    entity: int
    currency: str
    issued_at: datetime
    due_at: datetime
    lines: list
    offset: int

    @property
    def total(self):
        return sum(line["amount_minor"] for line in self.lines)


@dataclass
class Payment:
    key: int
    entity: int
    currency: str
    paid_at: datetime
    amount: int
    refs: list | None
    offset: int
    kind: str = "receipt"
    refund_of: int | None = None


@dataclass
class Allocation:
    invoice: int
    applied: int
    converted: int
    rate_day: date


@dataclass
class Booking:
    allocations: list
    unapplied: int


@dataclass
class World:
    base: str
    foreign: list
    offset: int
    period_start: datetime
    period_end: datetime
    close_at: datetime
    rate_rule: str
    fx: dict = field(default_factory=dict)
    entities: dict = field(default_factory=dict)
    invoices: dict = field(default_factory=dict)
    payments: dict = field(default_factory=dict)
    invoice_posting: dict = field(default_factory=dict)
    payment_booking: dict = field(default_factory=dict)
    tags: dict = field(default_factory=dict)
    signatures: set = field(default_factory=set)
    counter: int = 0

    @property
    def tz(self):
        return timezone(timedelta(minutes=self.offset))

    @property
    def currencies(self):
        return [self.base, *self.foreign]

    def next_key(self):
        self.counter += 1
        return self.counter

    def in_period(self, moment):
        return self.period_start <= moment < self.period_end

    def local_day(self, moment):
        return moment.astimezone(self.tz).date()

    def rate_day(self, moment):
        day = self.local_day(moment)
        if self.rate_rule == "previous_local_date":
            day -= timedelta(days=1)
        return day

    def has_rates(self, day):
        return all((day, cur) in self.fx for cur in self.foreign)

    def rates(self, day):
        table = {self.base: Decimal(1)}
        for cur in self.foreign:
            table[cur] = Decimal(self.fx[(day, cur)])
        return table

    def tag(self, kind, key, label):
        self.tags.setdefault((kind, key), set()).add(label)

    def has_tag(self, kind, key, label):
        return label in self.tags.get((kind, key), ())

    def sorted_entities(self):
        return [self.entities[k] for k in sorted(self.entities)]


def add_months(moment, months):
    index = moment.month - 1 + months
    return moment.replace(year=moment.year + index // 12, month=index % 12 + 1)


def random_instant(rng, start, end):
    span = int((end - start).total_seconds() // 60)
    if span <= 1:
        return start
    return start + timedelta(minutes=rng.randint(0, span - 1))


def unit_price(usd, currency):
    major = usd / float(USD_VALUE[currency])
    step = 1 if major >= 50 else 0.5
    major = max(step, round(major / step) * step)
    return max(1, int(round(major * 10 ** EXPONENTS[currency])))


def make_line(sku, description, quantity, unit, begin, finish):
    return {
        "sku": sku,
        "description": description,
        "quantity": quantity,
        "unit_price_minor": unit,
        "amount_minor": quantity * unit,
        "service_period": f"{begin.isoformat()}/{finish.isoformat()}",
    }


def make_lines(world, rng, entity, issued_at):
    day = world.local_day(issued_at)
    lines = []
    for _ in range(rng.choice([1, 1, 1, 2, 2, 3])):
        sku, description, usd, (low, high) = rng.choice(CATALOG)
        begin = day - timedelta(days=rng.randint(0, 25))
        finish = begin + timedelta(days=rng.randint(0, 13))
        unit = unit_price(usd * entity.price_factor, entity.currency)
        lines.append(make_line(sku, description, rng.randint(low, high), unit, begin, finish))
    return lines


def signature(entity_key, currency, lines):
    return (entity_key, currency, json.dumps(lines, sort_keys=True))


def display_offset(world, rng, entity):
    return rng.choice([entity.offset, world.offset, 0])


def add_invoice(world, rng, entity, issued_at, lines=None, post=True):
    fixed = lines is not None
    for _ in range(30):
        candidate = lines if fixed else make_lines(world, rng, entity, issued_at)
        sig = signature(entity.key, entity.currency, candidate)
        if sig not in world.signatures:
            break
        if fixed:
            return None
    else:
        raise GenerationError("could not produce unique invoice lines")
    world.signatures.add(sig)
    key = world.next_key()
    due = issued_at + timedelta(days=rng.choice(PAYMENT_TERMS))
    invoice = Invoice(key, entity.key, entity.currency, issued_at, due, candidate, display_offset(world, rng, entity))
    world.invoices[key] = invoice
    if post:
        world.invoice_posting[key] = issued_at
    return invoice


def other_currency(world, rng, currency):
    if currency != world.base and rng.random() < 0.6:
        return world.base
    return rng.choice([c for c in world.currencies if c != currency])


def max_payable(target, source, invoice_currency, rates):
    amount = convert(target, invoice_currency, source, rates)
    guard = 0
    while amount > 0 and convert(amount, source, invoice_currency, rates) > target:
        amount -= 1
        guard += 1
        if guard > 100_000:
            raise GenerationError("conversion search diverged")
    while convert(amount + 1, source, invoice_currency, rates) <= target:
        amount += 1
        guard += 1
        if guard > 100_000:
            raise GenerationError("conversion search diverged")
    return amount


def clean_booking(world, payment):
    if payment.kind == "refund":
        return Booking([], 0)
    if payment.refs is None:
        return Booking([], payment.amount)
    day = world.rate_day(payment.paid_at)
    rates = world.rates(day)
    allocations = []
    for invoice_key, amount in payment.refs:
        invoice = world.invoices[invoice_key]
        converted = convert(amount, payment.currency, invoice.currency, rates)
        allocations.append(Allocation(invoice_key, amount, converted, day))
    return Booking(allocations, payment.amount - sum(amount for _, amount in payment.refs))


def add_receipt(world, rng, invoice, part, currency, paid_at, post=True):
    day = world.rate_day(paid_at)
    rates = world.rates(day)
    if currency == invoice.currency:
        amount = part
    else:
        amount = max_payable(part, currency, invoice.currency, rates)
        if amount <= 0 or convert(amount, currency, invoice.currency, rates) <= 0:
            return None
    if amount <= 0:
        return None
    entity = world.entities[invoice.entity]
    key = world.next_key()
    payment = Payment(key, invoice.entity, currency, paid_at, amount, [[invoice.key, amount]], display_offset(world, rng, entity))
    world.payments[key] = payment
    if post:
        world.payment_booking[key] = clean_booking(world, payment)
    return payment


def settle(world, rng, invoice, outcome, cross_share):
    currency = invoice.currency
    if rng.random() < cross_share:
        currency = other_currency(world, rng, invoice.currency)
    total = invoice.total
    if outcome == "full":
        parts = [total]
    elif outcome == "split":
        count = rng.randint(2, 3)
        if total < 100 * count:
            parts = [total]
        else:
            weights = [rng.uniform(0.25, 1.0) for _ in range(count)]
            scale = sum(weights)
            parts = [int(total * w / scale) for w in weights[:-1]]
            parts.append(total - sum(parts))
    elif outcome == "partial":
        parts = [max(1, int(total * rng.uniform(0.3, 0.7)))]
    else:
        parts = []
    moment = invoice.issued_at
    made = []
    for part in parts:
        moment = moment + timedelta(minutes=rng.randint(6 * 60, 18 * 24 * 60))
        if moment >= world.period_end:
            if moment < world.close_at and rng.random() < 0.5:
                late = add_receipt(world, rng, invoice, part, currency, moment, post=False)
                if late:
                    world.tag("payment", late.key, "late")
            break
        payment = add_receipt(world, rng, invoice, part, currency, moment)
        if payment:
            made.append(payment)
    return made


def build_world(rng, cfg):
    base = rng.choice(BASE_CHOICES)
    pool = [c for c in FOREIGN_CHOICES if c != base]
    foreign = sorted(rng.sample(pool, rng.randint(*cfg["foreign_currencies"])))
    offset = rng.choice(OFFSET_CHOICES)
    tz = timezone(timedelta(minutes=offset))
    year = rng.choice([2025, 2026])
    if cfg["period"] == "quarter":
        month, span = rng.choice([1, 4, 7, 10]), 3
    else:
        month, span = rng.randint(1, 12), 1
    start = datetime(year, month, 1, tzinfo=tz)
    end = add_months(start, span)
    rule = rng.choice(["transaction_local_date", "previous_local_date"])
    world = World(base, foreign, offset, start, end, end + timedelta(days=6), rule)
    _make_fx(world, rng)
    _make_entities(world, rng, cfg)
    _make_invoices(world, rng, cfg)
    _make_receipts(world, rng, cfg)
    _make_distractors(world, rng, cfg)
    return world


def _make_fx(world, rng):
    first = world.period_start.date() - timedelta(days=20)
    last = world.close_at.date() + timedelta(days=3)
    for cur in world.foreign:
        value = float(Decimal(USD_VALUE[cur]) / Decimal(USD_VALUE[world.base])) * rng.uniform(0.94, 1.06)
        day = first
        while day <= last:
            value *= math.exp(rng.gauss(0, 0.0045))
            world.fx[(day, cur)] = rate_text(value)
            day += timedelta(days=1)


def _make_entities(world, rng, cfg):
    names = set()
    target = rng.randint(*cfg["customers"])
    while len(world.entities) < target:
        name = f"{rng.choice(NAME_HEADS)}{rng.choice(NAME_TAILS)} {rng.choice(NAME_KINDS)}"
        if name in names:
            continue
        names.add(name)
        currency = world.base if rng.random() < 0.55 else rng.choice(world.foreign)
        key = world.next_key()
        world.entities[key] = Entity(key, name, currency, rng.choice(OFFSET_CHOICES), rng.uniform(0.85, 1.2))
    if all(e.currency == world.base for e in world.entities.values()):
        world.sorted_entities()[0].currency = world.foreign[0]


def _make_invoices(world, rng, cfg):
    entities = world.sorted_entities()
    for _ in range(rng.randint(*cfg["invoices"])):
        issued = random_instant(rng, world.period_start, world.period_end - timedelta(hours=2))
        add_invoice(world, rng, rng.choice(entities), issued)
    for _ in range(rng.randint(*cfg["late_invoices"])):
        issued = random_instant(rng, world.period_end, world.period_end + timedelta(days=5))
        invoice = add_invoice(world, rng, rng.choice(entities), issued, post=False)
        world.tag("invoice", invoice.key, "late")


def _make_receipts(world, rng, cfg):
    weights = [50, 15 if cfg["distractors"]["split_payments"] else 0, 10, 25]
    posted = sorted((world.invoices[k] for k in world.invoice_posting), key=lambda i: (i.issued_at, i.key))
    for invoice in posted:
        outcome = rng.choices(["full", "split", "partial", "open"], weights=weights)[0]
        settle(world, rng, invoice, outcome, cfg["cross_currency_share"])
    _merge_receipts(world, rng)


def _merge_receipts(world, rng):
    groups = {}
    for key in sorted(world.payment_booking):
        payment = world.payments[key]
        if payment.kind != "receipt" or payment.refs is None or len(payment.refs) != 1:
            continue
        if world.invoices[payment.refs[0][0]].currency != payment.currency:
            continue
        groups.setdefault((payment.entity, payment.currency), []).append(payment)
    for group in sorted(groups):
        payments = sorted(groups[group], key=lambda p: (p.paid_at, p.key))
        index = 0
        while index + 1 < len(payments):
            first, second = payments[index], payments[index + 1]
            close = second.paid_at - first.paid_at <= timedelta(days=10)
            if first.refs[0][0] != second.refs[0][0] and close and rng.random() < 0.35:
                key = world.next_key()
                merged = Payment(
                    key, first.entity, first.currency, second.paid_at, first.amount + second.amount,
                    [list(first.refs[0]), list(second.refs[0])], second.offset,
                )
                for old in (first, second):
                    del world.payments[old.key]
                    del world.payment_booking[old.key]
                world.payments[key] = merged
                world.payment_booking[key] = clean_booking(world, merged)
                index += 2
            else:
                index += 1


def _make_distractors(world, rng, cfg):
    flags = cfg["distractors"]
    if flags["recurring_amounts"]:
        _recurring(world, rng)
    if flags["unreferenced_receipts"]:
        _unreferenced(world, rng)
    if flags["refunds"]:
        _refund(world, rng)
    if flags["timezone_edges"]:
        _edges(world, rng)


def _recurring(world, rng):
    entities = world.sorted_entities()
    for entity in rng.sample(entities, min(2, len(entities))):
        sku, description, usd, (low, high) = rng.choice(CATALOG)
        quantity = rng.randint(low, high)
        unit = unit_price(usd * entity.price_factor, entity.currency)
        count = rng.randint(2, 3)
        window = (world.period_end - world.period_start - timedelta(hours=3)) / count
        for i in range(count):
            begin_at = world.period_start + window * i
            issued = random_instant(rng, begin_at, begin_at + window * 0.6)
            day = world.local_day(issued)
            lines = [make_line(sku, description, quantity, unit, day - timedelta(days=6), day)]
            invoice = add_invoice(world, rng, entity, issued, lines)
            if invoice is None:
                continue
            world.tag("invoice", invoice.key, "distractor")
            for payment in settle(world, rng, invoice, rng.choice(["full", "full", "open"]), 0.0):
                world.tag("payment", payment.key, "distractor")


def _unreferenced(world, rng):
    entities = world.sorted_entities()
    for _ in range(rng.randint(1, 2)):
        entity = rng.choice(entities)
        amount = unit_price(rng.uniform(80, 2500), entity.currency)
        paid_at = random_instant(rng, world.period_start, world.period_end - timedelta(hours=1))
        key = world.next_key()
        payment = Payment(key, entity.key, entity.currency, paid_at, amount, None, display_offset(world, rng, entity))
        world.payments[key] = payment
        world.payment_booking[key] = clean_booking(world, payment)
        world.tag("payment", key, "distractor")


def _refund(world, rng):
    limit = world.period_end - timedelta(days=4)
    candidates = []
    for key in sorted(world.payment_booking):
        payment = world.payments[key]
        if payment.kind != "receipt" or payment.refs is None or len(payment.refs) != 1:
            continue
        if world.invoices[payment.refs[0][0]].currency != payment.currency or payment.paid_at >= limit:
            continue
        if world.has_tag("payment", key, "distractor"):
            continue
        candidates.append(payment)
    if not candidates:
        return
    original = rng.choice(candidates)
    again_at = original.paid_at + timedelta(minutes=rng.randint(30, 48 * 60))
    refund_at = again_at + timedelta(minutes=rng.randint(60, 48 * 60))
    if refund_at >= world.period_end:
        return
    again = Payment(world.next_key(), original.entity, original.currency, again_at, original.amount, None, original.offset)
    world.payments[again.key] = again
    world.payment_booking[again.key] = clean_booking(world, again)
    refund = Payment(
        world.next_key(), original.entity, original.currency, refund_at, -original.amount, None,
        world.offset, kind="refund", refund_of=again.key,
    )
    world.payments[refund.key] = refund
    world.payment_booking[refund.key] = clean_booking(world, refund)
    for key in (original.key, again.key, refund.key):
        world.tag("payment", key, "distractor")


def _edges(world, rng):
    entities = world.sorted_entities()
    entity = rng.choice(entities)
    inside = add_invoice(world, rng, entity, world.period_end - timedelta(minutes=rng.randint(5, 170)))
    inside.offset = rng.choice([0, entity.offset])
    world.tag("invoice", inside.key, "distractor")
    entity = rng.choice(entities)
    outside = add_invoice(world, rng, entity, world.period_end + timedelta(minutes=rng.randint(5, 170)), post=False)
    outside.offset = rng.choice([0, entity.offset])
    world.tag("invoice", outside.key, "late")


def postings(world):
    rows = []

    def add(moment, account, debit, credit, currency, entity, invoice, payment, ref):
        rows.append({
            "ts": moment, "account": account, "debit": debit, "credit": credit, "currency": currency,
            "entity": entity, "invoice": invoice, "payment": payment, "ref": ref,
        })

    for key in sorted(world.invoice_posting):
        moment = world.invoice_posting[key]
        invoice = world.invoices[key]
        add(moment, spec.AR, invoice.total, 0, invoice.currency, invoice.entity, key, None, ("inv_ar", key))
        add(moment, spec.REVENUE, 0, invoice.total, invoice.currency, invoice.entity, key, None, ("inv_rev", key))
    for key in sorted(world.payment_booking):
        payment = world.payments[key]
        booking = world.payment_booking[key]
        moment, cur, ent = payment.paid_at, payment.currency, payment.entity
        if payment.kind == "refund":
            add(moment, spec.UNAPPLIED, -payment.amount, 0, cur, ent, None, key, ("refund_unapplied", key))
            add(moment, spec.BANK, 0, -payment.amount, cur, ent, None, key, ("refund_bank", key))
            continue
        add(moment, spec.BANK, payment.amount, 0, cur, ent, None, key, ("pay_bank", key))
        for alloc in booking.allocations:
            invoice = world.invoices[alloc.invoice]
            if invoice.currency == cur:
                add(moment, spec.AR, 0, alloc.applied, cur, ent, alloc.invoice, key, ("alloc_ar", key, alloc.invoice))
            else:
                add(moment, spec.FX_CLEARING, 0, alloc.applied, cur, ent, alloc.invoice, key, ("alloc_clear_out", key, alloc.invoice))
                add(moment, spec.FX_CLEARING, alloc.converted, 0, invoice.currency, ent, alloc.invoice, key, ("alloc_clear_in", key, alloc.invoice))
                add(moment, spec.AR, 0, alloc.converted, invoice.currency, ent, alloc.invoice, key, ("alloc_ar", key, alloc.invoice))
        if booking.unapplied:
            add(moment, spec.UNAPPLIED, 0, booking.unapplied, cur, ent, None, key, ("unapplied", key))
    return rows


def check_balance(world, rows=None):
    rows = postings(world) if rows is None else rows
    problems = []
    totals = {}
    for row in rows:
        debit, credit = row["debit"], row["credit"]
        if debit < 0 or credit < 0 or (debit == 0) == (credit == 0):
            problems.append(f"row {row['ref']} must have exactly one positive side")
        if not world.in_period(row["ts"]):
            problems.append(f"row {row['ref']} is outside the period")
        side = totals.setdefault(row["currency"], [0, 0])
        side[0] += debit
        side[1] += credit
    for cur in sorted(totals):
        if totals[cur][0] != totals[cur][1]:
            problems.append(f"{cur} debits {totals[cur][0]} != credits {totals[cur][1]}")
    return problems


def check_books(world):
    rows = postings(world)
    problems = check_balance(world, rows)
    receivable = {}
    unapplied = {}
    for row in rows:
        if row["account"] == spec.AR and row["invoice"] is not None:
            side = receivable.setdefault(row["invoice"], [0, 0])
            side[0] += row["debit"]
            side[1] += row["credit"]
        if row["account"] == spec.UNAPPLIED:
            unapplied[row["currency"]] = unapplied.get(row["currency"], 0) + row["credit"] - row["debit"]
    seen = set()
    for key in sorted(world.invoices):
        invoice = world.invoices[key]
        posted = key in world.invoice_posting
        if posted != world.in_period(invoice.issued_at):
            problems.append(f"invoice {key} posting does not match its period")
        if posted:
            debit, credit = receivable.get(key, (0, 0))
            if debit != invoice.total:
                problems.append(f"invoice {key} receivable {debit} != total {invoice.total}")
            if credit > invoice.total:
                problems.append(f"invoice {key} is over-credited")
        sig = signature(invoice.entity, invoice.currency, invoice.lines)
        if sig in seen:
            problems.append(f"invoice {key} duplicates another invoice")
        seen.add(sig)
    for key in sorted(world.payments):
        payment = world.payments[key]
        booking = world.payment_booking.get(key)
        if booking is None:
            if world.in_period(payment.paid_at):
                problems.append(f"payment {key} in period is not booked")
            continue
        if payment.kind == "refund":
            continue
        applied = sum(a.applied for a in booking.allocations)
        if applied + booking.unapplied != payment.amount:
            problems.append(f"payment {key} booked amount mismatch")
        if payment.refs is None:
            if booking.allocations:
                problems.append(f"payment {key} without references was applied")
            continue
        if booking.unapplied:
            problems.append(f"payment {key} with references left unapplied cash")
        expected = {ik: amount for ik, amount in payment.refs}
        actual = {a.invoice: a.applied for a in booking.allocations}
        if expected != actual:
            problems.append(f"payment {key} allocations differ from references")
        rates = world.rates(world.rate_day(payment.paid_at))
        for alloc in booking.allocations:
            invoice = world.invoices[alloc.invoice]
            if alloc.invoice not in world.invoice_posting:
                problems.append(f"payment {key} applied to an unposted invoice")
            if alloc.converted != convert(alloc.applied, payment.currency, invoice.currency, rates):
                problems.append(f"payment {key} conversion differs from policy")
            if alloc.converted <= 0:
                problems.append(f"payment {key} converts to zero")
    for cur, balance in sorted(unapplied.items()):
        if balance < 0:
            problems.append(f"unapplied receipts in {cur} are negative")
    return problems
