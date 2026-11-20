import random
from collections import defaultdict
from datetime import timezone

from .. import spec
from ..timeutil import parse_ts
from .common import TableReader, act, finding


class RandomExplorer:
    name = "random_explorer"

    def __init__(self, seed=0):
        self.seed = seed

    def reset(self, observation, info, env=None):
        self.rng = random.Random(f"{self.seed}|{info['episode_id']}")
        self.episode_id = observation["manifest"]["episode_id"]
        self.budget = self.rng.randint(3, 30)
        self.seen = defaultdict(set)
        self.cursors = {}
        self.pending = None

    def _remember(self, observation):
        result = observation.get("result")
        table, self.pending = self.pending, None
        if not isinstance(result, dict):
            return
        records = result.get("records") if "records" in result else [result]
        for record in records or []:
            if not isinstance(record, dict):
                continue
            for key in ("row_id", "doc_id", "invoice_id", "payment_id", "fx_id"):
                if record.get(key):
                    self.seen[key].add(record[key])
        if table and "next_cursor" in result:
            self.cursors[table] = result["next_cursor"]

    def _pick(self, key):
        values = sorted(self.seen.get(key, ()))
        return self.rng.choice(values) if values else f"X{self.rng.randint(0, 99999)}"

    def act(self, observation):
        self._remember(observation)
        if self.budget <= 0 or observation["remaining_steps"] <= 1:
            return act("submit", findings={"episode_id": self.episode_id, "findings": self._guess()})
        self.budget -= 1
        choice = self.rng.random()
        if choice < 0.45:
            table = self.rng.choice(spec.TABLES)
            self.pending = table
            return act("list_records", table=table, cursor=self.cursors.get(table), limit=self.rng.randint(1, 50))
        if choice < 0.65:
            table = self.rng.choice(["ledger", "invoices", "payments"])
            key = {"ledger": "row_id", "invoices": "invoice_id", "payments": "payment_id"}[table]
            return act("get_record", table=table, id=self._pick(key))
        if choice < 0.85:
            return act("get_document", doc_id=self._pick("doc_id"))
        if choice < 0.95:
            return act("calculator", expression=f"{self.rng.randint(1, 9999)} * {self.rng.random():.4f}")
        return {"name": self.rng.choice(["shell", "sql", "read_file"]), "args": {}}

    def _guess(self):
        out = []
        for _ in range(self.rng.randint(0, 3)):
            kind = self.rng.choice(spec.FINDING_TYPES)
            ids = [self._pick("invoice_id")]
            if kind in ("payment_allocation", "fx_rate"):
                ids.append(self._pick("payment_id"))
            if kind == "duplicate_invoice":
                ids.append(self._pick("invoice_id"))
            out.append(finding(
                kind, ids, self.rng.randint(1, 10**6), self.rng.choice(["USD", "EUR", "KZT"]),
                [self._pick("doc_id"), self._pick("row_id")], "random guess",
            ))
        return out


class NaiveReconciler:
    name = "naive_reconciler"

    def reset(self, observation, info, env=None):
        self.manifest = observation["manifest"]
        self.reader = TableReader(("invoices", "payments", "ledger"))

    def act(self, observation):
        self.reader.absorb(observation)
        if not self.reader.done and observation["remaining_steps"] > 1:
            return self.reader.next_action()
        return act("submit", findings={"episode_id": self.manifest["episode_id"], "findings": self._rules()[: spec.MAX_FINDINGS]})

    def _rules(self):
        data = self.reader.data
        start = parse_ts(self.manifest["period_start"]).astimezone(timezone.utc).date()
        end = parse_ts(self.manifest["period_end"]).astimezone(timezone.utc).date()
        rows_by_invoice = defaultdict(list)
        for row in data["ledger"]:
            if row["invoice_id"]:
                rows_by_invoice[row["invoice_id"]].append(row)
        out = []
        groups = defaultdict(list)
        for doc in data["invoices"]:
            groups[(doc["entity_id"], doc["total_minor"])].append(doc)
        for docs in groups.values():
            if len(docs) > 1:
                docs.sort(key=lambda d: d["issued_at"])
                for other in docs[1:]:
                    out.append(finding(
                        "duplicate_invoice", [docs[0]["invoice_id"], other["invoice_id"]], other["total_minor"],
                        other["currency"], [docs[0]["doc_id"], other["doc_id"]], "same customer and amount",
                    ))
        by_payment = {p["payment_id"]: p for p in data["payments"]}
        for row in data["ledger"]:
            if row["account"] != spec.UNAPPLIED or row["credit_minor"] <= 0:
                continue
            payment = by_payment.get(row["payment_id"])
            if payment is None:
                continue
            refs = payment["invoice_refs"] or []
            invoice_id = refs[0]["invoice_id"] if refs else row["entity_id"]
            out.append(finding(
                "payment_allocation", [payment["payment_id"], invoice_id], row["credit_minor"], row["currency"],
                [payment["doc_id"], row["row_id"]], "cash left in unapplied receipts",
            ))
        for doc in data["invoices"]:
            day = parse_ts(doc["issued_at"]).astimezone(timezone.utc).date()
            booked = [r for r in rows_by_invoice.get(doc["invoice_id"], []) if r["account"] == spec.AR and r["debit_minor"] > 0]
            if booked and not start <= day < end:
                out.append(finding(
                    "period_cutoff", [doc["invoice_id"]], doc["total_minor"], doc["currency"],
                    [doc["doc_id"], booked[0]["row_id"]], "issue date outside period (UTC)",
                ))
        return out


class FlagEverything:
    name = "flag_all"

    def reset(self, observation, info, env=None):
        self.manifest = observation["manifest"]
        self.queue = ["invoices", "payments", "ledger"]
        self.data = defaultdict(list)
        self.pending = None

    def act(self, observation):
        result = observation.get("result")
        if self.pending and isinstance(result, dict) and result.get("records"):
            self.data[self.pending].extend(result["records"])
        if self.queue:
            self.pending = self.queue.pop(0)
            return act("list_records", table=self.pending, limit=50)
        return act("submit", findings={"episode_id": self.manifest["episode_id"], "findings": self._everything()})

    def _everything(self):
        invoices = self.data["invoices"]
        payments = [p for p in self.data["payments"] if p.get("invoice_refs")]
        rows = [r["row_id"] for r in self.data["ledger"]]
        out = []
        index = 0
        while len(out) < spec.MAX_FINDINGS and (invoices or payments):
            kind = spec.FINDING_TYPES[index % 4]
            if kind in ("payment_allocation", "fx_rate") and payments:
                payment = payments[(index // 4) % len(payments)]
                ref = payment["invoice_refs"][0]
                out.append(finding(
                    kind, [payment["payment_id"], ref["invoice_id"]], ref["amount_minor"], payment["currency"],
                    [payment["doc_id"], *rows[:9]], "flag",
                ))
            elif invoices:
                first = invoices[(index // 4) % len(invoices)]
                second = invoices[(index // 4 + 1) % len(invoices)]
                ids = [first["invoice_id"], second["invoice_id"]] if kind == "duplicate_invoice" else [first["invoice_id"]]
                out.append(finding(
                    kind, ids, first["total_minor"], first["currency"],
                    [first["doc_id"], second["doc_id"], *rows[:8]], "flag",
                ))
            index += 1
        return out


class Oracle:
    name = "oracle"

    def reset(self, observation, info, env=None):
        if env is None:
            raise RuntimeError("oracle needs the environment handle")
        self.findings = env.oracle_findings()

    def act(self, observation):
        return act("submit", findings=self.findings)


class Silent:
    name = "silent"

    def reset(self, observation, info, env=None):
        self.episode_id = observation["manifest"]["episode_id"]

    def act(self, observation):
        return act("submit", findings={"episode_id": self.episode_id, "findings": []})
