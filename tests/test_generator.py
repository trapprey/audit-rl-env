import json
from collections import Counter, defaultdict

import pytest

from auditgym import spec
from auditgym.config import ConfigError, preset_names, resolve
from auditgym.env import AuditEnv
from auditgym.generator import generate_episode
from auditgym.judge import Judge
from auditgym.money import EXPONENTS
from auditgym.policies.scripted import ScriptedAuditor
from auditgym.runner import run_episode
from auditgym.storage import load_public, save_episode
from auditgym.timeutil import parse_ts
from auditgym.verify import verify_truth

CASES = [(preset, seed) for preset in preset_names() for seed in range(8)]


@pytest.mark.parametrize("preset, seed", CASES)
def test_episode_is_consistent(preset, seed):
    episode = generate_episode(seed, {"preset": preset})
    public = episode.public
    assert verify_truth(public, episode.truth) == []
    manifest = public["manifest"]
    start, end = parse_ts(manifest["period_start"]), parse_ts(manifest["period_end"])
    totals = defaultdict(lambda: [0, 0])
    for row in public["ledger"]:
        assert start <= parse_ts(row["timestamp"]) < end
        assert isinstance(row["debit_minor"], int) and isinstance(row["credit_minor"], int)
        assert (row["debit_minor"] > 0) != (row["credit_minor"] > 0)
        totals[row["currency"]][0] += row["debit_minor"]
        totals[row["currency"]][1] += row["credit_minor"]
    assert all(d == c for d, c in totals.values())
    for doc in public["invoices"]:
        assert parse_ts(doc["issued_at"]).tzinfo is not None
        assert doc["total_minor"] == sum(line["amount_minor"] for line in doc["line_items"])
        assert doc["currency"] in EXPONENTS
    for row in public["fx"]:
        assert isinstance(row["rate"], str) and "e" not in row["rate"].lower()
        assert row["target_currency"] == manifest["base_currency"]
    for key, table in (("ledger", "ledger"), ("invoices", "invoices"), ("payments", "payments"), ("fx", "fx")):
        assert manifest["record_counts"][key] == len(public[table])
    ids = (
        [r["row_id"] for r in public["ledger"]]
        + [d["doc_id"] for d in public["invoices"] + public["payments"]]
        + [d["invoice_id"] for d in public["invoices"]]
        + [d["payment_id"] for d in public["payments"]]
    )
    assert len(ids) == len(set(ids))
    assert len(episode.truth["issues"]) <= spec.MAX_ISSUES


def test_clean_preset_has_no_issues_but_has_distractors():
    for seed in range(10):
        episode = generate_episode(seed, {"preset": "clean"})
        assert episode.truth["issues"] == []
        public = episode.public
        kinds = Counter(p["kind"] for p in public["payments"])
        assert kinds["refund"] >= 0
        assert any(p["invoice_refs"] is None for p in public["payments"])
        refs = Counter(r["invoice_id"] for p in public["payments"] if p["invoice_refs"] for r in p["invoice_refs"])
        totals = Counter((d["entity_id"], d["total_minor"]) for d in public["invoices"])
        assert max(totals.values()) >= 2
        assert refs


def test_every_distractor_kind_appears_across_seeds():
    seen = Counter()
    for seed in range(15):
        public = generate_episode(seed, {"preset": "clean"}).public
        seen["refund"] += any(p["kind"] == "refund" for p in public["payments"])
        seen["unreferenced"] += any(p["invoice_refs"] is None and p["kind"] == "receipt" for p in public["payments"])
        refs = Counter(r["invoice_id"] for p in public["payments"] if p["invoice_refs"] for r in p["invoice_refs"])
        seen["split"] += any(n >= 2 for n in refs.values())
        seen["multi_invoice_receipt"] += any(p["invoice_refs"] and len(p["invoice_refs"]) > 1 for p in public["payments"])
        seen["cross_currency"] += any(
            p["invoice_refs"] and p["currency"] != next(i for i in public["invoices"] if i["invoice_id"] == p["invoice_refs"][0]["invoice_id"])["currency"]
            for p in public["payments"]
        )
    assert all(seen[k] > 0 for k in ("refund", "unreferenced", "split", "multi_invoice_receipt", "cross_currency")), seen


@pytest.mark.parametrize("kind", spec.FINDING_TYPES)
def test_error_types_are_independently_configurable(kind):
    errors = {k: 0 for k in spec.FINDING_TYPES}
    errors[kind] = 2
    for seed in range(5):
        truth = generate_episode(seed, {"errors": errors}).truth
        assert Counter(i["type"] for i in truth["issues"]) == Counter({kind: 2})


def test_single_preset_has_one_of_each():
    for seed in range(5):
        truth = generate_episode(seed, {"preset": "single"}).truth
        assert Counter(i["type"] for i in truth["issues"]) == Counter({t: 1 for t in spec.FINDING_TYPES})


def test_interacting_scenarios_share_records():
    for seed in range(5):
        truth = generate_episode(seed, {"preset": "interacting"}).truth
        by_scenario = defaultdict(list)
        for issue in truth["issues"]:
            by_scenario[issue["scenario"]].append(issue)
        boundary = by_scenario["fx_partial_boundary"]
        assert sorted(i["type"] for i in boundary) == ["fx_rate", "payment_allocation"]
        assert boundary[0]["affected_ids"] == boundary[1]["affected_ids"]
        duplicate = by_scenario["duplicate_fx"]
        assert sorted(i["type"] for i in duplicate) == ["duplicate_invoice", "fx_rate"]
        dup = next(i for i in duplicate if i["type"] == "duplicate_invoice")
        fx = next(i for i in duplicate if i["type"] == "fx_rate")
        assert set(dup["affected_ids"]) & set(fx["affected_ids"])


def test_generation_is_reproducible():
    a = generate_episode(77, {"preset": "heavy"})
    b = generate_episode(77, {"preset": "heavy"})
    assert json.dumps(a.public, sort_keys=True) == json.dumps(b.public, sort_keys=True)
    assert a.truth == b.truth and a.provenance == b.provenance
    c = generate_episode(78, {"preset": "heavy"})
    assert c.public["ledger"] != a.public["ledger"]


def test_secret_changes_content():
    plain = generate_episode(5, {"preset": "standard"})
    salted = generate_episode(5, {"preset": "standard"}, secret="jury")
    assert plain.public["ledger"] != salted.public["ledger"]
    assert plain.public["manifest"]["episode_id"] != salted.public["manifest"]["episode_id"]


def test_presentation_seed_keeps_content_and_changes_ids():
    a = generate_episode(31, {"preset": "heavy"})
    b = generate_episode(31, {"preset": "heavy", "presentation_seed": 1234})
    shape = lambda ep: sorted((i["type"], i["expected_amount_minor"], i["currency"]) for i in ep.truth["issues"])
    assert shape(a) == shape(b)
    assert {r["row_id"] for r in a.public["ledger"]} != {r["row_id"] for r in b.public["ledger"]}
    assert sorted(r["debit_minor"] for r in a.public["ledger"]) == sorted(r["debit_minor"] for r in b.public["ledger"])
    assert [d["total_minor"] for d in a.public["invoices"]] != [d["total_minor"] for d in b.public["invoices"]] or len(a.public["invoices"]) < 3


def test_scripted_auditor_raises_no_false_alarms_on_clean_books():
    for seed in range(25):
        env = AuditEnv(judge=Judge("train"), mode="train")
        summary = run_episode(env, ScriptedAuditor(), seed, {"preset": "clean"})
        assert summary["train_metrics"]["fp"] == 0
        assert summary["train_metrics"]["f1"] == 1.0


def test_export_roundtrip(tmp_path):
    episode = generate_episode(3, {"preset": "interacting"})
    save_episode(episode, tmp_path / "ep")
    assert load_public(tmp_path / "ep" / "public") == json.loads(json.dumps(episode.public))
    header = (tmp_path / "ep" / "public" / "ledger.csv").read_text().splitlines()[0]
    assert header.split(",") == list(spec.LEDGER_COLUMNS)
    assert (tmp_path / "ep" / "grader" / "truth.json").exists()


@pytest.mark.parametrize(
    "config",
    [
        {"nonsense": 1},
        {"errors": {"fraud": 1}},
        {"errors": {"fx_rate": [3, 1]}},
        {"errors": {"fx_rate": True}},
        {"period": "year"},
        {"cross_currency_share": 2},
        {"preset": "missing"},
        {"presentation_seed": "abc"},
    ],
)
def test_config_validation(config):
    with pytest.raises(ConfigError):
        resolve(config)


def test_issue_cap():
    cfg = {"errors": {k: 12 for k in spec.FINDING_TYPES}, "interactions": {"fx_partial_boundary": 3, "duplicate_fx": 3}}
    truth = generate_episode(4, cfg).truth
    assert len(truth["issues"]) == spec.MAX_ISSUES
