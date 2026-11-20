import json

import pytest

from auditgym.config import preset_names
from auditgym.generator import generate_episode
from auditgym.policies import make_policy
from auditgym.runner import run_episode
from auditgym.schema import names, validate


@pytest.mark.parametrize("preset", preset_names())
def test_generated_files_match_schemas(preset):
    for seed in range(4):
        episode = generate_episode(seed, {"preset": preset})
        public = episode.public
        errors = validate(public["manifest"], "manifest") + validate(public["policy"], "policy")
        errors += validate(episode.truth, "truth")
        for table, schema in (("ledger", "ledger_row"), ("invoices", "invoice"), ("payments", "payment"), ("fx", "fx_row")):
            for record in public[table]:
                errors += validate(record, schema)
        assert errors == []


class Sink:
    def __init__(self):
        self.records = []

    def write(self, record):
        self.records.append(json.loads(json.dumps(record)))


@pytest.mark.parametrize("policy", ["oracle", "scripted_auditor", "naive_reconciler", "flag_all", "silent"])
def test_trajectories_actions_and_submissions_match_schemas(make_env, policy):
    sink = Sink()
    env = make_env(allow_oracle=True)
    for seed in (1, 2):
        run_episode(env, make_policy(policy), seed, {"preset": "interacting"}, sink)
    for record in sink.records:
        assert validate(record, "trajectory_record") == []
        action = record["action"]
        if action is None:
            continue
        assert validate(action, "action") == []
        if action["name"] == "submit":
            assert validate(action["args"]["findings"], "findings") == []


def test_every_fx_record_seen_by_agent_has_an_id(make_env):
    env = make_env()
    env.reset(5)
    records = env.step({"name": "list_records", "args": {"table": "fx"}})[0]["result"]["records"]
    assert all(validate(r, "fx_row") == [] and "fx_id" in r for r in records)


@pytest.mark.parametrize(
    "instance, schema",
    [
        ({"episode_id": "E", "findings": [{"type": "fraud"}]}, "findings"),
        ({"episode_id": "E", "findings": []}, "truth"),
        ({"name": "shell"}, "action"),
        ({"row_id": "R", "timestamp": "2026-01-01 10:00", "account": "1200", "debit_minor": 1, "credit_minor": 0,
          "currency": "USD", "entity_id": "C", "invoice_id": None, "payment_id": None}, "ledger_row"),
        ({"date": "2026-01-01", "source_currency": "EUR", "target_currency": "USD", "rate": "1e3"}, "fx_row"),
    ],
)
def test_schemas_reject_bad_instances(instance, schema):
    assert validate(instance, schema) != []


def test_all_schemas_are_valid_json_with_ids():
    for name in names():
        body = json.loads(json.dumps(__import__("auditgym.schema", fromlist=["load"]).load(name)))
        assert body["$schema"].endswith("2020-12/schema")
        assert body["title"] == name
