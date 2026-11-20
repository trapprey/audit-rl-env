from collections import defaultdict

import pytest

from auditgym import spec
from auditgym.env import AuditEnv
from auditgym.generator import generate_episode
from auditgym.judge.client import JudgeClient
from auditgym.timeutil import parse_ts


def nested(depth):
    value = []
    for _ in range(depth):
        value = [value]
    return value


HOSTILE_ACTIONS = [
    {"name": "list_records", "args": {"table": ["ledger"]}},
    {"name": "list_records", "args": {"table": {"name": "ledger"}}},
    {"name": "get_record", "args": {"table": {"a": 1}, "id": "x"}},
    {"name": "get_record", "args": {"table": "ledger", "id": ["x"]}},
    {"name": "get_document", "args": {"doc_id": {"x": 1}}},
    {"name": "list_records", "args": {"table": "ledger", "cursor": 50}},
    {"name": "list_records", "args": {"table": "ledger", "cursor": "ledger@٥٠"}},
    {"name": "list_records", "args": {"table": "ledger", "limit": 50.0}},
    {"name": "calculator", "args": {"expression": nested(3000)}},
    {"name": "calculator", "args": {"expression": "9" * 299}},
    {"name": ["submit"], "args": {}},
    "[" * 100_000 + "]" * 100_000,
    "[" * 300_000,
    b"\xff\xfe",
    None,
]


@pytest.mark.parametrize("action", HOSTILE_ACTIONS, ids=[f"action{i}" for i in range(len(HOSTILE_ACTIONS))])
def test_hostile_actions_return_errors_instead_of_crashing(make_env, action):
    env = make_env()
    env.reset(1)
    obs, reward, terminated, truncated, _ = env.step(action)
    assert obs["error"] is not None and obs["result"] is None
    assert (reward, terminated, truncated) == (0.0, False, False)
    assert obs["remaining_steps"] == spec.MAX_STEPS - 1


@pytest.mark.parametrize(
    "findings",
    [
        {"episode_id": "x", "findings": {1, 2}},
        nested(5000),
        {"episode_id": "x", "findings": [{"type": float("nan")}]},
        {"episode_id": "x", "findings": [], "blob": "x" * 300_000},
    ],
    ids=["python_set", "deep_nesting", "nan", "too_large"],
)
def test_unserializable_submissions_score_minus_one_through_subprocess_judge(findings):
    client = JudgeClient(mode="train")
    env = AuditEnv(judge=client, mode="train")
    try:
        env.reset(3, {"preset": "single"})
        obs, reward, terminated, truncated, info = env.step({"name": "submit", "args": {"findings": findings}})
        assert terminated and reward == -1.0
        assert info["metrics"]["valid"] is False
    finally:
        client.close()


def test_duplicate_copies_do_not_stand_out_by_timezone():
    unusual = total = 0
    for seed in range(90):
        episode = generate_episode(seed, {"preset": "heavy"})
        public = episode.public
        company = public["manifest"]["timezone"]
        invoices = {d["invoice_id"]: d for d in public["invoices"]}
        offsets = defaultdict(list)
        for doc in public["invoices"] + public["payments"]:
            stamp = doc.get("issued_at") or doc["timestamp"]
            offsets[doc["entity_id"]].append((doc["doc_id"], "+00:00" if stamp.endswith("Z") else stamp[-6:]))
        for issue in episode.truth["issues"]:
            if issue["type"] != "duplicate_invoice":
                continue
            copy = max((invoices[i] for i in issue["affected_ids"]), key=lambda d: parse_ts(d["issued_at"]))
            mine = "+00:00" if copy["issued_at"].endswith("Z") else copy["issued_at"][-6:]
            others = {off for doc_id, off in offsets[copy["entity_id"]] if doc_id != copy["doc_id"]}
            total += 1
            unusual += mine not in others and mine not in (company, "+00:00")
    assert total > 50
    assert unusual / total < 0.08
