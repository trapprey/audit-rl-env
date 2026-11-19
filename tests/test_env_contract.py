import json

import pytest

from auditgym import spec
from conftest import submit


def calc():
    return {"tool": "calculator", "args": {"expression": "1 + 1"}}


def test_reset_contract(make_env):
    env = make_env()
    obs, info = env.reset(11, {"preset": "standard"})
    json.dumps(obs)
    json.dumps(info)
    assert set(obs) == {"step", "steps_remaining", "task", "manifest", "policy", "tools"}
    assert obs["steps_remaining"] == spec.MAX_STEPS
    assert set(spec.FINDING_TYPES) == set(obs["policy"]["findings"])
    manifest = obs["manifest"]
    for key in ("schema_version", "episode_id", "base_currency", "period_start", "period_end", "timezone", "record_counts", "rounding"):
        assert key in manifest
    assert info["episode_id"] == manifest["episode_id"]


def test_step_contract(make_env):
    env = make_env()
    env.reset(11)
    out = env.step({"tool": "list_records", "args": {"table": "ledger"}})
    assert len(out) == 5
    obs, reward, terminated, truncated, info = out
    json.dumps(obs)
    assert reward == 0.0 and terminated is False and truncated is False
    assert obs["steps_remaining"] == spec.MAX_STEPS - 1
    assert len(obs["result"]["records"]) == spec.PAGE_LIMIT


def test_step_before_reset_raises(make_env):
    with pytest.raises(RuntimeError):
        make_env().step(calc())


@pytest.mark.parametrize("table", spec.TABLES)
def test_pagination_covers_table_once_and_cursors_are_stable(make_env, table):
    env = make_env()
    obs, _ = env.reset(12, {"preset": "quarter"})
    total = obs["manifest"]["record_counts"][table]
    seen, cursor, pages = [], None, []
    while True:
        result = env.step({"tool": "list_records", "args": {"table": table, "cursor": cursor, "limit": 50}})[0]["result"]
        assert len(result["records"]) <= spec.PAGE_LIMIT
        pages.append((cursor, result["records"]))
        seen += result["records"]
        cursor = result["next_cursor"]
        if cursor is None:
            break
    assert len(seen) == total
    key = {"ledger": "row_id", "invoices": "invoice_id", "payments": "payment_id", "fx": "fx_id"}[table]
    assert len({r[key] for r in seen}) == total
    again = env.step({"tool": "list_records", "args": {"table": table, "cursor": pages[-1][0], "limit": 50}})[0]
    assert again["result"]["records"] == pages[-1][1]


@pytest.mark.parametrize(
    "action, code",
    [
        ({"tool": "list_records", "args": {"table": "ledger", "limit": 51}}, "bad_arguments"),
        ({"tool": "list_records", "args": {"table": "ledger", "limit": 0}}, "bad_arguments"),
        ({"tool": "list_records", "args": {"table": "secrets"}}, "bad_arguments"),
        ({"tool": "list_records", "args": {"table": "ledger", "cursor": "fx@50"}}, "bad_cursor"),
        ({"tool": "list_records", "args": {"table": "ledger", "cursor": "ledger@99999999"}}, "bad_cursor"),
        ({"tool": "list_records", "args": {"table": "ledger", "cursor": "ledger@-1"}}, "bad_cursor"),
        ({"tool": "get_record", "args": {"table": "ledger", "id": "NOPE"}}, "not_found"),
        ({"tool": "get_record", "args": {"table": "ledger"}}, "bad_arguments"),
        ({"tool": "get_document", "args": {"doc_id": "NOPE"}}, "not_found"),
        ({"tool": "calculator", "args": {"expression": "__import__('os')"}}, "calculator_error"),
        ({"tool": "shell", "args": {"cmd": "ls"}}, "unknown_tool"),
        ({"tool": "list_records", "args": "ledger"}, "bad_arguments"),
        ({"args": {}}, "invalid_action"),
        ("{not json", "invalid_action"),
        (42, "invalid_action"),
    ],
)
def test_bad_actions_cost_a_step_and_return_bounded_errors(make_env, action, code):
    env = make_env()
    env.reset(13)
    obs, reward, terminated, truncated, _ = env.step(action)
    assert obs["error"]["code"] == code
    assert len(obs["error"]["message"]) <= 200
    assert obs["steps_remaining"] == spec.MAX_STEPS - 1
    assert (reward, terminated, truncated) == (0.0, False, False)


def test_error_messages_do_not_echo_input(make_env):
    env = make_env()
    env.reset(13)
    obs = env.step({"tool": "get_record", "args": {"table": "invoices", "id": "PROBE-12345"}})[0]
    assert "PROBE-12345" not in json.dumps(obs)


def test_lookup_tools(make_env):
    env = make_env()
    env.reset(14)
    invoice = env.step({"tool": "list_records", "args": {"table": "invoices", "limit": 1}})[0]["result"]["records"][0]
    by_id = env.step({"tool": "get_record", "args": {"table": "invoices", "id": invoice["invoice_id"]}})[0]
    by_doc = env.step({"tool": "get_document", "args": {"doc_id": invoice["doc_id"]}})[0]
    assert by_id["result"]["record"] == invoice == by_doc["result"]["document"]
    fx = env.step({"tool": "list_records", "args": {"table": "fx", "limit": 1}})[0]["result"]["records"][0]
    again = env.step({"tool": "get_record", "args": {"table": "fx", "id": fx["fx_id"]}})[0]
    assert again["result"]["record"] == fx
    value = env.step('{"tool": "calculator", "args": {"expression": "12.5 * 4"}}')[0]
    assert value["result"]["value"] == "50"


def test_returned_records_cannot_mutate_the_environment(make_env):
    env = make_env()
    env.reset(14)
    first = env.step({"tool": "list_records", "args": {"table": "ledger", "limit": 1}})[0]["result"]["records"][0]
    first["debit_minor"] = -1
    again = env.step({"tool": "list_records", "args": {"table": "ledger", "limit": 1}})[0]["result"]["records"][0]
    assert again["debit_minor"] != -1


def test_step_limit_truncates_with_empty_answer(make_env):
    env = make_env()
    env.reset(15, {"preset": "single"})
    for _ in range(spec.MAX_STEPS - 1):
        _, _, terminated, truncated, _ = env.step(calc())
        assert not terminated and not truncated
    obs, reward, terminated, truncated, info = env.step(calc())
    assert truncated is True and terminated is False
    assert obs["steps_remaining"] == 0
    assert info["metrics"]["f1"] == 0.0
    assert reward == pytest.approx(-1.0)
    after = env.step(calc())
    assert after[0]["error"]["code"] == "episode_finished"
    assert after[1] == 0.0 and after[3] is True


def test_submit_on_last_step_terminates(make_env):
    env = make_env()
    obs, info = env.reset(15, {"preset": "clean"})
    for _ in range(spec.MAX_STEPS - 1):
        env.step(calc())
    obs, reward, terminated, truncated, info = env.step(submit(info["episode_id"], []))
    assert terminated is True and truncated is False
    assert info["metrics"]["inspection_steps"] == spec.MAX_STEPS - 1
    assert reward == pytest.approx(1 - spec.STEP_COST * 99)


def test_submit_terminates_and_resubmission_is_ignored(make_env):
    env = make_env()
    _, info = env.reset(16, {"preset": "clean"})
    obs, reward, terminated, truncated, _ = env.step(submit(info["episode_id"], []))
    assert terminated and not truncated and reward == pytest.approx(1.0)
    obs, reward, terminated, truncated, _ = env.step(submit(info["episode_id"], []))
    assert obs["error"]["code"] == "episode_finished" and reward == 0.0
    assert len(env.judge.results()) == 1


def test_invalid_submission_ends_episode_with_minus_one(make_env):
    env = make_env()
    env.reset(17)
    obs, reward, terminated, truncated, info = env.step({"tool": "submit", "args": {"findings": {"oops": True}}})
    assert terminated and reward == -1.0
    assert info["metrics"]["valid"] is False
    assert env.step({"tool": "submit", "args": {}})[0]["error"]["code"] == "episode_finished"


def test_time_limit_truncates(make_env):
    env = make_env(time_limit=0.0)
    env.reset(18)
    obs, reward, terminated, truncated, _ = env.step(calc())
    assert truncated and not terminated
    assert obs["error"]["code"] == "time_limit"


def test_reset_isolates_episodes(make_env):
    env = make_env()
    env.reset(19)
    for _ in range(5):
        env.step(calc())
    obs, _ = env.reset(20)
    assert obs["steps_remaining"] == spec.MAX_STEPS
    first = env.step({"tool": "list_records", "args": {"table": "invoices", "limit": 5}})[0]
    assert first["steps_remaining"] == spec.MAX_STEPS - 1
    env.reset(19)
    assert env.step({"tool": "list_records", "args": {"table": "invoices", "limit": 5}})[0] != first
