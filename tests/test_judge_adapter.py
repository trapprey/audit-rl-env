import inspect
import json
import shutil

import pytest

import judge_adapter
from auditgym import spec
from auditgym.transform import transform_folder


def oracle_from_truth(folder):
    truth = json.loads((folder / "grader" / "truth.json").read_text(encoding="utf-8"))
    return {
        "episode_id": truth["episode_id"],
        "findings": [
            {
                "type": issue["type"],
                "affected_ids": issue["affected_ids"],
                "claimed_amount_minor": issue["expected_amount_minor"],
                "currency": issue["currency"],
                "evidence_ids": issue["evidence_sets"][0],
                "explanation": "oracle",
            }
            for issue in truth["issues"]
        ],
    }


@pytest.fixture
def episode(tmp_path, monkeypatch):
    monkeypatch.setenv("AUDITGYM_JUDGE_LOG", str(tmp_path / "judge_log.jsonl"))

    def factory(seed=7, preset="heavy"):
        folder = tmp_path / f"ep-{seed}-{preset}"
        judge_adapter.generate_episode(seed, {"preset": preset}, str(folder))
        return folder

    return factory


def closed(folder):
    return {"mode": "closed", "episode_dir": str(folder), "team_config": {}}


def act(name, **args):
    return {"name": name, "args": args}


def test_adapter_signatures_have_no_defaults():
    expected = {
        "generate_episode": ["seed", "config", "output_dir"],
        "make_env": [],
        "score_episode": ["episode_dir", "findings", "inspection_steps"],
    }
    for name, params in expected.items():
        sig = inspect.signature(getattr(judge_adapter, name))
        assert list(sig.parameters) == params
        assert all(p.default is inspect.Parameter.empty for p in sig.parameters.values())


def test_generate_accepts_team_config_wrapper(tmp_path):
    judge_adapter.generate_episode(3, {"team_config": {"preset": "single"}}, str(tmp_path / "a"))
    judge_adapter.generate_episode(3, {"preset": "single"}, str(tmp_path / "b"))
    a = (tmp_path / "a" / "grader" / "truth.json").read_text(encoding="utf-8")
    b = (tmp_path / "b" / "grader" / "truth.json").read_text(encoding="utf-8")
    assert a == b


def test_closed_contract_end_to_end(episode):
    folder = episode()
    env = judge_adapter.make_env()
    obs, info = env.reset(7, closed(folder))
    assert set(obs) == {"task", "manifest", "policy", "remaining_steps"}
    assert obs["remaining_steps"] == 100
    seen = 0
    cursor = None
    while True:
        obs, reward, terminated, truncated, info = env.step(act("list_records", table="ledger", cursor=cursor, limit=50))
        assert set(obs) == {"result", "error", "remaining_steps"} and obs["error"] is None
        assert set(obs["result"]) == {"records", "next_cursor"}
        assert len(obs["result"]["records"]) <= 50
        assert reward == 0.0 and info == {}
        seen += len(obs["result"]["records"])
        cursor = obs["result"]["next_cursor"]
        if cursor is None:
            break
    first = env.step(act("list_records", table="invoices", cursor=None, limit=1))[0]["result"]["records"][0]
    assert env.step(act("get_record", table="invoices", id=first["invoice_id"]))[0]["result"] == first
    assert env.step(act("get_document", doc_id=first["doc_id"]))[0]["result"] == first
    assert env.step(act("calculator", expression="2.5 * 4"))[0]["result"] == "10"
    missing = env.step(act("get_record", table="ledger", id="NOPE"))[0]
    assert missing["result"] is None and missing["error"]["code"] == "not_found"
    unknown = env.step({"name": "shell", "args": {}})[0]
    assert unknown["result"] is None and unknown["error"]["code"] == "unknown_action"
    used = 100 - unknown["remaining_steps"]
    findings = oracle_from_truth(folder)
    obs, reward, terminated, truncated, info = env.step(act("submit", findings=findings))
    assert obs == {"result": {"accepted": True}, "error": None, "remaining_steps": 100 - used - 1}
    assert reward == 0.0 and terminated and not truncated and info == {}
    assert env.submission["inspection_steps"] == used
    score = judge_adapter.score_episode(str(folder), findings, used)
    assert set(score) == {"tp", "fp", "fn", "precision", "recall", "f1", "return"}
    assert score["f1"] == 1.0 and score["return"] == pytest.approx(1 - spec.STEP_COST * used)


def test_closed_mode_never_reads_the_grader_folder(episode, tmp_path):
    folder = episode()
    public_only = tmp_path / "public-only"
    shutil.copytree(folder / "public", public_only / "public")
    env = judge_adapter.make_env()
    obs, _ = env.reset(7, closed(public_only))
    assert obs["manifest"]["episode_id"]
    obs, reward, terminated, _, _ = env.step(act("submit", findings={"episode_id": obs["manifest"]["episode_id"], "findings": []}))
    assert terminated and reward == 0.0 and obs["result"] == {"accepted": True}


@pytest.mark.parametrize("steps, expected", [(0, 1.0), (10, 0.98), (50, 0.90), (100, 0.80)])
def test_reward_examples_from_the_rules(episode, steps, expected):
    folder = episode()
    score = judge_adapter.score_episode(str(folder), oracle_from_truth(folder), steps)
    assert score["return"] == pytest.approx(expected)


def test_invalid_structures_give_minus_one_and_null_fields(episode, tmp_path):
    folder = episode()
    good = oracle_from_truth(folder)
    first = good["findings"][0]
    broken = [
        dict(good, findings=[first] * 31),
        dict(good, findings=[dict(first, evidence_ids=[f"X{i}" for i in range(11)])]),
        dict(good, findings=[dict(first, explanation="x" * 1001)]),
        {"findings": good["findings"]},
        None,
    ]
    for payload in broken:
        score = judge_adapter.score_episode(str(folder), payload, 0)
        assert score["return"] == -1.0
        assert all(score[k] is None for k in ("tp", "fp", "fn", "precision", "recall", "f1"))
    log = [json.loads(line) for line in (tmp_path / "judge_log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert sum(1 for r in log if not r["valid"]) == len(broken)
    assert all(r["error"] for r in log if not r["valid"])


def test_invalid_submit_in_closed_mode_returns_a_bounded_error(episode):
    folder = episode()
    env = judge_adapter.make_env()
    env.reset(7, closed(folder))
    obs, reward, terminated, _, info = env.step(act("submit", findings={"oops": 1}))
    assert obs["result"] is None and obs["error"]["code"] == "invalid_submission"
    assert len(obs["error"]["message"]) <= 200 and terminated and reward == 0.0 and info == {}


def test_duplicates_and_clean_books(episode):
    folder = episode()
    good = oracle_from_truth(folder)
    doubled = dict(good, findings=good["findings"] * 2)
    once, twice = (judge_adapter.score_episode(str(folder), x, 0) for x in (good, doubled))
    assert twice["tp"] == once["tp"] and twice["fp"] == once["tp"] and twice["return"] < once["return"]
    clean = episode(3, "clean")
    clean_id = oracle_from_truth(clean)["episode_id"]
    silent = judge_adapter.score_episode(str(clean), {"episode_id": clean_id, "findings": []}, 0)
    accused = judge_adapter.score_episode(str(clean), dict(good, episode_id=clean_id), 0)
    assert silent["f1"] == 1.0 and silent["return"] == 1.0
    assert accused["return"] < silent["return"]


def test_step_limit_and_submit_at_action_100(episode):
    folder = episode()
    env = judge_adapter.make_env()
    env.reset(7, closed(folder))
    for _ in range(99):
        env.step(act("calculator", expression="1+1"))
    obs, reward, terminated, truncated, _ = env.step(act("submit", findings=oracle_from_truth(folder)))
    assert terminated and not truncated and obs["result"] == {"accepted": True}
    assert env.submission["inspection_steps"] == 99
    env.reset(7, closed(folder))
    for i in range(100):
        obs, reward, terminated, truncated, _ = env.step(act("calculator", expression="1+1"))
    assert truncated and not terminated and obs["remaining_steps"] == 0 and reward == 0.0


@pytest.mark.parametrize("mode", ["permute", "relabel", "both"])
def test_paired_checks_keep_oracle_score_and_steps(episode, tmp_path, mode):
    folder = episode(11, "interacting")
    moved = tmp_path / f"moved-{mode}"
    transform_folder(folder, moved, 5)
    scores = []
    for f in (folder, moved):
        env = judge_adapter.make_env()
        env.reset(0, closed(f))
        env.step(act("list_records", table="ledger", cursor=None, limit=50))
        findings = oracle_from_truth(f)
        env.step(act("submit", findings=findings))
        scores.append((judge_adapter.score_episode(str(f), findings, env.submission["inspection_steps"]), env.submission["inspection_steps"]))
    assert scores[0] == scores[1]


def test_train_mode_reports_the_same_reward_as_the_scorer(episode):
    folder = episode()
    env = judge_adapter.make_env()
    env.reset(0, {"mode": "train", "episode_dir": str(folder), "team_config": {}})
    env.step(act("list_records", table="fx", cursor=None, limit=50))
    findings = oracle_from_truth(folder)
    _, reward, terminated, _, _ = env.step(act("submit", findings=findings))
    env.close()
    assert terminated and reward == pytest.approx(judge_adapter.score_episode(str(folder), findings, 1)["return"])
