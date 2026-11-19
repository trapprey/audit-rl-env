import json
import os

import pytest

from auditgym.env import AuditEnv
from auditgym.judge import JudgeError
from auditgym.judge.client import JudgeClient
from auditgym.policies import make_policy
from auditgym.runner import TrajectoryWriter, run_episode
from conftest import submit, walk_keys

FORBIDDEN_KEYS = {
    "issue_id", "issues", "expected_amount_minor", "evidence_sets", "scenario", "mutations",
    "provenance", "truth", "seed", "presentation_seed", "plan", "config_digest", "token",
}
FORBIDDEN_TEXT = ("truth.json", "provenance.json", "grader/", "evidence_sets")


def test_trajectories_carry_no_private_data(tmp_path, make_env):
    path = tmp_path / "traj.jsonl"
    writer = TrajectoryWriter(path)
    for name in ("scripted_auditor", "random_explorer", "naive_reconciler"):
        env = make_env(mode="train")
        for seed in (1, 2):
            run_episode(env, make_policy(name), seed, {"preset": "heavy"}, writer)
    writer.close()
    for line in path.read_text().splitlines():
        record = json.loads(line)
        keys = set(walk_keys(record["observation"]))
        assert not keys & FORBIDDEN_KEYS
        text = json.dumps(record["observation"])
        assert not any(marker in text for marker in FORBIDDEN_TEXT)


def test_environment_state_holds_no_truth():
    client = JudgeClient(mode="eval")
    env = AuditEnv(judge=client, mode="eval")
    try:
        env.reset(9, {"preset": "heavy"})
        env.step({"tool": "list_records", "args": {"table": "ledger"}})
        state = {k: v for k, v in env.__dict__.items() if k != "judge"}
        dump = json.dumps(state, default=str)
        assert "evidence_sets" not in dump and "expected_amount_minor" not in dump
        assert client._proc.pid != os.getpid()
    finally:
        env.close()
        client.close()


def test_eval_mode_hides_reward_and_diagnostics():
    client = JudgeClient(mode="eval")
    env = AuditEnv(judge=client, mode="eval")
    try:
        _, info = env.reset(10, {"preset": "single"})
        assert set(info) == {"episode_id", "mode", "steps_remaining"}
        obs, reward, terminated, _, info = env.step(submit(info["episode_id"], []))
        assert terminated and reward == 0.0
        assert set(info) == {"steps_remaining"}
        assert set(obs) == {"step", "steps_remaining", "result"}
        _, info = env.reset(10, {"preset": "single"})
        obs, reward, _, _, info = env.step({"tool": "submit", "args": {"findings": "garbage"}})
        assert reward == 0.0 and set(info) == {"steps_remaining"}
        results = client.results()
        assert results[0]["reward"] == pytest.approx(-1.0)
        assert results[1]["valid"] is False
    finally:
        client.close()


def test_oracle_is_disabled_unless_requested(make_env):
    env = make_env()
    env.reset(1)
    with pytest.raises(JudgeError):
        env.oracle_findings()


def test_subprocess_judge_matches_in_process_judge(make_env):
    client = JudgeClient(mode="train", allow_oracle=True)
    remote = AuditEnv(judge=client, mode="train")
    local = make_env(allow_oracle=True)
    try:
        a, _ = remote.reset(21, {"preset": "interacting"})
        b, _ = local.reset(21, {"preset": "interacting"})
        assert a == b
        ra = remote.step({"tool": "submit", "args": {"findings": remote.oracle_findings()}})
        rb = local.step({"tool": "submit", "args": {"findings": local.oracle_findings()}})
        assert ra[1] == rb[1] == 1.0
    finally:
        client.close()
