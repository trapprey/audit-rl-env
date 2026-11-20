import json

import pytest

from auditgym.generator import generate_episode
from auditgym.judge import JudgeError
from auditgym.policies import make_policy
from auditgym.runner import run_episode
from auditgym.storage import save_episode
from auditgym.transform import load_episode, transform_folder


@pytest.fixture
def saved(tmp_path):
    def factory(seed, preset):
        folder = tmp_path / f"{preset}-{seed}"
        save_episode(generate_episode(seed, {"preset": preset}), folder)
        return folder

    return factory


def oracle_reward(env, folder):
    env.reset(0, {"episode_dir": str(folder)})
    findings = env.oracle_findings()
    return env.step({"name": "submit", "args": {"findings": findings}})[1], findings


def test_episode_loaded_from_files_matches_generated(make_env, saved):
    folder = saved(4, "interacting")
    env = make_env()
    from_files, _ = env.reset(0, {"episode_dir": str(folder)})
    generated, _ = env.reset(4, {"preset": "interacting"})
    assert from_files == generated


@pytest.mark.parametrize("seed, preset", [(1, "heavy"), (2, "interacting"), (3, "clean"), (4, "quarter"), (5, "single")])
def test_shuffle_and_relabel_preserve_oracle_and_scripted_results(make_env, saved, tmp_path, seed, preset):
    source = saved(seed, preset)
    target = tmp_path / "moved"
    transform_folder(source, target, seed + 500)
    env = make_env(allow_oracle=True)
    reward_a, oracle_a = oracle_reward(env, source)
    reward_b, oracle_b = oracle_reward(env, target)
    assert reward_a == reward_b == 1.0
    shape = lambda o: sorted((f["type"], f["claimed_amount_minor"], f["currency"]) for f in o["findings"])
    assert shape(oracle_a) == shape(oracle_b)
    ids_a = {i for f in oracle_a["findings"] for i in f["affected_ids"]}
    ids_b = {i for f in oracle_b["findings"] for i in f["affected_ids"]}
    assert not ids_a & ids_b
    scripted = [
        run_episode(env, make_policy("scripted_auditor"), 0, {"episode_dir": str(folder)})["train_metrics"]
        for folder in (source, target)
    ]
    assert scripted[0]["f1"] == scripted[1]["f1"]
    assert scripted[0]["tp"] == scripted[1]["tp"]


def test_transform_really_moves_rows_and_ids(saved, tmp_path):
    source = saved(7, "standard")
    transform_folder(source, tmp_path / "moved", 1)
    a, b = load_episode(source).public, load_episode(tmp_path / "moved").public
    assert len(a["ledger"]) == len(b["ledger"])
    assert {r["row_id"] for r in a["ledger"]}.isdisjoint({r["row_id"] for r in b["ledger"]})
    assert [r["debit_minor"] for r in a["ledger"]] != [r["debit_minor"] for r in b["ledger"]]
    assert sorted(r["debit_minor"] for r in a["ledger"]) == sorted(r["debit_minor"] for r in b["ledger"])


def test_broken_or_tampered_files_are_refused(make_env, saved, tmp_path):
    env = make_env()
    with pytest.raises(JudgeError):
        env.reset(0, {"episode_dir": str(tmp_path / "missing")})
    folder = saved(2, "single")
    truth_path = folder / "grader" / "truth.json"
    truth = json.loads(truth_path.read_text())
    truth["issues"][0]["expected_amount_minor"] += 1000
    truth_path.write_text(json.dumps(truth))
    with pytest.raises(JudgeError):
        env.reset(0, {"episode_dir": str(folder)})


def test_reset_accepts_gymnasium_style_arguments(make_env):
    env = make_env()
    obs, info = env.reset(seed=None)
    assert obs["remaining_steps"] == 100
    first, _ = env.reset(seed=8, options={"preset": "clean"})
    second, _ = env.reset(8, {"preset": "clean"})
    assert first == second
