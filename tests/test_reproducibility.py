import json

import pytest

from auditgym.policies import make_policy
from auditgym.runner import run_episode
from conftest import submit


class Recorder:
    def __init__(self):
        self.records = []

    def write(self, record):
        self.records.append(json.dumps(record, sort_keys=True))


@pytest.mark.parametrize("policy", ["scripted_auditor", "random_explorer", "naive_reconciler"])
def test_same_seed_config_and_actions_reproduce_everything(make_env, policy):
    runs = []
    for _ in range(2):
        sink = Recorder()
        run_episode(make_env(), make_policy(policy), 42, {"preset": "interacting"}, sink)
        runs.append(sink.records)
    assert runs[0] == runs[1]


def test_reset_in_the_middle_does_not_leak_state(make_env):
    env = make_env()
    first = Recorder()
    run_episode(env, make_policy("scripted_auditor"), 5, {"preset": "single"}, first)
    run_episode(env, make_policy("random_explorer"), 6, {"preset": "heavy"}, Recorder())
    again = Recorder()
    run_episode(env, make_policy("scripted_auditor"), 5, {"preset": "single"}, again)
    assert first.records == again.records


@pytest.mark.parametrize("seed", [3, 8, 13])
@pytest.mark.parametrize("presentation", [111, 222])
def test_permutation_and_relabeling_preserve_oracle_and_reward(make_env, seed, presentation):
    env = make_env(allow_oracle=True)
    base_cfg = {"preset": "heavy"}
    moved_cfg = {"preset": "heavy", "presentation_seed": presentation, "ledger_order": "shuffled"}

    obs_a, _ = env.reset(seed, base_cfg)
    oracle_a = env.oracle_findings()
    reward_a = env.step({"name": "submit", "args": {"findings": oracle_a}})[1]

    obs_b, _ = env.reset(seed, moved_cfg)
    oracle_b = env.oracle_findings()
    reward_b = env.step({"name": "submit", "args": {"findings": oracle_b}})[1]

    assert reward_a == reward_b == 1.0
    shape = lambda o: sorted((f["type"], f["claimed_amount_minor"], f["currency"]) for f in o["findings"])
    assert shape(oracle_a) == shape(oracle_b)
    ids_a = {i for f in oracle_a["findings"] for i in f["affected_ids"]}
    ids_b = {i for f in oracle_b["findings"] for i in f["affected_ids"]}
    assert not ids_a & ids_b

    scripted = []
    for cfg in (base_cfg, moved_cfg):
        summary = run_episode(env, make_policy("scripted_auditor"), seed, cfg)
        scripted.append(summary["train_metrics"]["f1"])
    assert scripted[0] == scripted[1]


def test_old_ids_do_not_score_after_relabeling(make_env):
    env = make_env(allow_oracle=True)
    env.reset(4, {"preset": "single"})
    stale = env.oracle_findings()
    _, info = env.reset(4, {"preset": "single", "presentation_seed": 999})
    stale["episode_id"] = info["episode_id"]
    _, reward, _, _, info = env.step({"name": "submit", "args": {"findings": stale}})
    assert info["metrics"]["tp"] == 0 and reward == -1.0
