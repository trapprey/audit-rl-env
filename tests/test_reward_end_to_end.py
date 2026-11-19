import pytest

from auditgym.policies import make_policy
from auditgym.runner import run_episode
from conftest import submit

ERRONEOUS = [(seed, {"preset": preset}) for seed in (1, 2, 3) for preset in ("single", "interacting", "heavy")]
CLEAN = [(seed, {"preset": "clean"}) for seed in range(6)]


def reward_of(env, seed, config, findings_fn, warmup=0):
    _, info = env.reset(seed, config)
    for _ in range(warmup):
        env.step({"tool": "calculator", "args": {"expression": "1+1"}})
    findings = findings_fn(env, info)
    return env.step({"tool": "submit", "args": {"findings": findings}})[1]


def oracle(env, info):
    return env.oracle_findings()


def silent(env, info):
    return {"episode_id": info["episode_id"], "findings": []}


@pytest.mark.parametrize("seed, config", ERRONEOUS)
def test_correct_answer_beats_silence_on_erroneous_books(make_env, seed, config):
    env = make_env(allow_oracle=True)
    assert reward_of(env, seed, config, oracle) > reward_of(env, seed, config, silent)


@pytest.mark.parametrize("seed, config", CLEAN)
def test_silence_beats_accusations_on_clean_books(make_env, seed, config):
    env = make_env()
    quiet = reward_of(env, seed, config, silent)
    for name in ("flag_all", "naive_reconciler"):
        summary = run_episode(env, make_policy(name), seed, config)
        assert quiet > summary["train_metrics"]["f1"] * 2 - 1 - 0.002 * summary["train_metrics"]["inspection_steps"] - 1e-9
        assert summary["train_metrics"]["fp"] > 0


@pytest.mark.parametrize("seed, config", ERRONEOUS[:4])
def test_duplicated_findings_do_not_help(make_env, seed, config):
    env = make_env(allow_oracle=True)

    def doubled(env, info):
        out = env.oracle_findings()
        out["findings"] = out["findings"] * 2
        return out

    assert reward_of(env, seed, config, doubled) < reward_of(env, seed, config, oracle)


@pytest.mark.parametrize("seed, config", ERRONEOUS + CLEAN)
def test_flag_all_attack_is_negative(make_env, seed, config):
    summary = run_episode(make_env(), make_policy("flag_all"), seed, config)
    assert summary["env_return"] < 0


def test_step_savings_only_slightly_separate_equal_answers(make_env):
    env = make_env(allow_oracle=True)
    fast = reward_of(env, 2, {"preset": "single"}, oracle)
    slow = reward_of(env, 2, {"preset": "single"}, oracle, warmup=30)
    assert fast > slow
    assert fast - slow == pytest.approx(0.06)


def test_guessed_and_unsupported_evidence_fail(make_env):
    env = make_env(allow_oracle=True)

    def tamper(mutate):
        def fn(env, info):
            out = env.oracle_findings()
            for f in out["findings"]:
                mutate(f)
            return out

        return fn

    def guessed(f):
        f["evidence_ids"] = f["evidence_ids"][:-1] + ["INVENTED-ID"]

    def unsupported(f):
        f["evidence_ids"] = f["evidence_ids"][:1]

    def wrong_currency(f):
        f["currency"] = "XXX"

    for mutate in (guessed, unsupported, wrong_currency):
        assert reward_of(env, 3, {"preset": "single"}, tamper(mutate)) == pytest.approx(-1.0)


def test_scripted_auditor_solves_within_budget(make_env):
    for seed, config in ERRONEOUS:
        summary = run_episode(make_env(), make_policy("scripted_auditor"), seed, config)
        assert summary["train_metrics"]["f1"] == 1.0
        assert summary["steps"] < 40
