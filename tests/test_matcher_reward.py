import pytest

from auditgym import spec
from auditgym.judge.matcher import evaluate, terminal_reward

EID = "EP-test"
KNOWN = {"D1", "D2", "D3", "D4", "L1", "L2", "L3", "L4", "L5", "P1", "I1", "I2", "I3"}
TRUTH = {
    "episode_id": EID,
    "issues": [
        {
            "issue_id": "ISS-01",
            "type": "duplicate_invoice",
            "affected_ids": ["I1", "I2"],
            "expected_amount_minor": 50000,
            "currency": "USD",
            "evidence_sets": [["D1", "D2"]],
            "scenario": "single",
        },
        {
            "issue_id": "ISS-02",
            "type": "payment_allocation",
            "affected_ids": ["I3", "P1"],
            "expected_amount_minor": 1200,
            "currency": "EUR",
            "evidence_sets": [["D3", "L3"]],
            "scenario": "single",
        },
    ],
}
CLEAN = {"episode_id": EID, "issues": []}


def f(kind, ids, amount, currency, evidence):
    return {
        "type": kind,
        "affected_ids": ids,
        "claimed_amount_minor": amount,
        "currency": currency,
        "evidence_ids": evidence,
        "explanation": "x",
    }


GOOD = [
    f("duplicate_invoice", ["I2", "I1"], 50000, "USD", ["D1", "D2"]),
    f("payment_allocation", ["P1", "I3"], 1200, "EUR", ["D3", "L3", "L4"]),
]


def run(findings, truth=TRUTH, steps=10, episode_id=EID):
    return evaluate({"episode_id": episode_id, "findings": findings}, truth, KNOWN, steps)


def test_full_correct_answer_beats_silence_on_erroneous_books():
    correct, silent = run(GOOD), run([])
    assert correct["f1"] == 1.0 and correct["tp"] == 2
    assert correct["reward"] > silent["reward"]
    assert silent["reward"] == pytest.approx(-1.0)


def test_silence_beats_false_accusations_on_clean_books():
    silent = run([], CLEAN)
    accused = run(GOOD, CLEAN)
    assert silent["f1"] == 1.0
    assert accused["f1"] == 0.0
    assert silent["reward"] > accused["reward"]


def test_repeating_a_finding_does_not_pay():
    once = run(GOOD)
    twice = run(GOOD + GOOD)
    assert twice["tp"] == once["tp"] == 2
    assert twice["fp"] == 2
    assert twice["reward"] < once["reward"]


def test_flagging_everything_scores_badly():
    spam = [
        f(kind, [a, b], 1, cur, ["D1", "D2", "D3", "L1"])
        for kind in spec.FINDING_TYPES
        for a, b in [("I1", "I2"), ("I3", "P1"), ("I1", "I3")]
        for cur in ("USD", "EUR")
    ][: spec.MAX_FINDINGS]
    result = run(spam)
    assert result["reward"] < run([])["reward"] + 0.5
    assert result["reward"] < 0


def test_step_cost_only_separates_equal_answers_slightly():
    fast, slow = run(GOOD, steps=5), run(GOOD, steps=45)
    assert fast["f1"] == slow["f1"]
    assert 0 < fast["reward"] - slow["reward"] <= 0.1
    assert fast["reward"] - slow["reward"] == pytest.approx(40 * spec.STEP_COST)


def test_guessed_evidence_ids_are_rejected():
    guessed = [f("duplicate_invoice", ["I1", "I2"], 50000, "USD", ["D1", "D2", "D999"])]
    assert run(guessed)["tp"] == 0


def test_evidence_must_cover_an_allowed_set():
    partial = [f("duplicate_invoice", ["I1", "I2"], 50000, "USD", ["D1", "L1"])]
    assert run(partial)["tp"] == 0
    assert run(partial)["fp"] == 1


def test_currency_must_match():
    wrong = [f("duplicate_invoice", ["I1", "I2"], 50000, "EUR", ["D1", "D2"])]
    assert run(wrong)["tp"] == 0


def test_currency_and_ids_are_normalized():
    loose = [f("duplicate_invoice", [" I2", "I1", "I1"], 50000, "usd", ["D2", "D1"])]
    assert run(loose)["tp"] == 1


@pytest.mark.parametrize("delta, matches", [(0, True), (1, True), (-1, True), (2, False), (-2, False)])
def test_amount_tolerance(delta, matches):
    finding = [f("duplicate_invoice", ["I1", "I2"], 50000 + delta, "USD", ["D1", "D2"])]
    assert (run(finding)["tp"] == 1) is matches


def test_wrong_type_or_ids_do_not_match():
    assert run([f("period_cutoff", ["I1", "I2"], 50000, "USD", ["D1", "D2"])])["tp"] == 0
    assert run([f("duplicate_invoice", ["I1"], 50000, "USD", ["D1", "D2"])])["tp"] == 0


def test_empty_truth_rules():
    assert run([], CLEAN)["f1"] == 1.0
    assert run(GOOD[:1], CLEAN)["f1"] == 0.0
    assert run([], CLEAN)["precision"] == 0.0


BAD_PAYLOADS = [
    None,
    [],
    {"episode_id": EID},
    {"episode_id": EID, "findings": {}},
    {"episode_id": "EP-other", "findings": []},
    {"episode_id": EID, "findings": [], "extra": 1},
    {"episode_id": EID, "findings": [dict(GOOD[0], extra=1)]},
    {"episode_id": EID, "findings": [dict(GOOD[0], type="fraud")]},
    {"episode_id": EID, "findings": [dict(GOOD[0], claimed_amount_minor=True)]},
    {"episode_id": EID, "findings": [dict(GOOD[0], claimed_amount_minor=-5)]},
    {"episode_id": EID, "findings": [dict(GOOD[0], claimed_amount_minor=12.5)]},
    {"episode_id": EID, "findings": [dict(GOOD[0], currency="DOLLARS")]},
    {"episode_id": EID, "findings": [dict(GOOD[0], affected_ids=[])]},
    {"episode_id": EID, "findings": [dict(GOOD[0], affected_ids="I1")]},
    {"episode_id": EID, "findings": [dict(GOOD[0], evidence_ids=[f"L{i}" for i in range(11)])]},
    {"episode_id": EID, "findings": [dict(GOOD[0], explanation="x" * 1001)]},
    {"episode_id": EID, "findings": [GOOD[0]] * 31},
]


@pytest.mark.parametrize("payload", BAD_PAYLOADS)
def test_invalid_structure_costs_minus_one(payload):
    result = evaluate(payload, TRUTH, KNOWN, 3)
    assert result["valid"] is False
    assert result["reward"] == -1.0


def test_truncation_is_an_empty_answer():
    result = evaluate(None, TRUTH, KNOWN, 100, truncated=True)
    assert result["valid"] and result["f1"] == 0.0
    clean = evaluate(None, CLEAN, KNOWN, 100, truncated=True)
    assert clean["f1"] == 1.0
    assert clean["reward"] == pytest.approx(1 - 0.2)


def test_reward_is_clipped():
    assert terminal_reward(1.0, 0) == 1.0
    assert terminal_reward(0.0, 100) == -1.0
    assert -1.0 <= terminal_reward(0.3, 77) <= 1.0


def test_matching_is_deterministic_and_one_to_one():
    findings = [GOOD[0], GOOD[0], GOOD[1]]
    first = run(findings)
    second = run(findings)
    assert first["matches"] == second["matches"] == [[0, "ISS-01"], [2, "ISS-02"]]
