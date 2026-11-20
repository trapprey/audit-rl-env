import pytest

from auditgym.judge.matcher import evaluate

KNOWN = {"R1", "R2", "R3", "R4", "R5", "D1", "D2", "D3"}
TYPES = ("duplicate_invoice", "payment_allocation", "fx_rate", "period_cutoff")


def issue(issue_id="A", affected=("R1",), evidence=("D1", "D2")):
    return {
        "issue_id": issue_id,
        "type": "duplicate_invoice",
        "affected_ids": list(affected),
        "expected_amount_minor": 10000,
        "currency": "KZT",
        "evidence_sets": [list(evidence)],
    }


def finding(**changes):
    out = {
        "type": "duplicate_invoice",
        "affected_ids": ["R1"],
        "claimed_amount_minor": 10000,
        "currency": "KZT",
        "evidence_ids": ["D1", "D2"],
        "explanation": "short",
    }
    out.update(changes)
    return out


def score(findings, issues, steps=10):
    result = evaluate({"episode_id": "E", "findings": findings}, {"episode_id": "E", "issues": issues}, KNOWN, steps)
    return result


A = issue()
B = issue("B", ("R2",))

CASES = [
    ("01", [finding()], [A], 10, (1, 0, 0), 0.98),
    ("02", [], [A], 10, (0, 0, 1), -1.0),
    ("03", [], [], 10, (0, 0, 0), 0.98),
    ("04", [finding()], [], 10, (0, 1, 0), -1.0),
    ("05", [finding(), finding()], [A], 10, (1, 1, 0), 47 / 150),
    ("06", [finding(type="fx_rate")], [A], 10, (0, 1, 1), -1.0),
    ("07", [finding(affected_ids=["R3"])], [A], 10, (0, 1, 1), -1.0),
    ("08", [finding(claimed_amount_minor=10001)], [A], 10, (0, 1, 1), -1.0),
    ("09", [finding(currency="USD")], [A], 10, (0, 1, 1), -1.0),
    ("10", [finding(evidence_ids=["D1"])], [A], 10, (0, 1, 1), -1.0),
    ("11", [finding(evidence_ids=["D1", "U"])], [A], 10, (0, 1, 1), -1.0),
    ("12", [finding(type=t, affected_ids=[r]) for r in ("R1", "R2", "R3", "R4", "R5") for t in TYPES], [A], 10, (1, 19, 0), -871 / 1050),
    ("13", [finding()], [A, B], 10, (1, 0, 1), 47 / 150),
    ("14", [finding(), finding(affected_ids=["R3"])], [A, B], 10, (1, 1, 1), -0.02),
    ("15", [finding()], [A], 0, (1, 0, 0), 1.0),
    ("16", [finding()], [A], 50, (1, 0, 0), 0.90),
    ("17", [finding()], [A], 100, (1, 0, 0), 0.80),
]


@pytest.mark.parametrize("name, findings, issues, steps, counts, reward", CASES, ids=[c[0] for c in CASES])
def test_published_matcher_table(name, findings, issues, steps, counts, reward):
    result = score(findings, issues, steps)
    assert result["valid"]
    assert (result["tp"], result["fp"], result["fn"]) == counts
    assert abs(result["reward"] - reward) <= 1e-9


@pytest.mark.parametrize(
    "findings",
    [
        [finding()] * 31,
        [finding(evidence_ids=[f"D{i}" for i in range(11)])],
        [finding(explanation="x" * 1001)],
    ],
    ids=["18_31_findings", "19_11_evidence", "20_1001_chars"],
)
def test_published_structure_limits(findings):
    result = score(findings, [A])
    assert result["valid"] is False and result["reward"] == -1.0 and result["error"]


A2 = issue("A", ("R1", "R2"))


def test_m01_affected_ids_are_a_set():
    assert score([finding(affected_ids=["R2", "R1", "R1"])], [A2])["tp"] == 1


def test_m02_extra_affected_record_is_fp():
    r = score([finding(affected_ids=["R1", "R2", "R3"])], [A2])
    assert (r["fp"], r["fn"]) == (1, 1)


def test_m03_extra_existing_evidence_is_fine():
    assert score([finding(affected_ids=["R1", "R2"], evidence_ids=["D2", "D1", "D3"])], [A2])["tp"] == 1


def test_m04_unknown_evidence_blocks_the_match():
    r = score([finding(affected_ids=["R1", "R2"], evidence_ids=["D1", "D2", "U"])], [A2])
    assert (r["fp"], r["fn"]) == (1, 1)


def test_m05_eleven_evidence_ids_with_repeats_is_structural():
    r = score([finding(affected_ids=["R1", "R2"], evidence_ids=["D1"] * 10 + ["D2"])], [A2])
    assert r["valid"] is False and r["reward"] == -1.0


@pytest.mark.parametrize("currency", ["kzt", "KZT ", " KZT"])
def test_m06_currency_is_exact(currency):
    r = score([finding(affected_ids=["R1", "R2"], currency=currency)], [A2])
    assert (r["fp"], r["fn"]) == (1, 1)


def test_m07_amount_off_by_one_with_zero_tolerance():
    r = score([finding(affected_ids=["R1", "R2"], claimed_amount_minor=10001)], [A2])
    assert (r["fp"], r["fn"]) == (1, 1)


def test_m08_explicit_tolerance_is_inclusive():
    from auditgym.judge.matcher import is_match, parse_submission

    def check(amount):
        f = parse_submission({"episode_id": "E", "findings": [finding(affected_ids=["R1", "R2"], claimed_amount_minor=amount)]}, "E")[0]
        return is_match(f, A2, KNOWN, tolerance=1)

    assert check(10001) and not check(10002)


@pytest.mark.parametrize("amount", ["10000", True])
def test_m09_amount_must_be_a_json_integer(amount):
    assert score([finding(affected_ids=["R1", "R2"], claimed_amount_minor=amount)], [A2])["reward"] == -1.0


def test_m10_maximum_matching_beats_greedy():
    a = issue("A")
    b = dict(issue("B"), evidence_sets=[["D3"]])
    r = score([finding(evidence_ids=["D1", "D2", "D3"]), finding()], [a, b])
    assert r["tp"] == 2 and r["fp"] == 0 and sorted(r["matches"]) == [[0, "B"], [1, "A"]]


def test_m11_deterministic_tie_break():
    a, b = issue("A"), issue("B")
    first = score([finding(), finding()], [a, b])
    second = score([finding(), finding()], [b, a])
    assert first["tp"] == second["tp"] == 2
    assert sorted(first["matches"]) == sorted(second["matches"]) == [[0, "A"], [1, "B"]]


def test_m12_identical_correct_findings():
    r = score([finding(affected_ids=["R1", "R2"]), finding(affected_ids=["R1", "R2"])], [A2])
    assert (r["tp"], r["fp"], r["fn"]) == (1, 1, 0)
    assert abs(r["reward"] - 47 / 150) <= 1e-9
