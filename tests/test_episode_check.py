import copy

import pytest

from auditgym.config import preset_names
from auditgym.episode_check import check_episode
from auditgym.generator import Episode, generate_episode


def failing(episode):
    return {name: problems for name, problems in check_episode(episode) if problems}


@pytest.mark.parametrize("preset", preset_names())
def test_generated_episodes_pass_every_check(preset):
    for seed in range(6):
        assert failing(generate_episode(seed, {"preset": preset})) == {}


def tampered(mutate, seed=3, preset="heavy"):
    episode = generate_episode(seed, {"preset": preset})
    public, truth, provenance = (copy.deepcopy(x) for x in (episode.public, episode.truth, episode.provenance))
    mutate(public, truth, provenance)
    return Episode(public, truth, provenance)


def test_hidden_undeclared_error_is_caught():
    def drop_issue(public, truth, provenance):
        truth["issues"].pop(0)

    problems = failing(tampered(drop_issue))
    assert "books differ from policy exactly by declared issues" in problems


def test_invented_issue_is_caught():
    def invent(public, truth, provenance):
        fake = copy.deepcopy(truth["issues"][0])
        fake["issue_id"] = "ISS-99"
        fake["expected_amount_minor"] += 7
        truth["issues"].append(fake)

    problems = failing(tampered(invent))
    assert "books differ from policy exactly by declared issues" in problems
    assert "evidence is visible and sufficient" in problems


def test_unbalanced_ledger_is_caught():
    def unbalance(public, truth, provenance):
        public["ledger"][0]["debit_minor" if public["ledger"][0]["debit_minor"] else "credit_minor"] += 1

    assert "bookkeeping equalities" in failing(tampered(unbalance))


def test_dangling_reference_and_count_mismatch_are_caught():
    def dangle(public, truth, provenance):
        public["ledger"][5]["invoice_id"] = "NOPE"
        public["invoices"].pop()

    problems = failing(tampered(dangle))
    assert "identifiers and references" in problems
    assert "record counts and episode id" in problems


def test_provenance_must_match_truth():
    def forget(public, truth, provenance):
        provenance["mutations"].pop()

    assert "provenance matches truth" in failing(tampered(forget))


def test_schema_violation_is_caught():
    def break_time(public, truth, provenance):
        public["invoices"][0]["issued_at"] = "yesterday"

    assert "schemas" in failing(tampered(break_time, preset="clean"))
