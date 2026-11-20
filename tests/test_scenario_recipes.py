from collections import Counter

import pytest

from auditgym.episode_check import check_episode
from auditgym.generator import generate_episode

ONLY = {"split_payments": False, "refunds": False, "recurring_amounts": False, "unreferenced_receipts": False, "timezone_edges": False}


def only(name):
    return dict(ONLY, **{name: True})


RECIPES = {
    "D01": ({"preset": "clean"}, Counter()),
    "D02": ({"preset": "clean", "distractors": only("split_payments")}, Counter()),
    "D03": ({"preset": "clean", "distractors": only("refunds")}, Counter()),
    "D04": ({"preset": "clean", "distractors": only("recurring_amounts")}, Counter()),
    "D05": ({"preset": "clean", "distractors": only("timezone_edges"), "cross_currency_share": 0.5}, Counter()),
    "D06": ({"preset": "clean", "errors": {"duplicate_invoice": 1}}, Counter({("duplicate_invoice", "single"): 1})),
    "D09": ({"preset": "clean", "errors": {"payment_allocation": 1}, "allocation_mode": "missing"}, Counter({("payment_allocation", "single"): 1})),
    "D10": ({"preset": "clean", "errors": {"payment_allocation": 1}, "allocation_mode": "partial"}, Counter({("payment_allocation", "single"): 1})),
    "D12": ({"preset": "clean", "errors": {"fx_rate": 1}}, Counter({("fx_rate", "single"): 1})),
    "D14": ({"preset": "clean", "errors": {"period_cutoff": 1}}, Counter({("period_cutoff", "single"): 1})),
    "D16": ({"preset": "clean", "interactions": {"fx_partial_boundary": 1}},
            Counter({("payment_allocation", "fx_partial_boundary"): 1, ("fx_rate", "fx_partial_boundary"): 1})),
    "D19": ({"preset": "clean", "interactions": {"duplicate_fx": 1}},
            Counter({("duplicate_invoice", "duplicate_fx"): 1, ("fx_rate", "duplicate_fx"): 1})),
}


@pytest.mark.parametrize("name", sorted(RECIPES))
def test_readme_recipe_produces_the_scenario(name):
    config, expected = RECIPES[name]
    for seed in range(4):
        episode = generate_episode(seed, config)
        found = Counter((i["type"], i["scenario"]) for i in episode.truth["issues"])
        assert found == expected
        assert all(not problems for _, problems in check_episode(episode))


def refs_per_invoice(public):
    return Counter(r["invoice_id"] for p in public["payments"] if p["invoice_refs"] for r in p["invoice_refs"])


def test_clean_recipes_contain_their_distractor():
    seen = Counter()
    for seed in range(6):
        split = generate_episode(seed, RECIPES["D02"][0]).public
        seen["split"] += any(n >= 2 for n in refs_per_invoice(split).values())
        refund = generate_episode(seed, RECIPES["D03"][0]).public
        seen["refund"] += any(p["kind"] == "refund" for p in refund["payments"])
        repeat = generate_episode(seed, RECIPES["D04"][0]).public
        totals = Counter((d["entity_id"], d["total_minor"]) for d in repeat["invoices"])
        seen["repeat"] += max(totals.values()) >= 2
    assert seen["split"] >= 4 and seen["refund"] >= 4 and seen["repeat"] == 6


def test_allocation_modes_differ():
    for seed in range(5):
        missing = generate_episode(seed, RECIPES["D09"][0]).provenance["mutations"][0]
        partial = generate_episode(seed, RECIPES["D10"][0]).provenance["mutations"][0]
        assert missing["applied_after"] == 0
        assert 0 < partial["applied_after"] < partial["applied_before"]
