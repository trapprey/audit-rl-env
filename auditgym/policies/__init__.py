from .baselines import FlagEverything, NaiveReconciler, Oracle, RandomExplorer, Silent
from .scripted import ScriptedAuditor

BASELINES = {
    "random_explorer": lambda: RandomExplorer(seed=7),
    "naive_reconciler": NaiveReconciler,
    "flag_all": FlagEverything,
    "oracle": Oracle,
}

POLICIES = {**BASELINES, "scripted_auditor": ScriptedAuditor, "silent": Silent}


def make_policy(name):
    if name not in POLICIES:
        raise KeyError(f"unknown policy {name!r}")
    return POLICIES[name]()


__all__ = ["BASELINES", "POLICIES", "make_policy"]
