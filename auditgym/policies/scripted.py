from .. import spec
from ..reconcile import Books, audit
from .common import TableReader, act


class ScriptedAuditor:
    name = "scripted_auditor"
    order = ("invoices", "payments", "ledger", "fx")

    def reset(self, observation, info, env=None):
        self.manifest = observation["manifest"]
        self.policy = observation["policy"]
        self.reader = TableReader(self.order)

    def act(self, observation):
        self.reader.absorb(observation)
        if not self.reader.done and observation["steps_remaining"] > 1:
            return self.reader.next_action()
        books = Books(self.manifest, self.policy, self.reader.data)
        return act("submit", findings={"episode_id": self.manifest["episode_id"], "findings": audit(books)[: spec.MAX_FINDINGS]})
