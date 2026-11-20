import pytest

from auditgym.env import AuditEnv
from auditgym.judge import Judge


@pytest.fixture
def make_env():
    def factory(mode="train", allow_oracle=False, **kwargs):
        return AuditEnv(judge=Judge(mode, allow_oracle=allow_oracle), mode=mode, **kwargs)

    return factory


def submit(episode_id, findings):
    return {"name": "submit", "args": {"findings": {"episode_id": episode_id, "findings": findings}}}


def walk_keys(value):
    if isinstance(value, dict):
        for key, inner in value.items():
            yield key
            yield from walk_keys(inner)
    elif isinstance(value, list):
        for inner in value:
            yield from walk_keys(inner)
