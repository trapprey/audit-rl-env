import hashlib
import random
from dataclasses import dataclass

from .. import spec
from ..config import digest, resolve
from ..verify import verify_truth
from .inject import Injector, draw_plan
from .render import render
from .world import GenerationError, build_world, check_balance, check_books


@dataclass
class Episode:
    public: dict
    truth: dict
    provenance: dict


def episode_id_for(seed, cfg_digest, presentation, secret):
    raw = f"{secret}|{seed}|{cfg_digest}|{presentation}|{spec.GENERATOR_VERSION}"
    return "EP-" + hashlib.sha256(raw.encode()).hexdigest()[:12]


def generate_episode(seed, config=None, secret=""):
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    cfg = resolve(config)
    cfg_digest = digest(cfg)
    presentation = cfg["presentation_seed"] if cfg["presentation_seed"] is not None else seed
    content_rng = random.Random(f"content|{secret}|{seed}|{cfg_digest}|{spec.GENERATOR_VERSION}")
    presentation_rng = random.Random(f"presentation|{secret}|{presentation}|{cfg_digest}|{spec.GENERATOR_VERSION}")

    world = build_world(content_rng, cfg)
    problems = check_books(world)
    if problems:
        raise GenerationError("clean books violate invariants: " + "; ".join(problems[:5]))

    interactions, singles = draw_plan(content_rng, cfg)
    issues, mutations = Injector(world, content_rng, cfg["allocation_mode"]).run(interactions, singles)
    problems = check_balance(world)
    if problems:
        raise GenerationError("books unbalanced after injection: " + "; ".join(problems[:5]))

    eid = episode_id_for(seed, cfg_digest, presentation, secret)
    meta = {
        "generator_version": spec.GENERATOR_VERSION,
        "schema_version": spec.SCHEMA_VERSION,
        "seed": seed,
        "presentation_seed": presentation,
        "config": cfg,
        "config_digest": cfg_digest,
        "plan": {"interactions": interactions, "singles": singles},
        "clean_books_checked": True,
    }
    public, truth, provenance = render(world, issues, mutations, presentation_rng, cfg, eid, meta)
    problems = verify_truth(public, truth)
    if problems:
        raise GenerationError("truth is not supported by visible evidence: " + "; ".join(problems[:5]))
    return Episode(public, truth, provenance)


__all__ = ["Episode", "GenerationError", "generate_episode"]
