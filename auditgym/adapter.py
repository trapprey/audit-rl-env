import json
import os
import time
from pathlib import Path

from . import spec
from .env import ADAPTER_KEYS, AuditEnv, public_dir
from .generator import generate_episode as _generate
from .judge.matcher import evaluate
from .storage import load_public, save_episode
from .verify import PublicIndex

SCORE_KEYS = ("tp", "fp", "fn", "precision", "recall", "f1", "return")


def _team_config(config):
    if config is None:
        return {}
    if not isinstance(config, dict):
        raise ValueError("config must be an object")
    if "team_config" in config:
        team = config["team_config"] or {}
        if not isinstance(team, dict):
            raise ValueError("team_config must be an object")
        return team
    return {k: v for k, v in config.items() if k not in ADAPTER_KEYS}


def generate_episode(seed: int, config: dict, output_dir: str) -> None:
    episode = _generate(seed, _team_config(config), os.environ.get("AUDITGYM_SECRET", ""))
    save_episode(episode, output_dir)


def make_env():
    return AuditEnv(mode="closed")


def _truth_path(episode_dir):
    folder = Path(episode_dir)
    nested = folder / "grader" / "truth.json"
    return nested if nested.is_file() else folder / "truth.json"


def _journal(record):
    path = os.environ.get("AUDITGYM_JUDGE_LOG", "judge_log.jsonl")
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    except OSError:
        pass


def score_episode(episode_dir: str, findings: dict, inspection_steps: int) -> dict:
    truth = json.loads(_truth_path(episode_dir).read_text(encoding="utf-8"))
    known = PublicIndex(load_public(public_dir(episode_dir))).known_ids()
    steps = inspection_steps if isinstance(inspection_steps, int) and not isinstance(inspection_steps, bool) else 0
    steps = max(0, min(spec.MAX_STEPS, steps))
    result = evaluate(findings, truth, known, steps)
    _journal({
        "time": time.time(),
        "episode_id": truth["episode_id"],
        "valid": result["valid"],
        "error": result["error"],
        "inspection_steps": steps,
        "tp": result["tp"],
        "fp": result["fp"],
        "fn": result["fn"],
        "return": result["reward"],
    })
    if not result["valid"]:
        return {"tp": None, "fp": None, "fn": None, "precision": None, "recall": None, "f1": None, "return": -1.0}
    return {
        "tp": result["tp"],
        "fp": result["fp"],
        "fn": result["fn"],
        "precision": result["precision"],
        "recall": result["recall"],
        "f1": result["f1"],
        "return": result["reward"],
    }
