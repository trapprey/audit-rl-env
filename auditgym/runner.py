import json
import statistics
import time
from pathlib import Path

from .env import AuditEnv
from .judge.client import JudgeClient
from .policies import make_policy


class TrajectoryWriter:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w")

    def write(self, record):
        self._fh.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")

    def close(self):
        self._fh.close()


def run_episode(env, policy, seed, config, sink=None):
    started = time.perf_counter()
    observation, info = env.reset(seed, config)
    policy.reset(observation, info, env)
    episode_id = info["episode_id"]

    def log(step, action, obs, reward, terminated, truncated):
        if sink is not None:
            sink.write({
                "episode_id": episode_id,
                "policy": policy.name,
                "step": step,
                "action": action,
                "observation": obs,
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
            })

    log(0, None, observation, 0.0, False, False)
    total = 0.0
    step = 0
    terminated = truncated = False
    while not (terminated or truncated):
        action = policy.act(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        step += 1
        total += reward
        log(step, action, observation, reward, terminated, truncated)
    return {
        "episode_id": episode_id,
        "policy": policy.name,
        "seed": seed,
        "steps": step,
        "env_return": total,
        "terminated": terminated,
        "truncated": truncated,
        "seconds": time.perf_counter() - started,
        "train_metrics": info.get("metrics"),
    }


def run_policy(policy_name, episodes, mode="eval", sink=None, secret=None, log_path=None):
    judge = JudgeClient(mode=mode, allow_oracle=policy_name == "oracle", secret=secret, log_path=log_path)
    env = AuditEnv(judge=judge, mode=mode)
    try:
        summaries = []
        for seed, config in episodes:
            policy = make_policy(policy_name)
            summaries.append(run_episode(env, policy, seed, config, sink))
        graded = judge.results()
    finally:
        judge.close()
    for summary, result in zip(summaries, graded):
        if summary["episode_id"] != result["episode_id"]:
            raise RuntimeError("judge results are out of order")
        summary["judge"] = {k: v for k, v in result.items() if k not in ("token", "matches")}
    return summaries


def aggregate(summaries):
    def mean(key):
        return statistics.fmean(s["judge"][key] for s in summaries)

    return {
        "episodes": len(summaries),
        "precision": mean("precision"),
        "recall": mean("recall"),
        "f1": mean("f1"),
        "return": mean("reward"),
        "steps": statistics.fmean(s["steps"] for s in summaries),
        "seconds": statistics.fmean(s["seconds"] for s in summaries),
        "invalid": sum(1 for s in summaries if not s["judge"]["valid"]),
        "truncated": sum(1 for s in summaries if s["truncated"]),
    }


def by_type(summaries):
    totals = {}
    for summary in summaries:
        for kind, counts in summary["judge"]["by_type"].items():
            bucket = totals.setdefault(kind, {"tp": 0, "fp": 0, "fn": 0})
            for key in bucket:
                bucket[key] += counts[key]
    return totals
