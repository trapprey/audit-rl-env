import json
from pathlib import Path

from .. import spec
from ..generator import generate_episode
from ..schema import validate
from ..verify import PublicIndex, verify_truth
from .matcher import evaluate


class JudgeError(RuntimeError):
    pass


class Judge:
    def __init__(self, mode="train", allow_oracle=False, secret="", log_path=None):
        if mode not in ("train", "eval"):
            raise ValueError("mode must be train or eval")
        self.mode = mode
        self.allow_oracle = allow_oracle
        self.secret = secret
        self.log_path = Path(log_path) if log_path else None
        self._episodes = {}
        self._results = []

    def reset(self, seed, config=None):
        if isinstance(config, dict) and "episode_dir" in config:
            episode = self._load(config)
        else:
            try:
                episode = generate_episode(seed, config, self.secret)
            except ValueError as exc:
                raise JudgeError(f"bad reset request: {exc}") from None
        token = f"T{len(self._episodes) + 1:06d}"
        self._episodes[token] = {
            "episode": episode,
            "known": PublicIndex(episode.public).known_ids(),
            "graded": False,
            "seed": seed,
        }
        return {"token": token, "public": episode.public}

    def _load(self, config):
        from ..transform import load_episode

        if set(config) != {"episode_dir"} or not isinstance(config["episode_dir"], str):
            raise JudgeError("episode_dir must be the only config key")
        try:
            episode = load_episode(config["episode_dir"])
        except (OSError, ValueError, KeyError) as exc:
            raise JudgeError(f"cannot load episode: {type(exc).__name__}") from None
        problems = validate(episode.public["manifest"], "manifest") + validate(episode.truth, "truth")
        if not problems:
            problems = verify_truth(episode.public, episode.truth)
        if problems or episode.truth["episode_id"] != episode.public["manifest"]["episode_id"]:
            raise JudgeError("episode files are inconsistent")
        return episode

    def _get(self, token):
        entry = self._episodes.get(token)
        if entry is None:
            raise JudgeError("unknown episode")
        return entry

    def grade(self, token, payload, inspection_steps, truncated=False):
        entry = self._get(token)
        if entry["graded"]:
            raise JudgeError("episode already graded")
        if isinstance(inspection_steps, bool) or not isinstance(inspection_steps, int):
            raise JudgeError("inspection_steps must be an integer")
        if not 0 <= inspection_steps <= spec.MAX_STEPS:
            raise JudgeError("inspection_steps out of range")
        entry["graded"] = True
        episode = entry["episode"]
        result = evaluate(payload, episode.truth, entry["known"], inspection_steps, truncated)
        record = {"token": token, "episode_id": episode.truth["episode_id"], **result}
        self._results.append(record)
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a") as fh:
                fh.write(json.dumps(record, sort_keys=True) + "\n")
        if self.mode == "train":
            metrics = {k: result[k] for k in ("valid", "error", "precision", "recall", "f1", "tp", "fp", "fn", "inspection_steps")}
            return {"reward": result["reward"], "metrics": metrics}
        return {"accepted": True}

    def oracle(self, token):
        if not self.allow_oracle:
            raise JudgeError("oracle is disabled")
        truth = self._get(token)["episode"].truth
        return {
            "episode_id": truth["episode_id"],
            "findings": [
                {
                    "type": issue["type"],
                    "affected_ids": list(issue["affected_ids"]),
                    "claimed_amount_minor": issue["expected_amount_minor"],
                    "currency": issue["currency"],
                    "evidence_ids": list(issue["evidence_sets"][0]),
                    "explanation": "reference finding",
                }
                for issue in truth["issues"]
            ],
        }

    def results(self):
        return [dict(r) for r in self._results]

    def close(self):
        pass
