import json
import os
import subprocess
import sys
from pathlib import Path

from .core import JudgeError

PACKAGE_ROOT = str(Path(__file__).resolve().parents[2])


class JudgeClient:
    def __init__(self, mode="train", allow_oracle=False, secret=None, log_path=None):
        self.mode = mode
        command = [sys.executable, "-m", "auditgym.judge", "--mode", mode]
        if allow_oracle:
            command.append("--allow-oracle")
        if log_path:
            command += ["--log", str(log_path)]
        env = dict(os.environ)
        env["PYTHONPATH"] = PACKAGE_ROOT + os.pathsep + env.get("PYTHONPATH", "")
        env.pop("AUDITGYM_SECRET", None)
        if secret:
            env["AUDITGYM_SECRET"] = secret
        self._proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            env=env,
            bufsize=1,
        )

    def _call(self, op, **payload):
        if self._proc is None or self._proc.poll() is not None:
            raise JudgeError("judge process is not running")
        self._proc.stdin.write(json.dumps({"op": op, **payload}, separators=(",", ":")) + "\n")
        self._proc.stdin.flush()
        line = self._proc.stdout.readline()
        if not line:
            raise JudgeError("judge process stopped")
        reply = json.loads(line)
        if not reply.get("ok"):
            raise JudgeError(reply.get("error", "judge error"))
        return reply["result"]

    def reset(self, seed, config=None):
        return self._call("reset", seed=seed, config=config)

    def grade(self, token, payload, inspection_steps, truncated=False):
        return self._call("grade", token=token, payload=payload, inspection_steps=inspection_steps, truncated=truncated)

    def oracle(self, token):
        return self._call("oracle", token=token)

    def results(self):
        return self._call("results")

    def close(self):
        if self._proc is None:
            return
        try:
            if self._proc.poll() is None:
                self._proc.stdin.write(json.dumps({"op": "close"}) + "\n")
                self._proc.stdin.flush()
                self._proc.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            self._proc.kill()
        finally:
            for stream in (self._proc.stdin, self._proc.stdout):
                try:
                    stream.close()
                except OSError:
                    pass
            self._proc = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
