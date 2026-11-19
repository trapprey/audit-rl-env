import argparse
import json
import os
import sys

from ..generator import GenerationError
from .core import Judge, JudgeError


def handle(judge, request):
    op = request.get("op")
    if op == "reset":
        return judge.reset(request.get("seed"), request.get("config"))
    if op == "grade":
        return judge.grade(request.get("token"), request.get("payload"), request.get("inspection_steps"), bool(request.get("truncated")))
    if op == "oracle":
        return judge.oracle(request.get("token"))
    if op == "results":
        return judge.results()
    if op == "ping":
        return {"mode": judge.mode}
    raise JudgeError("unknown operation")


def serve(judge, stdin, stdout):
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise JudgeError("request must be an object")
            if request.get("op") == "close":
                stdout.write(json.dumps({"ok": True, "result": None}) + "\n")
                stdout.flush()
                return
            reply = {"ok": True, "result": handle(judge, request)}
        except (JudgeError, GenerationError, ValueError) as exc:
            reply = {"ok": False, "error": str(exc)[:300]}
        stdout.write(json.dumps(reply, separators=(",", ":")) + "\n")
        stdout.flush()


def main(argv=None):
    parser = argparse.ArgumentParser(prog="auditgym.judge")
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--allow-oracle", action="store_true")
    parser.add_argument("--log")
    args = parser.parse_args(argv)
    secret = os.environ.pop("AUDITGYM_SECRET", "")
    judge = Judge(args.mode, args.allow_oracle, secret, args.log)
    serve(judge, sys.stdin, sys.stdout)


if __name__ == "__main__":
    main()
