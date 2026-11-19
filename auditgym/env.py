import copy
import json
import secrets
import time

from . import spec
from .calc import CalcError, evaluate as calculate
from .money import fx_id

TASK = (
    "You are auditing the receivables books of one fictional company for one reporting period. "
    "Read records with the tools, then call submit once with findings for real discrepancies. "
    "Every finding needs the canonical affected_ids, the amount and currency defined in policy.findings, "
    "and evidence_ids that prove it. Books can be clean; false findings lower the reward. "
    "Each tool call costs one step, submit included."
)

TOOLS = {
    "list_records": {
        "args": {"table": list(spec.TABLES), "cursor": "string or null", "limit": f"1..{spec.PAGE_LIMIT}"},
        "returns": "records of one page and next_cursor (null on the last page)",
    },
    "get_record": {
        "args": {"table": list(spec.TABLES), "id": "row_id, invoice_id, payment_id or fx_id"},
        "returns": "one record",
    },
    "get_document": {"args": {"doc_id": "doc_id of an invoice or payment"}, "returns": "one document"},
    "calculator": {"args": {"expression": "decimal arithmetic with + - * / and parentheses"}, "returns": "decimal string"},
    "submit": {
        "args": {"findings": {"episode_id": "string", "findings": [{k: "..." for k in spec.FINDING_KEYS}]}},
        "returns": "ends the episode",
        "limits": {
            "findings": spec.MAX_FINDINGS,
            "evidence_ids": spec.MAX_EVIDENCE,
            "affected_ids": spec.MAX_AFFECTED,
            "explanation_chars": spec.MAX_EXPLANATION,
        },
    },
}

TABLE_KEYS = {"ledger": "row_id", "invoices": "invoice_id", "payments": "payment_id", "fx": "fx_id"}
TOOL_ARGS = {
    "list_records": ({"table"}, {"table", "cursor", "limit"}),
    "get_record": ({"table", "id"}, {"table", "id"}),
    "get_document": ({"doc_id"}, {"doc_id"}),
    "calculator": ({"expression"}, {"expression"}),
    "submit": (set(), {"findings"}),
}
MAX_ACTION_BYTES = 200_000


def _plain_json(value):
    try:
        text = json.dumps(value, allow_nan=False)
    except (TypeError, ValueError, RecursionError):
        return "unserializable submission"
    if len(text) > MAX_ACTION_BYTES:
        return "submission too large"
    return json.loads(text)


def _error(code, message):
    return {"error": {"code": code, "message": message}}


class AuditEnv:
    metadata = {"render_modes": []}

    def __init__(self, judge=None, mode="train", max_steps=spec.MAX_STEPS, time_limit=60.0):
        if mode not in ("train", "eval"):
            raise ValueError("mode must be train or eval")
        self._owns_judge = judge is None
        if judge is None:
            from .judge.client import JudgeClient

            judge = JudgeClient(mode=mode)
        self.judge = judge
        self.mode = mode
        self.max_steps = max_steps
        self.time_limit = time_limit
        self._token = None
        self._done = True
        self._terminated = False
        self._truncated = False
        self._steps = 0
        self._tables = {}
        self._index = {}
        self._docs = {}
        self.episode_id = None

    def reset(self, seed=0, config=None, options=None):
        if config is None:
            config = options
        if seed is None:
            seed = secrets.randbelow(2**31)
        reply = self.judge.reset(seed, config or {})
        public = reply["public"]
        self._token = reply["token"]
        self.episode_id = public["manifest"]["episode_id"]
        fx_rows = [
            {"fx_id": fx_id(r["date"], r["source_currency"], r["target_currency"]), **r} for r in public["fx"]
        ]
        self._tables = {
            "ledger": public["ledger"],
            "invoices": public["invoices"],
            "payments": public["payments"],
            "fx": fx_rows,
        }
        self._index = {
            table: {row[TABLE_KEYS[table]]: row for row in rows} for table, rows in self._tables.items()
        }
        self._docs = {d["doc_id"]: d for d in public["invoices"] + public["payments"]}
        self._steps = 0
        self._done = False
        self._terminated = False
        self._truncated = False
        self._started = time.perf_counter()
        observation = {
            "step": 0,
            "steps_remaining": self.max_steps,
            "task": TASK,
            "manifest": copy.deepcopy(public["manifest"]),
            "policy": copy.deepcopy(public["policy"]),
            "tools": copy.deepcopy(TOOLS),
        }
        return observation, {"episode_id": self.episode_id, "mode": self.mode, "steps_remaining": self.max_steps}

    def step(self, action):
        if self._token is None:
            raise RuntimeError("reset() must be called before step()")
        if self._done:
            return self._observation(_error("episode_finished", "the episode has ended")), 0.0, self._terminated, self._truncated, self._info()
        self._steps += 1
        if time.perf_counter() - self._started >= self.time_limit:
            return self._truncate(_error("time_limit", "the episode time limit was reached"))
        parsed, problem = self._parse(action)
        if problem is not None:
            payload = problem
        elif parsed[0] == "submit":
            return self._submit(parsed[1])
        else:
            payload = self._run(*parsed)
        if self._steps >= self.max_steps:
            return self._truncate(payload)
        return self._observation(payload), 0.0, False, False, self._info()

    def oracle_findings(self):
        return self.judge.oracle(self._token)

    def close(self):
        if self._owns_judge and hasattr(self.judge, "close"):
            self.judge.close()

    def _observation(self, payload):
        return {"step": self._steps, "steps_remaining": self.max_steps - self._steps, **payload}

    def _info(self, extra=None):
        info = {"steps_remaining": self.max_steps - self._steps}
        if extra and self.mode == "train":
            info.update(extra)
        return info

    def _submit(self, args):
        self._done = True
        self._terminated = True
        reply = self.judge.grade(self._token, _plain_json(args.get("findings")), self._steps - 1, False)
        reward = float(reply.get("reward", 0.0)) if self.mode == "train" else 0.0
        extra = {"metrics": reply.get("metrics")} if self.mode == "train" else None
        return self._observation({"result": {"submitted": True}}), reward, True, False, self._info(extra)

    def _truncate(self, payload):
        self._done = True
        self._truncated = True
        reply = self.judge.grade(self._token, None, min(self._steps, self.max_steps), True)
        reward = float(reply.get("reward", 0.0)) if self.mode == "train" else 0.0
        extra = {"metrics": reply.get("metrics")} if self.mode == "train" else None
        return self._observation(payload), reward, False, True, self._info(extra)

    def _parse(self, action):
        if isinstance(action, (str, bytes)):
            if len(action) > MAX_ACTION_BYTES:
                return None, _error("invalid_action", "action is too large")
            try:
                action = json.loads(action)
            except (ValueError, UnicodeDecodeError, RecursionError):
                return None, _error("invalid_action", "action is not valid JSON")
        if not isinstance(action, dict) or not set(action) <= {"tool", "args"} or "tool" not in action:
            return None, _error("invalid_action", "action must be an object with tool and args")
        tool = action["tool"]
        args = action.get("args", {})
        if not isinstance(tool, str) or tool not in TOOL_ARGS:
            return None, _error("unknown_tool", "unknown tool")
        if not isinstance(args, dict):
            return None, _error("bad_arguments", "args must be an object")
        required, allowed = TOOL_ARGS[tool]
        if not required <= set(args) or not set(args) <= allowed:
            return None, _error("bad_arguments", f"{tool} takes {sorted(allowed)}")
        return (tool, args), None

    def _run(self, tool, args):
        if tool == "list_records":
            return self._list(args)
        if tool == "get_record":
            table, key = args["table"], args["id"]
            if not isinstance(table, str) or table not in self._index:
                return _error("bad_arguments", "unknown table")
            if not isinstance(key, str) or key not in self._index[table]:
                return _error("not_found", "record not found")
            return {"result": {"table": table, "record": copy.deepcopy(self._index[table][key])}}
        if tool == "get_document":
            key = args["doc_id"]
            if not isinstance(key, str) or key not in self._docs:
                return _error("not_found", "document not found")
            return {"result": {"document": copy.deepcopy(self._docs[key])}}
        try:
            return {"result": {"value": calculate(args["expression"])}}
        except CalcError as exc:
            return _error("calculator_error", str(exc))

    def _list(self, args):
        table = args["table"]
        if not isinstance(table, str) or table not in self._tables:
            return _error("bad_arguments", "unknown table")
        limit = args.get("limit", spec.PAGE_LIMIT)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= spec.PAGE_LIMIT:
            return _error("bad_arguments", f"limit must be 1..{spec.PAGE_LIMIT}")
        rows = self._tables[table]
        offset = 0
        cursor = args.get("cursor")
        if cursor is not None:
            prefix = f"{table}@"
            digits = cursor[len(prefix):] if isinstance(cursor, str) and cursor.startswith(prefix) else ""
            if not (digits.isascii() and digits.isdigit() and len(digits) <= 9):
                return _error("bad_cursor", "cursor is not valid for this table")
            offset = int(digits)
            if offset > len(rows):
                return _error("bad_cursor", "cursor is not valid for this table")
        page = rows[offset:offset + limit]
        end = offset + len(page)
        return {
            "result": {
                "table": table,
                "records": copy.deepcopy(page),
                "next_cursor": f"{table}@{end}" if end < len(rows) else None,
                "total": len(rows),
            }
        }
