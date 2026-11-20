import copy
import json
import secrets
import time
from pathlib import Path

from . import spec
from .calc import CalcError, evaluate as calculate
from .judge.matcher import SubmissionError, parse_submission
from .money import fx_id

TASK = (
    "You are auditing the receivables books of one fictional company for one reporting period. "
    "Read records with the actions, then call submit once with findings for real discrepancies. "
    "Every finding needs the canonical affected_ids, the amount and currency defined in policy.findings, "
    "and evidence_ids that prove it. Books can be clean; false findings lower the reward. "
    "Actions are {\"name\": ..., \"args\": {...}}: "
    "list_records(table, cursor, limit<=50) with table in ledger, invoices, payments, fx and the first cursor null; "
    "get_record(table, id) where id is row_id, invoice_id, payment_id or fx_id; get_document(doc_id); "
    "calculator(expression) with numbers, + - * / and parentheses; "
    "submit(findings) with findings = {episode_id, findings: [{type, affected_ids, claimed_amount_minor, currency, "
    "evidence_ids, explanation}]}, at most 30 findings, 10 evidence ids and 1000 explanation characters each. "
    "Every action costs one of 100 steps, submit included."
)

TABLE_KEYS = {"ledger": "row_id", "invoices": "invoice_id", "payments": "payment_id", "fx": "fx_id"}
ACTION_ARGS = {
    "list_records": ({"table"}, {"table", "cursor", "limit"}),
    "get_record": ({"table", "id"}, {"table", "id"}),
    "get_document": ({"doc_id"}, {"doc_id"}),
    "calculator": ({"expression"}, {"expression"}),
    "submit": (set(), {"findings"}),
}
ADAPTER_KEYS = {"mode", "episode_dir", "team_config"}
CLOSED_MODES = {"closed", "eval"}
OPEN_MODES = {"train", "open"}
MAX_ACTION_BYTES = 200_000


def _plain_json(value):
    try:
        text = json.dumps(value, allow_nan=False)
    except (TypeError, ValueError, RecursionError):
        return None, "submission is not valid JSON"
    if len(text) > MAX_ACTION_BYTES:
        return None, "submission is too large"
    return json.loads(text), None


def _error(code, message):
    return {"code": code, "message": message[:200]}


def _mode(value):
    if value in CLOSED_MODES:
        return "eval"
    if value in OPEN_MODES:
        return "train"
    raise ValueError("mode must be closed or train")


def public_dir(episode_dir):
    folder = Path(episode_dir)
    return folder / "public" if (folder / "public").is_dir() else folder


class AuditEnv:
    metadata = {"render_modes": []}

    def __init__(self, judge=None, mode="train", max_steps=spec.MAX_STEPS, time_limit=60.0):
        self.mode = _mode(mode)
        self._judge = judge
        self._owns_judge = judge is None
        self.max_steps = max_steps
        self.time_limit = time_limit
        self._token = None
        self._local = False
        self._ready = False
        self._done = True
        self._terminated = False
        self._truncated = False
        self._steps = 0
        self._tables = {}
        self._index = {}
        self._docs = {}
        self.episode_mode = self.mode
        self.episode_id = None
        self.submission = None

    @property
    def judge(self):
        if self._judge is None:
            from .judge.client import JudgeClient

            self._judge = JudgeClient(mode="train")
        return self._judge

    def reset(self, seed=0, config=None, options=None):
        config = options if config is None else config
        config = {} if config is None else config
        if not isinstance(config, dict):
            raise ValueError("config must be an object")
        if seed is None:
            seed = secrets.randbelow(2**31)
        adapter = bool(ADAPTER_KEYS & set(config))
        self.episode_mode = _mode(config.get("mode", "closed" if self.mode == "eval" else "train")) if adapter else self.mode
        episode_dir = config.get("episode_dir") if adapter else None
        if adapter:
            team = config.get("team_config") or {}
            if not isinstance(team, dict):
                raise ValueError("team_config must be an object")
        if episode_dir is not None and self.episode_mode == "eval":
            from .storage import load_public

            public = load_public(public_dir(episode_dir))
            self._token = None
            self._local = True
        else:
            if episode_dir is not None:
                request = {"episode_dir": str(episode_dir)}
            elif adapter:
                request = team
            else:
                request = config
            reply = self.judge.reset(seed, request)
            public = reply["public"]
            self._token = reply["token"]
            self._local = False
        self._load(public)
        observation = {
            "task": TASK,
            "manifest": copy.deepcopy(public["manifest"]),
            "policy": copy.deepcopy(public["policy"]),
            "remaining_steps": self.max_steps,
        }
        return observation, {"episode_id": self.episode_id, "mode": "closed" if self.episode_mode == "eval" else "train"}

    def _load(self, public):
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
        self._index = {table: {row[TABLE_KEYS[table]]: row for row in rows} for table, rows in self._tables.items()}
        self._docs = {d["doc_id"]: d for d in public["invoices"] + public["payments"]}
        self._steps = 0
        self._ready = True
        self._done = False
        self._terminated = False
        self._truncated = False
        self.submission = None
        self._started = time.perf_counter()

    def step(self, action):
        if not self._ready:
            raise RuntimeError("reset() must be called before step()")
        if self._done:
            return self._observation(None, _error("episode_finished", "the episode has ended")), 0.0, self._terminated, self._truncated, {}
        self._steps += 1
        if time.perf_counter() - self._started >= self.time_limit:
            return self._truncate(None, _error("time_limit", "the episode time limit was reached"))
        parsed, problem = self._parse(action)
        if problem is not None:
            result, error = None, problem
        elif parsed[0] == "submit":
            return self._submit(parsed[1])
        else:
            result, error = self._run(*parsed)
        if self._steps >= self.max_steps:
            return self._truncate(result, error)
        return self._observation(result, error), 0.0, False, False, {}

    def oracle_findings(self):
        if self._local:
            raise RuntimeError("oracle is not available in closed mode")
        return self.judge.oracle(self._token)

    def close(self):
        if self._owns_judge and self._judge is not None and hasattr(self._judge, "close"):
            self._judge.close()
            self._judge = None

    def _observation(self, result, error):
        return {"result": result, "error": error, "remaining_steps": self.max_steps - self._steps}

    def _graded(self, payload, steps, truncated):
        if self._local:
            return 0.0, {}
        reply = self.judge.grade(self._token, payload, steps, truncated)
        if self.episode_mode != "train":
            return 0.0, {}
        return float(reply.get("reward", 0.0)), {"metrics": reply.get("metrics")}

    def _submit(self, args):
        self._done = True
        self._terminated = True
        inspection = self._steps - 1
        payload, problem = _plain_json(args.get("findings"))
        if problem is None:
            try:
                parse_submission(payload, self.episode_id)
            except SubmissionError as exc:
                problem = str(exc)
        self.submission = {"findings": payload, "inspection_steps": inspection, "valid": problem is None}
        reward, info = self._graded(payload if payload is not None else problem, inspection, False)
        if problem is not None:
            return self._observation(None, _error("invalid_submission", problem)), reward, True, False, info
        return self._observation({"accepted": True}, None), reward, True, False, info

    def _truncate(self, result, error):
        self._done = True
        self._truncated = True
        reward, info = self._graded(None, min(self._steps, self.max_steps), True)
        return self._observation(result, error), reward, False, True, info

    def _parse(self, action):
        if isinstance(action, (str, bytes)):
            if len(action) > MAX_ACTION_BYTES:
                return None, _error("invalid_action", "action is too large")
            try:
                action = json.loads(action)
            except (ValueError, UnicodeDecodeError, RecursionError):
                return None, _error("invalid_action", "action is not valid JSON")
        if not isinstance(action, dict):
            return None, _error("invalid_action", "action must be an object with name and args")
        key = "name" if "name" in action else "tool"
        if key not in action or not set(action) <= {key, "args"}:
            return None, _error("invalid_action", "action must be an object with name and args")
        name = action[key]
        args = action.get("args", {})
        if not isinstance(name, str) or name not in ACTION_ARGS:
            return None, _error("unknown_action", "unknown action")
        if not isinstance(args, dict):
            return None, _error("bad_arguments", "args must be an object")
        required, allowed = ACTION_ARGS[name]
        if not required <= set(args) or not set(args) <= allowed:
            return None, _error("bad_arguments", f"{name} takes {sorted(allowed)}")
        return (name, args), None

    def _run(self, name, args):
        if name == "list_records":
            return self._list(args)
        if name == "get_record":
            table, key = args["table"], args["id"]
            if not isinstance(table, str) or table not in self._index:
                return None, _error("bad_arguments", "unknown table")
            if not isinstance(key, str) or key not in self._index[table]:
                return None, _error("not_found", "record not found")
            return copy.deepcopy(self._index[table][key]), None
        if name == "get_document":
            key = args["doc_id"]
            if not isinstance(key, str) or key not in self._docs:
                return None, _error("not_found", "document not found")
            return copy.deepcopy(self._docs[key]), None
        try:
            return calculate(args["expression"]), None
        except CalcError as exc:
            return None, _error("calculator_error", str(exc))

    def _list(self, args):
        table = args["table"]
        if not isinstance(table, str) or table not in self._tables:
            return None, _error("bad_arguments", "unknown table")
        limit = args.get("limit", spec.PAGE_LIMIT)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= spec.PAGE_LIMIT:
            return None, _error("bad_arguments", f"limit must be 1..{spec.PAGE_LIMIT}")
        rows = self._tables[table]
        offset = 0
        cursor = args.get("cursor")
        if cursor is not None:
            prefix = f"{table}@"
            digits = cursor[len(prefix):] if isinstance(cursor, str) and cursor.startswith(prefix) else ""
            if not (digits.isascii() and digits.isdigit() and len(digits) <= 9):
                return None, _error("bad_cursor", "cursor is not valid for this table")
            offset = int(digits)
            if offset > len(rows):
                return None, _error("bad_cursor", "cursor is not valid for this table")
        page = rows[offset:offset + limit]
        end = offset + len(page)
        return {"records": copy.deepcopy(page), "next_cursor": f"{table}@{end}" if end < len(rows) else None}, None
