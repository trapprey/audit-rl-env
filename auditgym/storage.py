import csv
import json
from pathlib import Path

from . import spec

INT_COLUMNS = {"debit_minor", "credit_minor"}


def _cell(value):
    return "null" if value is None else value


def _parse(column, text):
    if text == "null":
        return None
    if column in INT_COLUMNS:
        return int(text)
    return text


def write_csv(path, columns, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(columns)
        for row in rows:
            writer.writerow([_cell(row[c]) for c in columns])


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return [{k: _parse(k, v) for k, v in row.items()} for row in reader]


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def save_episode(episode, folder):
    folder = Path(folder)
    public_dir = folder / "public"
    grader_dir = folder / "grader"
    public_dir.mkdir(parents=True, exist_ok=True)
    grader_dir.mkdir(parents=True, exist_ok=True)
    pub = episode.public
    write_json(public_dir / "manifest.json", pub["manifest"])
    write_json(public_dir / "policy.json", pub["policy"])
    write_csv(public_dir / "ledger.csv", spec.LEDGER_COLUMNS, pub["ledger"])
    write_jsonl(public_dir / "invoices.jsonl", pub["invoices"])
    write_jsonl(public_dir / "payments.jsonl", pub["payments"])
    write_csv(public_dir / "fx.csv", spec.FX_COLUMNS, pub["fx"])
    write_json(grader_dir / "truth.json", episode.truth)
    write_json(grader_dir / "provenance.json", episode.provenance)


def load_public(folder):
    folder = Path(folder)
    return {
        "manifest": json.loads((folder / "manifest.json").read_text(encoding="utf-8")),
        "policy": json.loads((folder / "policy.json").read_text(encoding="utf-8")),
        "ledger": read_csv(folder / "ledger.csv"),
        "invoices": read_jsonl(folder / "invoices.jsonl"),
        "payments": read_jsonl(folder / "payments.jsonl"),
        "fx": read_csv(folder / "fx.csv"),
    }
