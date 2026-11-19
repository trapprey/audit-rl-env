import json
import random
from pathlib import Path

from .generator import Episode
from .storage import load_public, save_episode

NAMESPACES = {
    "row_id": ("ledger", "R"),
    "doc_id": (("invoices", "payments"), "D"),
    "invoice_id": ("invoices", "I"),
    "payment_id": ("payments", "P"),
    "entity_id": (("invoices", "payments"), "E"),
}


def load_episode(folder):
    folder = Path(folder)
    public = load_public(folder / "public")
    truth = json.loads((folder / "grader" / "truth.json").read_text())
    provenance_path = folder / "grader" / "provenance.json"
    provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else {}
    return Episode(public, truth, provenance)


def _relabel(value, mapping):
    if isinstance(value, str):
        return mapping.get(value, value)
    if isinstance(value, list):
        return [_relabel(v, mapping) for v in value]
    if isinstance(value, dict):
        return {k: _relabel(v, mapping) for k, v in value.items()}
    return value


def transform(episode, seed):
    rng = random.Random(f"transform|{seed}")
    public = episode.public
    used = set()
    mapping = {}

    def fresh(prefix):
        while True:
            value = prefix + "".join(rng.choice("0123456789abcdef") for _ in range(10))
            if value not in used:
                used.add(value)
                return value

    for key, (tables, prefix) in NAMESPACES.items():
        tables = (tables,) if isinstance(tables, str) else tables
        values = sorted({record[key] for t in tables for record in public[t]})
        rng.shuffle(values)
        for value in values:
            mapping.setdefault(value, fresh(prefix))
    old_id = public["manifest"]["episode_id"]
    mapping[old_id] = "EP-" + "".join(rng.choice("0123456789abcdef") for _ in range(12))

    new_public = _relabel(public, mapping)
    for table in ("ledger", "invoices", "payments", "fx"):
        rng.shuffle(new_public[table])
    truth = _relabel(episode.truth, mapping)
    for issue in truth["issues"]:
        issue["affected_ids"] = sorted(issue["affected_ids"])
        issue["evidence_sets"] = [sorted(group) for group in issue["evidence_sets"]]
    provenance = _relabel(episode.provenance, mapping)
    provenance["transform"] = {"seed": seed, "source_episode_id": old_id, "kind": "shuffle rows and relabel ids"}
    return Episode(new_public, truth, provenance)


def transform_folder(source, target, seed):
    episode = transform(load_episode(source), seed)
    save_episode(episode, target)
    return episode
