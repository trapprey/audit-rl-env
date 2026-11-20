from dataclasses import dataclass

from .. import spec


class SubmissionError(ValueError):
    pass


@dataclass(frozen=True)
class Finding:
    index: int
    type: str
    affected: frozenset
    amount: int
    currency: str
    evidence: frozenset


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _id_list(value, name, index, limit=None):
    if not isinstance(value, list):
        raise SubmissionError(f"finding {index} {name} must be a list")
    if limit is not None and len(value) > limit:
        raise SubmissionError(f"finding {index} {name} has more than {limit} entries")
    for item in value:
        if not isinstance(item, str):
            raise SubmissionError(f"finding {index} {name} entries must be strings")
    return value


def parse_submission(payload, episode_id):
    if not isinstance(payload, dict):
        raise SubmissionError("submission must be an object")
    missing = [k for k in spec.SUBMISSION_KEYS if k not in payload]
    if missing:
        raise SubmissionError(f"submission is missing {missing}")
    if not isinstance(payload["episode_id"], str) or payload["episode_id"] != episode_id:
        raise SubmissionError("episode_id does not match")
    raw = payload["findings"]
    if not isinstance(raw, list):
        raise SubmissionError("findings must be a list")
    if len(raw) > spec.MAX_FINDINGS:
        raise SubmissionError(f"more than {spec.MAX_FINDINGS} findings")
    findings = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise SubmissionError(f"finding {index} must be an object")
        missing = [k for k in spec.FINDING_KEYS if k not in item]
        if missing:
            raise SubmissionError(f"finding {index} is missing {missing}")
        if not isinstance(item["type"], str):
            raise SubmissionError(f"finding {index} type must be a string")
        affected = _id_list(item["affected_ids"], "affected_ids", index)
        evidence = _id_list(item["evidence_ids"], "evidence_ids", index, spec.MAX_EVIDENCE)
        amount = item["claimed_amount_minor"]
        if not _is_int(amount):
            raise SubmissionError(f"finding {index} claimed_amount_minor must be an integer")
        if not isinstance(item["currency"], str):
            raise SubmissionError(f"finding {index} currency must be a string")
        explanation = item["explanation"]
        if not isinstance(explanation, str) or len(explanation) > spec.MAX_EXPLANATION:
            raise SubmissionError(f"finding {index} explanation must be a string up to {spec.MAX_EXPLANATION} characters")
        findings.append(Finding(index, item["type"], frozenset(affected), amount, item["currency"], frozenset(evidence)))
    return findings


def is_match(finding, issue, known_ids, tolerance=spec.AMOUNT_TOLERANCE_MINOR):
    if finding.type != issue["type"]:
        return False
    if finding.affected != frozenset(issue["affected_ids"]):
        return False
    if finding.currency != issue["currency"]:
        return False
    if abs(finding.amount - issue["expected_amount_minor"]) > tolerance:
        return False
    if not finding.evidence <= known_ids:
        return False
    return any(set(group) <= finding.evidence for group in issue["evidence_sets"])


def _max_matching(adjacency, n_left, fixed):
    owner = {}
    used_left = set()
    for issue, finding in fixed.items():
        owner[issue] = finding
        used_left.add(finding)

    def augment(i, seen):
        for j in adjacency[i]:
            if j in seen or j in fixed:
                continue
            seen.add(j)
            if j not in owner or augment(owner[j], seen):
                owner[j] = i
                return True
        return False

    for i in range(n_left):
        if i not in used_left:
            augment(i, set())
    return len(owner)


def issue_key(issue, position):
    return issue.get("issue_id", f"#{position:06d}")


def match(findings, issues, known_ids, tolerance=spec.AMOUNT_TOLERANCE_MINOR):
    adjacency = [[j for j, issue in enumerate(issues) if is_match(f, issue, known_ids, tolerance)] for f in findings]
    best = _max_matching(adjacency, len(findings), {})
    order = sorted(range(len(issues)), key=lambda j: issue_key(issues[j], j))
    fixed = {}
    for j in order:
        if len(fixed) == best:
            break
        for i in range(len(findings)):
            if j not in adjacency[i] or i in fixed.values():
                continue
            trial = dict(fixed)
            trial[j] = i
            if _max_matching(adjacency, len(findings), trial) == best:
                fixed = trial
                break
    return sorted((i, j) for j, i in fixed.items())


def scores(tp, fp, fn, n_truth, n_findings):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    if n_truth == 0 and n_findings == 0:
        f1 = 1.0
    elif n_truth == 0:
        f1 = 0.0
    else:
        f1 = 2 * tp / (2 * tp + fp + fn)
    return precision, recall, f1


def terminal_reward(f1, inspection_steps):
    value = 2 * f1 - 1 - spec.STEP_COST * inspection_steps
    return max(-1.0, min(1.0, value))


def evaluate(payload, truth, known_ids, inspection_steps, truncated=False):
    issues = truth["issues"]
    base = {"inspection_steps": inspection_steps, "truncated": truncated, "n_truth": len(issues)}
    if truncated:
        findings = []
    else:
        try:
            findings = parse_submission(payload, truth["episode_id"])
        except SubmissionError as exc:
            return {
                **base, "valid": False, "error": str(exc), "tp": 0, "fp": 0, "fn": len(issues),
                "precision": 0.0, "recall": 0.0, "f1": 0.0, "reward": -1.0, "n_findings": 0,
                "by_type": {}, "matches": [],
            }
    pairs = match(findings, issues, known_ids)
    tp = len(pairs)
    fp = len(findings) - tp
    fn = len(issues) - tp
    precision, recall, f1 = scores(tp, fp, fn, len(issues), len(findings))
    by_type = {t: {"tp": 0, "fp": 0, "fn": 0} for t in spec.FINDING_TYPES}
    matched_findings = {i for i, _ in pairs}
    matched_issues = {j for _, j in pairs}
    for j, issue in enumerate(issues):
        by_type[issue["type"]]["tp" if j in matched_issues else "fn"] += 1
    for finding in findings:
        if finding.index not in matched_findings:
            by_type.setdefault(finding.type if finding.type in by_type else "unknown", {"tp": 0, "fp": 0, "fn": 0})
            by_type[finding.type if finding.type in by_type else "unknown"]["fp"] += 1
    return {
        **base,
        "valid": True,
        "error": None,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "reward": terminal_reward(f1, inspection_steps),
        "n_findings": len(findings),
        "by_type": by_type,
        "matches": [[findings[i].index, issue_key(issues[j], j)] for i, j in pairs],
    }
