from dataclasses import dataclass

from .. import spec


class SubmissionError(ValueError):
    pass


@dataclass(frozen=True)
class Finding:
    index: int
    type: str
    affected: tuple
    amount: int
    currency: str
    evidence: frozenset


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _id_list(value, name, limit, allow_empty):
    if not isinstance(value, list):
        raise SubmissionError(f"{name} must be a list")
    if len(value) > limit:
        raise SubmissionError(f"{name} has more than {limit} entries")
    if not value and not allow_empty:
        raise SubmissionError(f"{name} must not be empty")
    out = []
    for item in value:
        if not isinstance(item, str) or not item.strip() or len(item) > 100:
            raise SubmissionError(f"{name} entries must be non-empty strings")
        out.append(item.strip())
    return out


def parse_submission(payload, episode_id):
    if not isinstance(payload, dict):
        raise SubmissionError("submission must be an object")
    if set(payload) != set(spec.SUBMISSION_KEYS):
        raise SubmissionError("submission must have exactly episode_id and findings")
    if payload["episode_id"] != episode_id:
        raise SubmissionError("episode_id does not match")
    raw = payload["findings"]
    if not isinstance(raw, list):
        raise SubmissionError("findings must be a list")
    if len(raw) > spec.MAX_FINDINGS:
        raise SubmissionError(f"more than {spec.MAX_FINDINGS} findings")
    findings = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict) or set(item) != set(spec.FINDING_KEYS):
            raise SubmissionError(f"finding {index} must have exactly {list(spec.FINDING_KEYS)}")
        if item["type"] not in spec.FINDING_TYPES:
            raise SubmissionError(f"finding {index} has unknown type")
        affected = _id_list(item["affected_ids"], "affected_ids", spec.MAX_AFFECTED, False)
        evidence = _id_list(item["evidence_ids"], "evidence_ids", spec.MAX_EVIDENCE, True)
        amount = item["claimed_amount_minor"]
        if not _is_int(amount) or not 0 <= amount <= spec.MAX_AMOUNT:
            raise SubmissionError(f"finding {index} amount must be a non-negative integer")
        currency = item["currency"]
        if not isinstance(currency, str) or len(currency.strip()) != 3 or not currency.strip().isalpha():
            raise SubmissionError(f"finding {index} currency must be a 3-letter code")
        explanation = item["explanation"]
        if not isinstance(explanation, str) or len(explanation) > spec.MAX_EXPLANATION:
            raise SubmissionError(f"finding {index} explanation must be a string up to {spec.MAX_EXPLANATION} chars")
        findings.append(Finding(
            index,
            item["type"],
            tuple(sorted(set(affected))),
            amount,
            currency.strip().upper(),
            frozenset(evidence),
        ))
    return findings


def is_match(finding, issue, known_ids, tolerance=spec.AMOUNT_TOLERANCE_MINOR):
    if finding.type != issue["type"]:
        return False
    if finding.affected != tuple(issue["affected_ids"]):
        return False
    if finding.currency != issue["currency"]:
        return False
    if abs(finding.amount - issue["expected_amount_minor"]) > tolerance:
        return False
    if not finding.evidence <= known_ids:
        return False
    return any(set(group) <= finding.evidence for group in issue["evidence_sets"])


def match(findings, issues, known_ids, tolerance=spec.AMOUNT_TOLERANCE_MINOR):
    edges = [[j for j, issue in enumerate(issues) if is_match(f, issue, known_ids, tolerance)] for f in findings]
    owner = {}

    def augment(i, seen):
        for j in edges[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in owner or augment(owner[j], seen):
                owner[j] = i
                return True
        return False

    for i in range(len(findings)):
        augment(i, set())
    return sorted((i, j) for j, i in owner.items())


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
            by_type[finding.type]["fp"] += 1
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
        "matches": [[findings[i].index, issues[j]["issue_id"]] for i, j in pairs],
    }
