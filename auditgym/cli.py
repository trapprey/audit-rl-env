import argparse
import json
import math
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

from . import spec
from .config import preset_names
from .env import AuditEnv
from .generator import generate_episode
from .judge import Judge
from .policies import BASELINES
from .policies.scripted import ScriptedAuditor
from .runner import TrajectoryWriter, aggregate, by_type, run_episode, run_policy
from .splits import SPLITS
from .storage import save_episode, write_json

EVAL_POLICIES = [*BASELINES, "scripted_auditor"]


def _seeds(text):
    if "-" in text:
        low, high = text.split("-", 1)
        return list(range(int(low), int(high) + 1))
    return [int(x) for x in text.split(",")]


def cmd_generate(args):
    config = json.loads(args.config) if args.config else {}
    if args.preset:
        config["preset"] = args.preset
    episode = generate_episode(args.seed, config)
    save_episode(episode, Path(args.out) / episode.public["manifest"]["episode_id"])
    print(episode.public["manifest"]["episode_id"])


def cmd_transform(args):
    from .transform import transform_folder

    episode = transform_folder(args.source, args.target, args.seed)
    print(episode.public["manifest"]["episode_id"])


def _episode_folders(paths):
    for path in paths:
        path = Path(path)
        if (path / "public").is_dir():
            yield path
        else:
            yield from sorted(p.parent for p in path.glob("*/public") if p.is_dir())


def verify_folders(paths, verbose=True):
    from .episode_check import check_episode
    from .transform import load_episode

    failed = 0
    total = 0
    for folder in _episode_folders(paths):
        total += 1
        results = check_episode(load_episode(folder))
        bad = [(name, problems) for name, problems in results if problems]
        failed += bool(bad)
        if verbose or bad:
            print(f"{'FAIL' if bad else 'ok  '} {folder}")
        for name, problems in bad:
            print(f"     {name}: {problems[:3]}")
    return total, failed


def cmd_verify_episode(args):
    total, failed = verify_folders(args.paths, verbose=not args.quiet)
    print(f"{total - failed}/{total} episodes passed every check")
    return 1 if failed or not total else 0


def cmd_export(args):
    out = Path(args.out)
    for split, items in SPLITS.items():
        if (out / split).is_dir():
            shutil.rmtree(out / split)
        index = []
        for seed, config in items:
            episode = generate_episode(seed, config)
            eid = episode.public["manifest"]["episode_id"]
            save_episode(episode, out / split / eid)
            index.append({"episode_id": eid, "seed": seed, "config": config, "issues": len(episode.truth["issues"])})
        write_json(out / split / "index.json", index)
        print(f"{split}: {len(index)} episodes -> {out / split}")


def validate(seeds, presets):
    rows = []
    failures = []
    for preset in presets:
        for seed in seeds:
            started = time.perf_counter()
            try:
                episode = generate_episode(seed, {"preset": preset})
            except Exception as exc:
                failures.append({"preset": preset, "seed": seed, "error": f"{type(exc).__name__}: {exc}"[:400]})
                continue
            generated = time.perf_counter() - started
            from .episode_check import check_episode

            broken = [name for name, problems in check_episode(episode) if problems]
            if broken:
                failures.append({"preset": preset, "seed": seed, "error": f"file checks failed: {broken}"})
                continue
            counts = episode.public["manifest"]["record_counts"]
            pages = sum(math.ceil(counts[t] / spec.PAGE_LIMIT) for t in spec.TABLES)
            env = AuditEnv(judge=Judge("train"), mode="train")
            summary = run_episode(env, ScriptedAuditor(), seed, {"preset": preset})
            metrics = summary["train_metrics"]
            rows.append({
                "preset": preset,
                "seed": seed,
                "generate_seconds": generated,
                "episode_seconds": summary["seconds"],
                "pages_to_read_all": pages,
                "counts": counts,
                "issues": Counter(i["type"] for i in episode.truth["issues"]),
                "scenarios": Counter(i["scenario"] for i in episode.truth["issues"]),
                "scripted_f1": metrics["f1"],
                "scripted_fp": metrics["fp"],
                "scripted_fn": metrics["fn"],
            })
    return rows, failures


def summarize_validation(rows, failures):
    issues = Counter()
    scenarios = Counter()
    for row in rows:
        issues.update(row["issues"])
        scenarios.update(row["scenarios"])
    misses = [r for r in rows if r["scripted_f1"] < 1.0]
    return {
        "episodes": len(rows) + len(failures),
        "generation_failures": failures,
        "issue_counts": dict(issues),
        "scenario_counts": dict(scenarios),
        "clean_episodes": sum(1 for r in rows if not r["issues"]),
        "max_pages_to_read_all": max((r["pages_to_read_all"] for r in rows), default=0),
        "max_generate_seconds": max((r["generate_seconds"] for r in rows), default=0.0),
        "mean_episode_seconds": statistics.fmean(r["episode_seconds"] for r in rows) if rows else 0.0,
        "ledger_rows": [min(r["counts"]["ledger"] for r in rows), max(r["counts"]["ledger"] for r in rows)] if rows else [],
        "scripted_auditor_imperfect": [
            {"preset": r["preset"], "seed": r["seed"], "f1": r["scripted_f1"], "fp": r["scripted_fp"], "fn": r["scripted_fn"]}
            for r in misses
        ],
    }


def cmd_validate(args):
    presets = args.presets.split(",") if args.presets else preset_names()
    rows, failures = validate(_seeds(args.seeds), presets)
    report = summarize_validation(rows, failures)
    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, report)
    print(json.dumps({k: v for k, v in report.items() if k != "scripted_auditor_imperfect"}, indent=1))
    print(f"scripted auditor imperfect on {len(report['scripted_auditor_imperfect'])} episodes")
    return 1 if failures else 0


def evaluate_split(split, out_dir, policies=EVAL_POLICIES):
    out_dir = Path(out_dir) / split
    writer = TrajectoryWriter(out_dir / "trajectories.jsonl")
    log_path = (out_dir / "judge_log.jsonl").resolve()
    if log_path.exists():
        log_path.unlink()
    results = {}
    try:
        for name in policies:
            summaries = run_policy(name, SPLITS[split], mode="eval", sink=writer, log_path=log_path)
            results[name] = {"summary": aggregate(summaries), "by_type": by_type(summaries), "episodes": summaries}
    finally:
        writer.close()
    write_json(out_dir / "metrics.json", results)
    return results


def metrics_table(split, results):
    lines = [
        f"### {split}",
        "",
        "| policy | P | R | F1 | return | steps | sec/episode | invalid | truncated |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, block in results.items():
        s = block["summary"]
        lines.append(
            f"| {name} | {s['precision']:.3f} | {s['recall']:.3f} | {s['f1']:.3f} | {s['return']:.3f} | "
            f"{s['steps']:.1f} | {s['seconds']:.3f} | {s['invalid']} | {s['truncated']} |"
        )
    lines += ["", "Errors by type (tp/fp/fn):", "", "| policy | " + " | ".join(spec.FINDING_TYPES) + " |", "|---" * 5 + "|"]
    for name, block in results.items():
        cells = [f"{block['by_type'][t]['tp']}/{block['by_type'][t]['fp']}/{block['by_type'][t]['fn']}" for t in spec.FINDING_TYPES]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def cmd_eval(args):
    splits = args.split.split(",")
    parts = ["# Baseline metrics", "", "P and R are 0 on clean episodes by definition; F1 is 1 there only for an empty answer.", ""]
    for split in splits:
        results = evaluate_split(split, args.out)
        parts.append(metrics_table(split, results))
    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(parts))
    print("\n".join(parts))


def cmd_demo(args):
    config = {"preset": args.preset}
    env = AuditEnv(judge=Judge("train"), mode="train")
    policy = ScriptedAuditor()
    observation, info = env.reset(args.seed, config)
    policy.reset(observation, info, env)
    manifest = observation["manifest"]
    print(f"episode {manifest['episode_id']}  base={manifest['base_currency']}  tz={manifest['timezone']}  "
          f"period={manifest['period_start']} .. {manifest['period_end']}")
    print(f"records: {manifest['record_counts']}  fx rate date: {observation['policy']['fx']['rate_date']}")
    done = False
    while not done:
        action = policy.act(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if action["tool"] == "submit":
            for f in action["args"]["findings"]["findings"]:
                print(f"  finding {f['type']:<19} {f['affected_ids']} {f['claimed_amount_minor']} {f['currency']}")
        else:
            result = observation.get("result", {})
            print(f"step {observation['step']:>3}  {action['tool']} {action['args'].get('table', '')} "
                  f"-> {len(result.get('records', []))} records")
    print(f"reward={reward:.3f} metrics={info.get('metrics')}")


def cmd_serve(args):
    from .judge.client import JudgeClient

    env = AuditEnv(judge=JudgeClient(mode=args.mode), mode=args.mode)
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            if request.get("op") == "reset":
                observation, info = env.reset(request.get("seed", 0), request.get("config"))
                reply = {"observation": observation, "info": info}
            elif request.get("op") == "step":
                observation, reward, terminated, truncated, info = env.step(request.get("action"))
                reply = {"observation": observation, "reward": reward, "terminated": terminated, "truncated": truncated, "info": info}
            elif request.get("op") == "close":
                break
            else:
                reply = {"error": "unknown op"}
        except Exception as exc:
            reply = {"error": str(exc)[:300]}
        sys.stdout.write(json.dumps(reply) + "\n")
        sys.stdout.flush()
    env.close()


def cmd_check(args):
    steps = []

    def stage(name, ok, detail=""):
        steps.append((name, ok))
        print(f"[{'ok' if ok else 'FAIL'}] {name} {detail}".rstrip(), flush=True)

    tests = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"], cwd=Path(__file__).resolve().parents[1])
    stage("unit tests", tests.returncode == 0)

    rows, failures = validate(range(args.validate_seeds), preset_names())
    report = summarize_validation(rows, failures)
    write_json(Path(args.reports) / "validation.json", report)
    stage("generator self-checks and independent episode checks", not failures, f"({report['episodes']} episodes, {len(failures)} failures)")
    stage("evidence reachable within budget", report["max_pages_to_read_all"] < spec.MAX_STEPS,
          f"(max pages {report['max_pages_to_read_all']})")

    cmd_export(argparse.Namespace(out=args.episodes))
    stage("export train/dev episodes", True)

    total, failed = verify_folders([Path(args.episodes) / split for split in SPLITS], verbose=False)
    stage("independent file checks of exported episodes", total == 20 and not failed, f"({total - failed}/{total})")

    with tempfile.TemporaryDirectory() as tmp:
        from .transform import transform_folder

        moved = 0
        for folder in _episode_folders([Path(args.episodes) / split for split in SPLITS]):
            transform_folder(folder, Path(tmp) / folder.name, 7)
            moved += 1
        total, failed = verify_folders([tmp], verbose=False)
        stage("shuffled and relabeled copies pass the same checks", total == moved and not failed, f"({total - failed}/{total})")

    parts = ["# Baseline metrics", ""]
    sane = True
    for split in SPLITS:
        results = evaluate_split(split, args.runs)
        parts.append(metrics_table(split, results))
        sane &= results["oracle"]["summary"]["return"] >= 0.99
        sane &= results["flag_all"]["summary"]["return"] < 0
        sane &= results["random_explorer"]["summary"]["return"] < results["oracle"]["summary"]["return"]
    Path(args.reports).mkdir(parents=True, exist_ok=True)
    (Path(args.reports) / "metrics.md").write_text("\n".join(parts))
    print("\n".join(parts))
    stage("baseline sanity (oracle ~1, flag_all < 0)", sane)

    failed = [name for name, ok in steps if not ok]
    print("CHECK PASSED" if not failed else f"CHECK FAILED: {failed}")
    return 1 if failed else 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="auditgym")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("generate")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--preset")
    p.add_argument("--config")
    p.add_argument("--out", default="episodes/custom")
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("verify-episode")
    p.add_argument("paths", nargs="+")
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=cmd_verify_episode)

    p = sub.add_parser("transform")
    p.add_argument("source")
    p.add_argument("target")
    p.add_argument("--seed", type=int, default=1)
    p.set_defaults(func=cmd_transform)

    p = sub.add_parser("export")
    p.add_argument("--out", default="episodes")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("validate")
    p.add_argument("--seeds", default="0-49")
    p.add_argument("--presets")
    p.add_argument("--report", default="reports/validation.json")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("eval")
    p.add_argument("--split", default="train,dev")
    p.add_argument("--out", default="runs")
    p.add_argument("--report", default="reports/metrics.md")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("demo")
    p.add_argument("--seed", type=int, default=2004)
    p.add_argument("--preset", default="interacting")
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("serve")
    p.add_argument("--mode", choices=["train", "eval"], default="eval")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("check")
    p.add_argument("--validate-seeds", type=int, default=40)
    p.add_argument("--episodes", default="episodes")
    p.add_argument("--runs", default="runs")
    p.add_argument("--reports", default="reports")
    p.set_defaults(func=cmd_check)

    args = parser.parse_args(argv)
    code = args.func(args)
    sys.exit(code or 0)
