from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import PROFILES, Settings
from .llm import make_client
from .orchestrators import make_orchestrator
from .workspace import default_workspace


def _cmd_run(args: argparse.Namespace) -> int:
    settings = Settings.from_env(profile=args.profile)
    orch = make_orchestrator(make_client(settings), settings)
    res = orch.run(args.prompt, default_workspace())
    if args.json:
        print(json.dumps(res.to_dict(), indent=2))
        return 0
    print(f"profile: {settings.profile}   model: {settings.llm_model}   latency: {res.total_latency_s:.2f}s")
    if res.mode == "planned":
        print(f"plan ({res.plan_status}):")
        for i, s in enumerate((res.plan or {}).get("steps", []), 1):
            print(f"  {i}. {s['tool']}({json.dumps(s.get('arguments', {}))})")
        for issue in res.validation_issues:
            print(f"  validator: [{issue['code']}] step {issue['step']}: {issue['message']}")
        if res.blocked_by_validator:
            print("  -> plan blocked by the validator, nothing executed")
    print("tool calls:" if res.calls else "tool calls: none")
    for c in res.calls:
        status = "ok " if c.ok else "ERR"
        detail = json.dumps(c.output) if c.ok else c.error
        print(f"  [{status}] {c.tool}({json.dumps(c.args)}) -> {detail}")
    if res.final_message:
        print(f"assistant: {res.final_message}")
    if res.error:
        print(f"error: {res.error}")
    return 0


def _cmd_bench(args: argparse.Namespace) -> int:
    from .bench.runner import DEFAULT_CASES, default_out_dir, load_cases, run_benchmark

    settings = Settings.from_env(profile=args.profile)
    cases_path = Path(args.cases) if args.cases else DEFAULT_CASES
    cases = load_cases(cases_path, args.category, args.ids, args.limit)
    if not cases:
        print("no cases selected", file=sys.stderr)
        return 1
    out = Path(args.out) if args.out else default_out_dir(settings)
    _, summary = run_benchmark(cases, make_client(settings), settings, out, repeats=args.repeats,
                               progress=not args.quiet, cases_path=cases_path)
    o = summary["overall"]
    print(f"\nprofile={settings.profile} model={settings.llm_model} cases={len(cases)} repeats={args.repeats}")
    for k in ("task_completion_rate", "task_completion_positive", "plan_exact_accuracy", "abstention_accuracy",
              "unsafe_execution_rate", "late_failure_side_effect_rate"):
        v = o[k]["value"]
        print(f"  {k:<32} {'-' if v is None else f'{v * 100:.1f}%'}")
    print(f"  latency_mean_s                   {o['latency_mean_s']}")
    print(f"results: {out}")
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    from .bench.report import compare_markdown

    md = compare_markdown([Path(p) for p in args.dirs])
    print(md)
    if args.out:
        Path(args.out).write_text(md + "\n")
    return 0


def _cmd_profiles(args: argparse.Namespace) -> int:
    for p in PROFILES.values():
        tag = "primary " if p.primary else "ablation"
        print(f"{p.name:<24} [{tag}] {p.description}")
    return 0


def _cmd_ui(args: argparse.Namespace) -> int:
    import subprocess

    app = Path(__file__).with_name("ui") / "app.py"
    return subprocess.call([sys.executable, "-m", "streamlit", "run", str(app), *args.streamlit_args])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="cvorch", description="LLM orchestration of CV workflows (simulated tools).")
    sub = p.add_subparsers(dest="cmd", required=True)
    profile_help = f"experiment profile (default: $ORCH_PROFILE or 'planned'): {', '.join(PROFILES)}"

    r = sub.add_parser("run", help="run a single request")
    r.add_argument("prompt")
    r.add_argument("--profile", help=profile_help)
    r.add_argument("--json", action="store_true")
    r.set_defaults(func=_cmd_run)

    b = sub.add_parser("bench", help="run the benchmark")
    b.add_argument("--profile", help=profile_help)
    b.add_argument("--cases", help="path to a cases JSONL file (default: benchmark/cases.jsonl)")
    b.add_argument("--category", nargs="*", help="only these categories")
    b.add_argument("--ids", nargs="*", help="only these case ids")
    b.add_argument("--limit", type=int)
    b.add_argument("--repeats", type=int, default=1, help="run every case N times (LLM non-determinism)")
    b.add_argument("--out", help="output directory (default: results/<timestamp>_<profile>_<model>)")
    b.add_argument("--quiet", action="store_true")
    b.set_defaults(func=_cmd_bench)

    c = sub.add_parser("compare", help="compare result directories (first one is the reference)")
    c.add_argument("dirs", nargs="+")
    c.add_argument("--out", help="also write the markdown report to this file")
    c.set_defaults(func=_cmd_compare)

    pr = sub.add_parser("profiles", help="list experiment profiles")
    pr.set_defaults(func=_cmd_profiles)

    u = sub.add_parser("ui", help="start the Streamlit demo (requires the 'ui' extra)")
    u.add_argument("streamlit_args", nargs=argparse.REMAINDER, help="extra arguments for `streamlit run`")
    u.set_defaults(func=_cmd_ui)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
