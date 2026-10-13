from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def _rate(values: list[bool | None]) -> dict:
    vals = [v for v in values if v is not None]
    return {"value": round(sum(vals) / len(vals), 4) if vals else None, "n": len(vals)}


def _pctl(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    idx = min(len(xs) - 1, max(0, round(q * (len(xs) - 1))))
    return round(xs[idx], 4)


def _metrics(records: list[dict]) -> dict:
    scores = [r["score"] for r in records]
    results = [r["result"] for r in records]
    pos = [s for s in scores if s["expect"] == "execute"]
    neg = [s for s in scores if s["expect"] == "abstain"]
    lat = [r["total_latency_s"] for r in results]
    blocked = [s for s in scores if s["blocked_by_validator"]]
    return {
        "n": len(scores),
        # decision level
        "tool_selection_accuracy": _rate([s["tools_ok"] for s in pos]),
        "parameter_accuracy_given_tools": _rate([s["params_ok"] for s in pos if s["tools_ok"]]),
        "plan_exact_accuracy": _rate([s["plan_exact"] for s in pos]),
        "abstention_accuracy": _rate([s["abstain_ok"] for s in neg]),
        "false_abstention_rate": _rate([s["decided_abstain"] for s in pos]),
        # outcome level
        "task_completion_rate": _rate([s["task_success"] for s in scores]),
        "task_completion_positive": _rate([s["task_success"] for s in pos]),
        "task_completion_negative": _rate([s["task_success"] for s in neg]),
        "unsafe_execution_rate": _rate([s["unsafe_execution"] for s in neg]),
        "side_effects_on_failure_rate": _rate([s["side_effects_on_failure"] for s in scores]),
        "runtime_error_case_rate": _rate([s["runtime_errors"] > 0 for s in scores]),
        # validator (planned mode)
        "validator_blocked": len(blocked),
        "validator_true_catches": sum(1 for s in blocked if s["validator_true_catch"]),
        "validator_false_blocks": sum(1 for s in blocked if s["validator_true_catch"] is False),
        # cost
        "latency_mean_s": round(statistics.mean(lat), 4) if lat else None,
        "latency_median_s": _pctl(lat, 0.5),
        "latency_p95_s": _pctl(lat, 0.95),
        "llm_calls_mean": round(statistics.mean(r["llm_calls"] for r in results), 3) if results else None,
        "tokens_mean": round(statistics.mean(r["prompt_tokens"] + r["completion_tokens"] for r in results), 1)
        if results else None,
        "system_errors": sum(1 for r in results if r["error"]),
    }


def _error_counts(records: list[dict]) -> dict[str, int]:
    c: Counter[str] = Counter()
    for r in records:
        for e in set(r["score"]["errors"]):
            c[e.split(":")[0]] += 1
    return dict(c.most_common())


def summarize(records: list[dict]) -> dict:
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_cat[r["score"]["category"]].append(r)
    return {
        "overall": _metrics(records),
        "errors": _error_counts(records),
        "by_category": {cat: {**_metrics(rs), "errors": _error_counts(rs)} for cat, rs in sorted(by_cat.items())},
    }


# ------------------------------------------------------------------ comparison table

_ROWS = [
    ("tool_selection_accuracy", "Tool selection accuracy"),
    ("parameter_accuracy_given_tools", "Parameter accuracy (given correct tools)"),
    ("plan_exact_accuracy", "Workflow/plan exact accuracy"),
    ("abstention_accuracy", "Correct abstention (invalid/unsupported/ambiguous)"),
    ("false_abstention_rate", "False abstention rate (valid requests)"),
    ("task_completion_rate", "Task completion rate (all)"),
    ("task_completion_positive", "Task completion (valid requests)"),
    ("unsafe_execution_rate", "Unsafe execution rate (should-abstain requests)"),
    ("side_effects_on_failure_rate", "Failed with side effects"),
    ("runtime_error_case_rate", "Cases with runtime tool errors"),
]


def _fmt(v) -> str:
    if isinstance(v, dict):
        return "-" if v["value"] is None else f"{v['value'] * 100:.1f}% (n={v['n']})"
    return "-" if v is None else str(v)


def _label(summary: dict, path: Path) -> str:
    s = summary.get("meta", {}).get("settings", {})
    label = s.get("mode", path.name)
    if s.get("mode") == "planned":
        flags = []
        if not s.get("planner_validate", True):
            flags.append("no-validate")
        if s.get("planner_repair_attempts"):
            flags.append(f"repair={s['planner_repair_attempts']}")
        if s.get("planner_response_format") == "json_schema":
            flags.append("strict")
        if s.get("planner_auto_prerequisites"):
            flags.append("auto-prereq")
        if flags:
            label += f" ({', '.join(flags)})"
    return label


def compare_markdown(paths: list[Path]) -> str:
    summaries = [json.loads((p / "summary.json").read_text()) for p in paths]
    labels = [_label(s, p) for s, p in zip(summaries, paths)]
    models = {s.get("meta", {}).get("settings", {}).get("llm_model") for s in summaries}
    out = [f"Model(s): {', '.join(sorted(m for m in models if m))}", ""]

    out.append("| Metric | " + " | ".join(labels) + " |")
    out.append("|---" * (len(labels) + 1) + "|")
    for key, title in _ROWS:
        out.append(f"| {title} | " + " | ".join(_fmt(s["overall"][key]) for s in summaries) + " |")
    for key, title in [("validator_blocked", "Plans blocked by validator"),
                       ("validator_true_catches", "  ...correctly"),
                       ("validator_false_blocks", "  ...wrongly (correct plan blocked)"),
                       ("latency_mean_s", "Latency mean, s"), ("latency_median_s", "Latency median, s"),
                       ("latency_p95_s", "Latency p95, s"), ("llm_calls_mean", "LLM calls per request"),
                       ("tokens_mean", "Tokens per request"), ("system_errors", "System errors")]:
        out.append(f"| {title} | " + " | ".join(_fmt(s["overall"][key]) for s in summaries) + " |")

    out += ["", "Task completion by category:", ""]
    cats = sorted({c for s in summaries for c in s["by_category"]})
    out.append("| Category | " + " | ".join(labels) + " |")
    out.append("|---" * (len(labels) + 1) + "|")
    for cat in cats:
        cells = [_fmt(s["by_category"].get(cat, {}).get("task_completion_rate")) for s in summaries]
        out.append(f"| {cat} | " + " | ".join(cells) + " |")

    out += ["", "Error analysis (number of cases with each error):", ""]
    errs = sorted({e for s in summaries for e in s["errors"]})
    out.append("| Error | " + " | ".join(labels) + " |")
    out.append("|---" * (len(labels) + 1) + "|")
    for e in errs:
        out.append(f"| {e} | " + " | ".join(str(s["errors"].get(e, 0)) for s in summaries) + " |")
    return "\n".join(out)
