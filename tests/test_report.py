from conftest import make_settings
from cvorch.bench.report import compare_markdown
from cvorch.bench.runner import load_cases, run_benchmark
from cvorch.bench.stats import bootstrap_ci, paired_diff_ci
from cvorch.llm import ScriptedClient, plan_turn, text_turn, tool_turn


def test_bootstrap_is_deterministic_and_bounded():
    vals = {str(i): float(i % 2) for i in range(40)}
    lo, hi = bootstrap_ci(vals)
    assert (lo, hi) == bootstrap_ci(vals) and 0 <= lo <= 0.5 <= hi <= 1
    assert bootstrap_ci({"a": 1.0}) is None


def test_paired_difference():
    a = {str(i): 0.0 for i in range(30)}
    b = {str(i): 1.0 for i in range(30)}
    diff, ci = paired_diff_ci(a, b)
    assert diff == 1.0 and ci == (1.0, 1.0)


def test_compare_report(tmp_path):
    cases = [c for c in load_cases() if c["id"] in ("st-07", "iv-01", "lf-05", "am-01")]
    by_prompt = {c["prompt"]: c for c in cases}

    def act(messages, kind):
        """Always tries the literal request."""
        case = by_prompt[messages[1].content]
        steps = case.get("plans", [None])[0] if case["expect"] == "execute" else case.get("literal")
        if kind == "plan":
            return plan_turn("ok", [(s["tool"], s["args"]) for s in steps]) if steps else plan_turn("needs_clarification")
        if not steps or messages[-1].type == "tool":
            return text_turn("ok")
        return tool_turn(*[(s["tool"], s["args"]) for s in steps])

    for p in ("baseline", "planned"):
        run_benchmark(cases, ScriptedClient(act), make_settings(p), out_dir=tmp_path / p, progress=False)
    md = compare_markdown([tmp_path / "baseline", tmp_path / "planned"])
    assert "Δ planned − baseline" in md
    assert "Partial execution on late-failure workflows" in md
    assert "Warning" not in md
    assert "| late_failure |" in md and "| partial_side_effects |" in md
    assert "Input tokens per request" in md
