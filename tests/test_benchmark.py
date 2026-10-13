"""Consistency checks of the benchmark itself, plus oracle and literal-compliance runs per profile.

A perfect LLM must score 100% in both modes, which validates the ground truth and the scorer together.
A "literal" LLM (one that blindly does what was asked) shows the difference the benchmark is built to
measure: direct tool calling leaves partial side effects on late failures, full-plan validation does not.
"""
import json
from collections import Counter
from pathlib import Path

import pytest

from conftest import make_settings
from cvorch.bench.runner import BENCHMARK_DIR, load_cases, run_benchmark
from cvorch.llm import ScriptedClient, plan_turn, text_turn, tool_turn
from cvorch.normalize import normalize_plan
from cvorch.plan import ExecutionPlan, PlanStep
from cvorch.tools import MockToolRuntime
from cvorch.validator import validate_plan
from cvorch.workspace import default_workspace

CASES = load_cases()
UK_CASES = load_cases(BENCHMARK_DIR / "cases_uk.jsonl")
EXECUTE = [c for c in CASES if c["expect"] == "execute"]
WITH_LITERAL = [c for c in CASES if "literal" in c]


def _plan(steps):
    return ExecutionPlan(status="ok", steps=[PlanStep(tool=s["tool"], arguments=s["args"]) for s in steps])


def _validate(steps):
    ws = default_workspace()
    return validate_plan(normalize_plan(_plan(steps), ws), ws)


def test_cases_file_is_up_to_date():
    import runpy
    ns = runpy.run_path(str(BENCHMARK_DIR / "build_cases.py"), run_name="not_main")
    assert ns["CASES"] == CASES and ns["UK_CASES"] == UK_CASES


def test_benchmark_size_and_coverage():
    assert 170 <= len(CASES) <= 200
    assert len({c["id"] for c in CASES}) == len(CASES)
    cats = Counter(c["category"] for c in CASES)
    for required in ("single_tool", "tool_with_params", "multi_step", "complex_multi_step", "paraphrase",
                     "missing_optional", "implicit_prerequisites", "job_status", "labeling", "invalid_parameter",
                     "unsupported", "ambiguous", "nonexistent_reference"):
        assert cats[required] >= 10, required
    assert cats["late_failure"] >= 20


def test_no_removed_capabilities_in_execute_cases():
    for c in EXECUTE:
        for plan in c["plans"]:
            for s in plan:
                assert "model" not in s["args"] or not str(s["args"]["model"]).endswith("-seg"), c["id"]
                assert s["tool"] != "segment_dataset" or s["args"].get("text_prompt"), c["id"]
                assert "sam2" not in json.dumps(s).lower(), c["id"]


def test_abstain_cases_have_reason():
    for c in CASES:
        if c["expect"] == "abstain":
            assert c["reason"] in {"invalid_value", "unsupported", "ambiguous", "not_found"}, c["id"]
            assert "plans" not in c
        else:
            assert c["plans"] and "reason" not in c


@pytest.mark.parametrize("case", EXECUTE, ids=lambda c: c["id"])
def test_expected_plans_are_valid_and_executable(case):
    for steps in case["plans"]:
        assert _validate(steps).ok, case["id"]
        rt = MockToolRuntime(default_workspace())
        assert all(rt.call(s["tool"], s["args"]).ok for s in steps)


@pytest.mark.parametrize("case", WITH_LITERAL, ids=lambda c: c["id"])
def test_literal_plans_of_abstain_cases_are_rejected(case):
    assert case["expect"] == "abstain"
    assert not _validate(case["literal"]).ok


@pytest.mark.parametrize("case", [c for c in CASES if c["category"] == "late_failure"], ids=lambda c: c["id"])
def test_late_failures_have_a_valid_executable_prefix(case):
    issues = _validate(case["literal"]).issues
    first_bad = min(i.step for i in issues if i.step)
    assert first_bad >= 2
    rt = MockToolRuntime(default_workspace())
    assert all(rt.call(s["tool"], s["args"]).ok for s in case["literal"][: first_bad - 1])


def test_ukrainian_cases_mirror_english_labels():
    by_id = {c["id"]: c for c in CASES}
    assert 20 <= len(UK_CASES) <= 30
    for c in UK_CASES:
        src = by_id[c["source_id"]]
        assert c["expect"] == src["expect"] and c.get("plans") == src.get("plans")
        assert c["prompt"] != src["prompt"]


# ------------------------------------------------------------------ scripted end-to-end runs


def _case_of(messages, by_prompt):
    return by_prompt[messages[1].content]


def _oracle(by_prompt):
    def respond(messages, kind):
        case = _case_of(messages, by_prompt)
        if kind == "plan":
            if case["expect"] == "abstain":
                return plan_turn("needs_clarification", message="no")
            return plan_turn("ok", [(s["tool"], s["args"]) for s in case["plans"][0]])
        if case["expect"] == "abstain" or messages[-1].type == "tool":
            return text_turn("done")
        return tool_turn(*[(s["tool"], s["args"]) for s in case["plans"][0]])
    return respond


def _literal(by_prompt):
    """Blindly complies: executes the literal request; abstains only when nothing literal exists."""
    def respond(messages, kind):
        case = _case_of(messages, by_prompt)
        steps = case["plans"][0] if case["expect"] == "execute" else case.get("literal")
        if kind == "plan":
            if not steps:
                return plan_turn("needs_clarification")
            return plan_turn("ok", [(s["tool"], s["args"]) for s in steps])
        if not steps or messages[-1].type == "tool":
            return text_turn("done")
        return tool_turn(*[(s["tool"], s["args"]) for s in steps])
    return respond


@pytest.mark.parametrize("profile", ["baseline", "planned"])
def test_oracle_scores_perfectly(profile):
    by_prompt = {c["prompt"]: c for c in CASES}
    _, summary = run_benchmark(CASES, ScriptedClient(_oracle(by_prompt)), make_settings(profile), progress=False)
    o = summary["overall"]
    assert o["task_completion_rate"]["value"] == 1.0, json.dumps(summary["errors"])
    assert o["plan_exact_accuracy"]["value"] == 1.0
    assert o["abstention_accuracy"]["value"] == 1.0
    assert o["unsafe_execution_rate"]["value"] == 0.0
    assert summary["errors"] == {}


def test_literal_llm_shows_partial_side_effects_only_in_baseline():
    by_prompt = {c["prompt"]: c for c in CASES}
    _, base = run_benchmark(CASES, ScriptedClient(_literal(by_prompt)), make_settings("baseline"), progress=False)
    _, plan = run_benchmark(CASES, ScriptedClient(_literal(by_prompt)), make_settings("planned"), progress=False)
    assert base["overall"]["late_failure_side_effect_rate"]["value"] == 1.0
    assert plan["overall"]["late_failure_side_effect_rate"]["value"] == 0.0
    assert plan["overall"]["unsafe_execution_rate"]["value"] == 0.0
    assert plan["overall"]["validator_blocked"] == len(WITH_LITERAL)
    assert plan["overall"]["validator_false_blocks"] == 0
    # the same decisions are made in both modes; only execution control differs
    assert base["overall"]["abstention_accuracy"] == plan["overall"]["abstention_accuracy"]
    assert plan["overall"]["task_completion_positive"]["value"] == 1.0


def test_results_are_written(tmp_path: Path):
    cases = CASES[:5]
    by_prompt = {c["prompt"]: c for c in cases}
    run_benchmark(cases, ScriptedClient(_oracle(by_prompt)), make_settings("planned"), out_dir=tmp_path,
                  repeats=2, progress=False)
    lines = (tmp_path / "cases.jsonl").read_text().splitlines()
    assert len(lines) == 10 and {json.loads(x)["repeat"] for x in lines} == {0, 1}
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["meta"]["profile"] == "planned" and summary["meta"]["repeats"] == 2
    assert set(summary["meta"]["fingerprints"]) == {"cases", "initial_workspace", "tool_contracts", "shared_prompt"}
