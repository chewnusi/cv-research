"""Consistency checks of the benchmark itself, plus an oracle run per mode:
a perfect LLM must score 100%, which validates the ground truth and the scorer together."""
import json
from collections import Counter
from dataclasses import replace

import pytest
from conftest import plan_response, text_response, tool_calls_response

from cvorch.bench.runner import load_cases, run_benchmark
from cvorch.llm import ScriptedClient
from cvorch.plan import Plan, PlanStep
from cvorch.tools import MockToolRuntime
from cvorch.validator import validate_plan
from cvorch.workspace import default_workspace

CASES = load_cases()


def test_benchmark_size_and_coverage():
    assert len(CASES) >= 100
    assert len({c["id"] for c in CASES}) == len(CASES)
    cats = Counter(c["category"] for c in CASES)
    for required in ("single_tool", "tool_with_params", "multi_step", "complex_multi_step", "paraphrase",
                     "missing_optional", "invalid_parameter", "unsupported", "ambiguous"):
        assert cats[required] >= 5, required


@pytest.mark.parametrize("case", [c for c in CASES if c["expect"] == "execute"], ids=lambda c: c["id"])
def test_expected_plans_are_valid_and_executable(case):
    for steps in case["plans"]:
        plan = Plan(status="ok", steps=[PlanStep(tool=s["tool"], arguments=s["args"]) for s in steps])
        assert validate_plan(plan, default_workspace()).ok
        rt = MockToolRuntime(default_workspace())
        assert all(rt.call(s["tool"], s["args"]).ok for s in steps)


def _oracle_planned(case_by_prompt):
    def respond(messages, tools):
        case = case_by_prompt[messages[1]["content"]]
        if case["expect"] == "abstain":
            return plan_response(status="needs_clarification")
        return plan_response(steps=[(s["tool"], s["args"]) for s in case["plans"][0]])
    return respond


def _oracle_baseline(case_by_prompt):
    def respond(messages, tools):
        case = case_by_prompt[messages[1]["content"]]
        if case["expect"] == "abstain" or messages[-1]["role"] == "tool":
            return text_response("done")
        return tool_calls_response(*[(s["tool"], s["args"]) for s in case["plans"][0]])
    return respond


@pytest.mark.parametrize("mode", ["baseline", "planned"])
def test_oracle_scores_perfectly(mode, settings):
    by_prompt = {c["prompt"]: c for c in CASES}
    oracle = _oracle_planned if mode == "planned" else _oracle_baseline
    _, summary = run_benchmark(CASES, ScriptedClient(oracle(by_prompt)), replace(settings, mode=mode),
                               progress=False)
    o = summary["overall"]
    assert o["task_completion_rate"]["value"] == 1.0, json.dumps(summary["errors"])
    assert o["plan_exact_accuracy"]["value"] == 1.0
    assert o["abstention_accuracy"]["value"] == 1.0
    assert o["unsafe_execution_rate"]["value"] == 0.0


def test_naive_executor_is_penalised(settings):
    """An LLM that always tries to act on invalid requests must be caught by the metrics."""
    def respond(messages, tools):
        return plan_response(steps=[("train_yolo", {"dataset": "parking", "epochs": -10})])
    cases = [c for c in CASES if c["id"] in ("iv-01", "st-08")]
    _, summary = run_benchmark(cases, ScriptedClient(respond), settings, progress=False)
    o = summary["overall"]
    assert o["validator_blocked"] == 2 and o["unsafe_execution_rate"]["value"] == 0.0
    assert o["abstention_accuracy"]["value"] == 0.0
