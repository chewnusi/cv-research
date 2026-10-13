import json

from conftest import plan_response

from cvorch.llm import ScriptedClient
from cvorch.normalize import normalize_plan
from cvorch.orchestrators import PlannedOrchestrator
from cvorch.plan import Plan, PlanStep, plan_json_schema
from cvorch.prompts import baseline_system_prompt, planner_system_prompt
from cvorch.workspace import default_workspace


def _norm(tool, **args):
    plan = Plan(status="ok", steps=[PlanStep(tool=tool, arguments=args)])
    return normalize_plan(plan, default_workspace()).steps[0].arguments


def test_normalization_is_format_only():
    assert _norm("export_yolo", dataset="./data/Warehouse/", val_split="10%", task=None) == {
        "dataset": "warehouse", "val_split": 0.1}
    assert _norm("train_yolo", dataset="parking", epochs="80") == {"dataset": "parking", "epochs": 80}
    assert _norm("train_yolo", dataset="parking", epochs=-10) == {"dataset": "parking", "epochs": -10}
    assert _norm("segment_dataset", dataset="birds") == {"dataset": "birds"}


def _objects(schema):
    if isinstance(schema, dict):
        if schema.get("type") == "object":
            yield schema
        for v in schema.values():
            yield from _objects(v)
    elif isinstance(schema, list):
        for v in schema:
            yield from _objects(v)


def test_strict_schema_is_structural_only():
    schema = plan_json_schema()
    for obj in _objects(schema):
        assert obj["additionalProperties"] is False
        assert set(obj["required"]) == set(obj["properties"])
    text = json.dumps(schema)
    for kw in ("minimum", "maximum", "exclusiveMinimum", "default", "yolo11n"):
        assert f'"{kw}"' not in text
    train = next(v for v in schema["properties"]["steps"]["items"]["anyOf"]
                 if v["properties"]["tool"]["enum"] == ["train_yolo"])
    names = train["properties"]["arguments"]["items"]["properties"]["name"]["enum"]
    assert names == ["dataset", "model", "epochs", "batch_size", "imgsz"]


def test_strict_argument_pairs_and_nulls_are_normalized(settings):
    from dataclasses import replace
    from cvorch.llm import LLMResponse
    content = json.dumps({"status": "ok", "message": "", "steps": [{"tool": "train_yolo", "arguments": [
        {"name": "dataset", "value": "data/parking"}, {"name": "epochs", "value": 80},
        {"name": "model", "value": None}]}]})
    llm = ScriptedClient([LLMResponse(content=content)])
    res = PlannedOrchestrator(llm, replace(settings, planner_response_format="json_schema")).run(
        "...", default_workspace())
    assert res.plan["steps"][0]["arguments"] == {"dataset": "parking", "epochs": 80}
    assert [c.ok for c in res.calls] == [True]
    assert '"name": "<parameter>"' in llm.requests[0][0]["content"]


def test_prompts_state_contract_rules_and_hide_job_details():
    ws = default_workspace()
    for prompt in (baseline_system_prompt(ws), planner_system_prompt(ws)):
        assert "Multiple training jobs may run at the same time" in prompt
        assert "'10%' or '10 percent' validation -> val_split 0.1" in prompt
        assert "A missing prerequisite is never a" in prompt
        assert "train-0042" in prompt
        assert "running" not in prompt and "completed" not in prompt
