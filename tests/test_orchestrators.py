from dataclasses import replace

from conftest import plan_response, text_response, tool_calls_response

from cvorch.llm import ScriptedClient
from cvorch.orchestrators import BaselineOrchestrator, PlannedOrchestrator, make_orchestrator
from cvorch.workspace import default_workspace


def test_mode_flag_selects_orchestrator(settings):
    llm = ScriptedClient([])
    assert isinstance(make_orchestrator(llm, replace(settings, mode="baseline")), BaselineOrchestrator)
    assert isinstance(make_orchestrator(llm, replace(settings, mode="planned")), PlannedOrchestrator)


def test_baseline_executes_tool_calls_and_feeds_results_back(settings):
    llm = ScriptedClient([
        tool_calls_response(("segment_dataset", {"dataset": "cats"})),
        tool_calls_response(("export_yolo", {"dataset": "cats"}), ("train_yolo", {"dataset": "cats"})),
        text_response("Done."),
    ])
    res = BaselineOrchestrator(llm, settings).run("Train on cats", default_workspace())
    assert [c.tool for c in res.executed] == ["segment_dataset", "export_yolo", "train_yolo"]
    assert res.llm_calls == 3 and res.final_message == "Done."
    assert llm.requests[1][-1]["role"] == "tool"


def test_baseline_partial_execution_on_invalid_last_step(settings):
    llm = ScriptedClient([
        tool_calls_response(("segment_dataset", {"dataset": "dogs"}), ("export_yolo", {"dataset": "dogs"}),
                            ("train_yolo", {"dataset": "dogs", "epochs": -30})),
        text_response("Training failed."),
    ])
    res = BaselineOrchestrator(llm, settings).run("...", default_workspace())
    assert [c.ok for c in res.calls] == [True, True, False]


def test_baseline_invalid_json_arguments(settings):
    from cvorch.llm import LLMResponse, ToolCall
    llm = ScriptedClient([
        LLMResponse(content=None, tool_calls=[ToolCall("c1", "train_yolo", "{not json")]),
        text_response("sorry"),
    ])
    res = BaselineOrchestrator(llm, settings).run("...", default_workspace())
    assert len(res.calls) == 1 and not res.calls[0].ok


def test_planned_executes_valid_plan(settings):
    llm = ScriptedClient([plan_response(steps=[("export_yolo", {"dataset": "warehouse"}),
                                               ("train_yolo", {"dataset": "warehouse", "epochs": 20})])])
    res = PlannedOrchestrator(llm, settings).run("...", default_workspace())
    assert not res.blocked_by_validator
    assert [c.tool for c in res.executed] == ["export_yolo", "train_yolo"]


def test_planned_blocks_invalid_plan_before_any_execution(settings):
    llm = ScriptedClient([plan_response(steps=[("segment_dataset", {"dataset": "dogs"}),
                                               ("export_yolo", {"dataset": "dogs"}),
                                               ("train_yolo", {"dataset": "dogs", "epochs": -30})])])
    res = PlannedOrchestrator(llm, settings).run("...", default_workspace())
    assert res.blocked_by_validator and res.calls == []
    assert res.validation_issues[0]["code"] == "invalid_arguments"


def test_planned_abstains(settings):
    llm = ScriptedClient([plan_response(status="unsupported", message="Faster R-CNN is not supported")])
    res = PlannedOrchestrator(llm, settings).run("...", default_workspace())
    assert res.plan_status == "unsupported" and res.calls == [] and not res.blocked_by_validator


def test_planned_repair_loop(settings):
    llm = ScriptedClient([
        plan_response(steps=[("train_yolo", {"dataset": "cats"})]),
        plan_response(steps=[("segment_dataset", {"dataset": "cats"}), ("export_yolo", {"dataset": "cats"}),
                             ("train_yolo", {"dataset": "cats"})]),
    ])
    res = PlannedOrchestrator(llm, replace(settings, planner_repair_attempts=1)).run("...", default_workspace())
    assert len(res.plan_attempts) == 2 and res.llm_calls == 2
    assert "dependency_violation" in llm.requests[1][-1]["content"]
    assert [c.tool for c in res.executed] == ["segment_dataset", "export_yolo", "train_yolo"]


def test_planned_without_validator_executes_until_runtime_failure(settings):
    llm = ScriptedClient([plan_response(steps=[("segment_dataset", {"dataset": "dogs"}),
                                               ("train_yolo", {"dataset": "dogs"}),
                                               ("export_yolo", {"dataset": "dogs"})])])
    res = PlannedOrchestrator(llm, replace(settings, planner_validate=False)).run("...", default_workspace())
    assert [c.ok for c in res.calls] == [True, False]


def test_planned_unparseable_output_is_blocked(settings):
    llm = ScriptedClient([text_response("I will segment the cats for you!")])
    res = PlannedOrchestrator(llm, settings).run("...", default_workspace())
    assert res.blocked_by_validator and res.calls == []
    assert res.validation_issues[0]["code"] == "plan_parse_error"


def test_planned_accepts_fenced_json(settings):
    resp = plan_response(steps=[("segment_dataset", {"dataset": "cats"})])
    resp.content = f"```json\n{resp.content}\n```"
    res = PlannedOrchestrator(ScriptedClient([resp]), settings).run("...", default_workspace())
    assert [c.tool for c in res.executed] == ["segment_dataset"]
