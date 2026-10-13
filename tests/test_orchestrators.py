from langchain_core.messages import AIMessage, ToolMessage

from conftest import make_settings
from cvorch.llm import PlanTurn, ScriptedClient, ToolCall, ToolTurn, plan_turn, text_turn, tool_turn
from cvorch.orchestrators import BaselineOrchestrator, PlannedOrchestrator, make_orchestrator
from cvorch.prompts import baseline_system_prompt, planner_system_prompt, shared_prompt_prefix

SEG = ("segment_dataset", {"dataset": "cats", "text_prompt": "cat"})
EXP = ("export_yolo", {"dataset": "cats"})
TRN = ("train_yolo", {"dataset": "cats", "model": "yolo11n", "epochs": 50})
BAD_TRN = ("train_yolo", {"dataset": "cats", "epochs": 0})


def test_profile_selects_orchestrator():
    llm = ScriptedClient([])
    assert isinstance(make_orchestrator(llm, make_settings("baseline")), BaselineOrchestrator)
    for p in ("planned", "planned_no_validator", "planned_strict_schema", "planned_auto_prereq", "planned_repair"):
        assert isinstance(make_orchestrator(llm, make_settings(p)), PlannedOrchestrator)


def test_both_modes_share_the_same_prompt_prefix(ws):
    prefix = shared_prompt_prefix(ws)
    assert baseline_system_prompt(ws).startswith(prefix)
    assert planner_system_prompt(ws).startswith(prefix)
    assert "SAM3" in prefix and "Never derive it from a dataset name" in prefix


# ------------------------------------------------------------------ baseline


def test_baseline_executes_tool_calls_and_feeds_results_back(ws):
    llm = ScriptedClient([tool_turn(SEG), tool_turn(EXP), text_turn("done")])
    res = BaselineOrchestrator(llm, make_settings("baseline")).run("...", ws)
    assert [c.tool for c in res.executed] == ["segment_dataset", "export_yolo"]
    assert res.final_message == "done" and res.llm_calls == 3
    second_request = llm.requests[1][1]
    assert isinstance(second_request[-1], ToolMessage) and second_request[-1].tool_call_id == "call_0"
    assert "annotation_path" in second_request[-1].content
    assert ws.datasets["cats"].state == "yolo"


def test_baseline_partial_execution_on_invalid_late_step(ws):
    llm = ScriptedClient([tool_turn(SEG, EXP, BAD_TRN), text_turn("training failed")])
    res = BaselineOrchestrator(llm, make_settings("baseline")).run("...", ws)
    assert [c.tool for c in res.executed] == ["segment_dataset", "export_yolo"]
    assert not res.calls[-1].ok
    assert ws.datasets["cats"].state == "yolo"  # side effects remain


def test_baseline_invalid_json_arguments_are_answered(ws):
    bad = ToolTurn(AIMessage(content="", invalid_tool_calls=[
        {"name": "train_yolo", "args": "{oops", "id": "c1", "error": "bad json", "type": "invalid_tool_call"}]),
        [ToolCall("c1", "train_yolo", None, raw_args="{oops", error="bad json")])
    llm = ScriptedClient([bad, text_turn("sorry")])
    res = BaselineOrchestrator(llm, make_settings("baseline")).run("...", ws)
    assert len(res.calls) == 1 and not res.calls[0].ok
    assert isinstance(llm.requests[1][1][-1], ToolMessage)


def test_baseline_max_rounds(ws):
    llm = ScriptedClient(lambda messages, kind: tool_turn(("get_job_status", {"job_id": "train-0041"})))
    res = BaselineOrchestrator(llm, make_settings("baseline", baseline_max_rounds=3)).run("...", ws)
    assert res.error == "max_rounds_exceeded" and res.llm_calls == 3


# ------------------------------------------------------------------ planned


def test_planned_executes_valid_plan(ws):
    llm = ScriptedClient([plan_turn("ok", [SEG, EXP, TRN])])
    res = PlannedOrchestrator(llm, make_settings()).run("...", ws)
    assert [c.tool for c in res.executed] == ["segment_dataset", "export_yolo", "train_yolo"]
    assert not res.blocked_by_validator and res.llm_calls == 1


def test_planned_blocks_late_failure_before_any_execution(ws):
    llm = ScriptedClient([plan_turn("ok", [SEG, EXP, BAD_TRN])])
    res = PlannedOrchestrator(llm, make_settings()).run("...", ws)
    assert res.blocked_by_validator and res.calls == []
    assert [i["step"] for i in res.validation_issues] == [3]
    assert ws.datasets["cats"].state == "raw"


def test_planned_blocks_unsupported_final_model(ws):
    llm = ScriptedClient([plan_turn("ok", [SEG, EXP, ("train_yolo", {"dataset": "cats", "model": "yolo11n-seg"})])])
    res = PlannedOrchestrator(llm, make_settings()).run("...", ws)
    assert res.blocked_by_validator and res.calls == [] and ws.datasets["cats"].state == "raw"


def test_prepare_never_executes(ws):
    llm = ScriptedClient([plan_turn("ok", [SEG, EXP])])
    orch = PlannedOrchestrator(llm, make_settings())
    res = orch.prepare("...", ws)
    assert res.executable and res.calls == [] and ws.datasets["cats"].state == "raw"
    orch.execute(res, ws)
    assert ws.datasets["cats"].state == "yolo"


def test_execute_revalidates_against_current_workspace(ws):
    llm = ScriptedClient([plan_turn("ok", [("train_yolo", {"dataset": "parking"})])])
    orch = PlannedOrchestrator(llm, make_settings())
    res = orch.prepare("...", ws)
    assert res.executable
    del ws.datasets["parking"]  # the workspace changed between planning and confirmation
    orch.execute(res, ws)
    assert res.blocked_by_validator and res.calls == []


def test_planned_abstains(ws):
    for status in ("unsupported", "needs_clarification"):
        llm = ScriptedClient([plan_turn(status, message="no")])
        res = PlannedOrchestrator(llm, make_settings()).run("...", ws)
        assert res.plan_status == status and res.calls == [] and not res.blocked_by_validator


def test_planned_without_repair_makes_one_llm_call(ws):
    llm = ScriptedClient([plan_turn("ok", [BAD_TRN])])
    res = PlannedOrchestrator(llm, make_settings()).run("...", ws)
    assert res.llm_calls == 1 and res.blocked_by_validator


def test_planned_repair_ablation(ws):
    llm = ScriptedClient([plan_turn("ok", [EXP]), plan_turn("ok", [SEG, EXP])])
    res = PlannedOrchestrator(llm, make_settings("planned_repair")).run("...", ws)
    assert res.llm_calls == 2 and not res.blocked_by_validator
    assert [c.tool for c in res.executed] == ["segment_dataset", "export_yolo"]
    assert "validator rejected" in llm.requests[1][1][-1].content


def test_planned_no_validator_ablation_executes_until_runtime_failure(ws):
    llm = ScriptedClient([plan_turn("ok", [SEG, EXP, BAD_TRN])])
    res = PlannedOrchestrator(llm, make_settings("planned_no_validator")).run("...", ws)
    assert [c.tool for c in res.executed] == ["segment_dataset", "export_yolo"] and not res.calls[-1].ok


def test_planned_auto_prereq_ablation(ws):
    llm = ScriptedClient([plan_turn("ok", [("train_yolo", {"dataset": "warehouse"})])])
    res = PlannedOrchestrator(llm, make_settings("planned_auto_prereq")).run("...", ws)
    assert res.inserted_prerequisites == 1
    assert [c.tool for c in res.executed] == ["export_yolo", "train_yolo"]


def test_planned_strict_schema_uses_argument_pairs(ws):
    data = {"status": "ok", "message": "", "steps": [
        {"tool": "train_yolo", "arguments": [{"name": "dataset", "value": "parking"},
                                             {"name": "epochs", "value": 10}, {"name": "imgsz", "value": None}]}]}
    llm = ScriptedClient([PlanTurn(data)])
    res = PlannedOrchestrator(llm, make_settings("planned_strict_schema")).run("...", ws)
    assert res.executed[0].args == {"dataset": "parking", "epochs": 10}
    assert '"name": "<parameter>"' in llm.requests[0][1][0].content


def test_planned_unparseable_output_is_not_executed(ws):
    for turn in (PlanTurn(None, raw_text="not json", parse_error="OutputParserException: bad"),
                 PlanTurn({"status": "maybe", "steps": []})):
        res = PlannedOrchestrator(ScriptedClient([turn]), make_settings()).run("...", ws)
        assert res.plan_parse_error and res.calls == [] and not res.executable


def test_llm_exception_is_recorded_as_system_error(ws):
    def boom(messages, kind):
        raise TimeoutError("api down")
    for profile in ("baseline", "planned"):
        res = make_orchestrator(ScriptedClient(boom), make_settings(profile)).run("...", ws)
        assert res.error and "api down" in res.error and res.executed == []
