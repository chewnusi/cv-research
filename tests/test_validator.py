from cvorch.normalize import normalize_plan
from cvorch.plan import ExecutionPlan, PlanStep
from cvorch.validator import complete_prerequisites, validate_plan


def plan(*steps, status="ok"):
    return ExecutionPlan(status=status, steps=[PlanStep(tool=t, arguments=a) for t, a in steps])


def issues(p, ws):
    return [(i.step, i.code) for i in validate_plan(normalize_plan(p, ws), ws).issues]


SEG = ("segment_dataset", {"dataset": "cats", "text_prompt": "cat"})
EXP = ("export_yolo", {"dataset": "cats"})
TRN = ("train_yolo", {"dataset": "cats", "model": "yolo11n", "epochs": 50})


def test_valid_full_workflow(ws):
    assert issues(plan(SEG, EXP, TRN), ws) == []


def test_full_plan_state_simulation_does_not_mutate_workspace(ws):
    assert issues(plan(SEG, EXP, TRN), ws) == []
    assert ws.datasets["cats"].state == "raw"


def test_invalid_export_on_raw(ws):
    assert issues(plan(EXP), ws) == [(1, "missing_prerequisite")]


def test_invalid_train_on_annotated(ws):
    assert issues(plan(("train_yolo", {"dataset": "warehouse"})), ws) == [(1, "missing_prerequisite")]


def test_invalid_order(ws):
    assert issues(plan(EXP, SEG), ws) == [(1, "invalid_order")]


def test_late_invalid_step_reports_the_late_step(ws):
    bad_train = ("train_yolo", {"dataset": "cats", "epochs": 0})
    assert issues(plan(SEG, EXP, bad_train), ws) == [(3, "out_of_range")]


def test_unknown_tool_and_argument_errors(ws):
    assert issues(plan(("deploy_model", {"dataset": "parking"})), ws) == [(1, "unknown_tool")]
    assert issues(plan(("train_yolo", {"dataset": "parking", "model": "yolo11n-seg"})), ws) == [
        (1, "invalid_enum_value")]
    assert issues(plan(("train_yolo", {"dataset": "parking", "task": "segment"})), ws) == [
        (1, "unsupported_argument")]
    assert issues(plan(("train_yolo", {"dataset": "parking", "epochs": "lots"})), ws) == [(1, "wrong_type")]
    assert issues(plan(("segment_dataset", {"dataset": "cats"})), ws) == [(1, "missing_argument")]


def test_nonexistent_dataset_and_job(ws):
    assert issues(plan(("train_yolo", {"dataset": "birds"})), ws) == [(1, "dataset_not_found")]
    assert issues(plan(("get_job_status", {"job_id": "train-9999"})), ws) == [(1, "job_not_found")]


def test_empty_ok_plan_is_invalid(ws):
    assert issues(plan(), ws) == [(None, "empty_plan")]


def test_abstention_plans_are_not_validated(ws):
    assert issues(plan(status="unsupported"), ws) == []
    assert issues(plan(status="needs_clarification"), ws) == []


def test_all_issues_are_reported(ws):
    p = plan(("train_yolo", {"dataset": "birds"}), ("get_job_status", {"job_id": "nope"}),
             ("export_yolo", {"dataset": "warehouse", "val_split": 0.9}))
    assert [c for _, c in issues(p, ws)] == ["dataset_not_found", "job_not_found", "out_of_range"]


def test_auto_prerequisites_ablation_only_inserts_export(ws):
    p, n = complete_prerequisites(plan(("train_yolo", {"dataset": "warehouse"})), ws)
    assert n == 1 and [s.tool for s in p.steps] == ["export_yolo", "train_yolo"]
    p, n = complete_prerequisites(plan(("train_yolo", {"dataset": "cats"})), ws)
    assert n == 0  # raw datasets need a user-provided SAM3 prompt
