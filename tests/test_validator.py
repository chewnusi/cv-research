import pytest

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


def req_issues(p, ws, request):
    return [(i.step, i.code) for i in validate_plan(normalize_plan(p, ws), ws, request=request).issues]


# ------------------------------------------------------------------ references, jobs, quota


def test_local_paths_and_projects_created_in_the_plan(ws):
    dock = "/data/local/dock_cam"
    p = plan(("create_labeling_project", {"project_name": "dock-qa", "dataset": dock}),
             ("segment_dataset", {"dataset": "dock-qa", "text_prompt": "forklift"}),
             ("export_yolo", {"dataset": "dock-qa"}), ("train_yolo", {"dataset": "dock-qa"}))
    assert issues(p, ws) == []
    assert issues(plan(("train_yolo", {"dataset": "data/parking"})), ws) == [(1, "dataset_not_found")]
    assert issues(plan(("create_labeling_project", {"project_name": "Parking", "dataset": dock})), ws) == [
        (1, "project_exists")]


def test_cancel_job_states_are_simulated(ws):
    c42 = ("cancel_job", {"job_id": "train-0042"})
    assert issues(plan(c42), ws) == []
    assert issues(plan(c42, c42), ws) == [(2, "job_not_cancellable")]
    assert issues(plan(("cancel_job", {"job_id": "train-0041"})), ws) == [(1, "job_not_cancellable")]
    assert issues(plan(("cancel_job", {"job_id": "train-0099"})), ws) == [(1, "job_not_found")]


def test_gpu_quota_is_simulated_over_the_plan(ws):
    t = lambda m: ("train_yolo", {"dataset": "parking", "model": m})  # noqa: E731
    c42 = ("cancel_job", {"job_id": "train-0042"})
    assert issues(plan(t("yolo11n"), t("yolo11s")), ws) == []
    assert issues(plan(t("yolo11n"), t("yolo11s"), t("yolo11m")), ws) == [(3, "quota_exceeded")]
    assert issues(plan(c42, t("yolo11n"), t("yolo11s"), t("yolo11m")), ws) == []
    assert issues(plan(t("yolo11n"), t("yolo11s"), t("yolo11m"), c42), ws) == [(3, "quota_exceeded")]


# ------------------------------------------------------------------ grounding in the request


def test_grounded_plan_passes(ws):
    r = 'Segment cats for "cats", export them and train yolo11n for 50 epochs.'
    assert req_issues(plan(SEG, EXP, TRN), ws, r) == []


@pytest.mark.parametrize("request_text,step", [
    ("Train the small model on drones.", ("train_yolo", {"dataset": "drones", "model": "yolo11s"})),
    ("Export the weld scans to YOLO.", ("export_yolo", {"dataset": "xray"})),
    ("Segment the street frames.", ("segment_dataset", {"dataset": "traffic", "text_prompt": "car"})),
    ("Segment traffic.", ("segment_dataset", {"dataset": "traffic", "text_prompt": "car"})),
    ("Check my last job.", ("get_job_status", {"job_id": "train-0043"})),
    ("Cancel the running job.", ("cancel_job", {"job_id": "train-0042"})),
    ("Train on parking for a while.", ("train_yolo", {"dataset": "parking", "epochs": 30})),
    ("Train yolo11s on shelves.", ("train_yolo", {"dataset": "/data/local/shelves", "model": "yolo11s"})),
    ("Create a labeling project for cats.", ("create_labeling_project", {"project_name": "cats-qa", "dataset": "cats"})),
])
def test_ungrounded_values_are_blocked(ws, request_text, step):
    assert ("ungrounded_value" in [c for _, c in req_issues(plan(step), ws, request_text)])


@pytest.mark.parametrize("request_text,step", [
    ("Train yolov8n on parking.", ("train_yolo", {"dataset": "parking"})),
    ("Train yolo11n-seg on parking.", ("train_yolo", {"dataset": "parking", "model": "yolo11n"})),
    ("Export warehouse to COCO.", ("export_yolo", {"dataset": "warehouse"})),
    ("Export warehouse and check train-9999.", ("export_yolo", {"dataset": "warehouse"})),
])
def test_dropped_request_values_are_blocked(ws, request_text, step):
    assert (None, "request_value_not_used") in req_issues(plan(step), ws, request_text)


def test_grounding_accepts_format_conversions_and_defaults(ws):
    assert req_issues(plan(("export_yolo", {"dataset": "warehouse", "val_split": 0.1})), ws,
                      "Export warehouse with 10% for validation.") == []
    assert req_issues(plan(("train_yolo", {"dataset": "drones", "epochs": 120})), ws,
                      "Use a hundred and twenty epochs and train on drones.") == []
    # explicit tool defaults are not invented values
    assert req_issues(plan(("train_yolo", {"dataset": "drones", "model": "yolo11n", "batch_size": 16})), ws,
                      "Train a YOLO11 model on drones.") == []
    assert req_issues(plan(("export_yolo", {"dataset": "s3://cv-projects/xray/"})), ws, "Export xray.") == []


def test_repair_guard(ws):
    from cvorch.validator import repair_guard_issues
    r = "Train on drones for -5 epochs."
    bad = normalize_plan(plan(("train_yolo", {"dataset": "drones", "epochs": -5})), ws)
    found = validate_plan(bad, ws, request=r).issues
    fixed = plan(("train_yolo", {"dataset": "drones", "epochs": 1}))
    assert [i.code for i in repair_guard_issues(bad, found, fixed, r)] == ["repair_changed_request"]
    assert repair_guard_issues(bad, found, plan(status="unsupported"), r) == []
    # a value the LLM invented itself may be repaired
    r2 = "Train on drones."
    found2 = validate_plan(bad, ws, request=r2).issues
    assert repair_guard_issues(bad, found2, plan(("train_yolo", {"dataset": "drones"})), r2) == []


@pytest.mark.parametrize("request_text,steps", [
    ("Segment the cats dataset.", [("segment_dataset", {"dataset": "cats", "text_prompt": "cat"})]),
    ("Run the full pipeline on dogs.", [("segment_dataset", {"dataset": "dogs", "text_prompt": "dog"}),
                                        ("export_yolo", {"dataset": "dogs"}), ("train_yolo", {"dataset": "dogs"})]),
    ("Annotate the fruits dataset with SAM3.", [("segment_dataset", {"dataset": "fruits", "text_prompt": "fruit"})]),
    ("Segment /data/local/dock_cam.", [("segment_dataset", {"dataset": "/data/local/dock_cam", "text_prompt": "dock"})]),
])
def test_sam3_prompt_derived_from_the_dataset_name_is_ungrounded(ws, request_text, steps):
    assert (1, "ungrounded_value") in req_issues(plan(*steps), ws, request_text)


@pytest.mark.parametrize("request_text,prompt", [
    ('Segment cats for "cats".', "cats"),  # a quoted prompt is explicit even if it equals the dataset name
    ("Run SAM3 on cats to segment every cat.", "cat"),
    ('Segment cats ("cat") and dogs ("dog").', "cat"),
])
def test_explicit_prompt_next_to_the_dataset_name_is_grounded(ws, request_text, prompt):
    assert req_issues(plan(("segment_dataset", {"dataset": "cats", "text_prompt": prompt})), ws, request_text) == []
