from cvorch.plan import Plan, PlanStep
from cvorch.validator import validate_plan
from cvorch.workspace import default_workspace


def _plan(*steps, status="ok"):
    return Plan(status=status, steps=[PlanStep(tool=t, arguments=a) for t, a in steps])


def _codes(plan):
    return [i.code for i in validate_plan(plan, default_workspace()).issues]


def test_full_pipeline_on_raw_dataset_is_valid():
    assert _codes(_plan(("segment_dataset", {"dataset": "cats"}),
                        ("export_yolo", {"dataset": "cats"}),
                        ("train_yolo", {"dataset": "cats", "epochs": 50}))) == []


def test_train_before_segment_is_rejected():
    assert _codes(_plan(("train_yolo", {"dataset": "cats"}),
                        ("segment_dataset", {"dataset": "cats"}))) == ["dependency_violation"]


def test_export_of_raw_dataset_is_rejected():
    assert _codes(_plan(("export_yolo", {"dataset": "dogs"}))) == ["dependency_violation"]


def test_annotated_dataset_can_be_exported_and_trained():
    assert _codes(_plan(("export_yolo", {"dataset": "warehouse"}), ("train_yolo", {"dataset": "warehouse"}))) == []


def test_yolo_dataset_can_be_trained_directly():
    assert _codes(_plan(("train_yolo", {"dataset": "data/parking/"}))) == []


def test_invalid_parameters():
    assert _codes(_plan(("train_yolo", {"dataset": "parking", "epochs": -10}))) == ["invalid_arguments"]
    assert _codes(_plan(("train_yolo", {"dataset": "parking", "model": "yolov5s"}))) == ["invalid_arguments"]
    assert _codes(_plan(("segment_dataset", {"dataset": "cats", "confidence": 1.5}))) == ["invalid_arguments"]
    assert _codes(_plan(("export_yolo", {"dataset": "warehouse", "format": "coco"}))) == ["invalid_arguments"]
    assert _codes(_plan(("train_yolo", {}))) == ["invalid_arguments"]


def test_unknown_tool_dataset_and_job():
    assert _codes(_plan(("deploy_model", {}))) == ["unknown_tool"]
    assert _codes(_plan(("segment_dataset", {"dataset": "birds"}))) == ["dataset_not_found"]
    assert _codes(_plan(("get_job_status", {"job_id": "train-9999"}))) == ["job_not_found"]


def _completed(*steps):
    from cvorch.validator import complete_prerequisites
    plan, inserted = complete_prerequisites(_plan(*steps), default_workspace())
    return [(s.tool, s.arguments.get("dataset")) for s in plan.steps], inserted


def test_complete_prerequisites():
    assert _completed(("train_yolo", {"dataset": "cats", "epochs": 5})) == (
        [("segment_dataset", "cats"), ("export_yolo", "cats"), ("train_yolo", "cats")], 2)
    assert _completed(("train_yolo", {"dataset": "warehouse"})) == (
        [("export_yolo", "warehouse"), ("train_yolo", "warehouse")], 1)
    assert _completed(("export_yolo", {"dataset": "data/dogs"})) == (
        [("segment_dataset", "dogs"), ("export_yolo", "data/dogs")], 1)
    full = (("segment_dataset", {"dataset": "cats"}), ("export_yolo", {"dataset": "cats"}),
            ("train_yolo", {"dataset": "cats"}))
    assert _completed(*full)[1] == 0
    assert _completed(("train_yolo", {"dataset": "parking"}), ("train_yolo", {"dataset": "birds"}))[1] == 0


def test_empty_ok_plan_and_abstentions():
    assert _codes(_plan()) == ["empty_plan"]
    assert _codes(_plan(status="unsupported")) == []
    assert _codes(_plan(status="needs_clarification")) == []
