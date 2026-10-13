import pytest

from cvorch.tools import (
    TOOLS,
    YOLO_DETECTION_MODELS,
    MockToolRuntime,
    check_arguments,
    normalize_arguments,
    openai_tools,
)


def _codes(tool, args):
    return {e.code for e in check_arguments(tool, args)[1]}


# ------------------------------------------------------------------ contracts


def test_tool_set_is_exactly_the_five_contract_tools():
    assert set(TOOLS) == {"segment_dataset", "export_yolo", "train_yolo", "get_job_status",
                          "create_labeling_project"}
    assert {t["function"]["name"] for t in openai_tools()} == set(TOOLS)


def test_segment_contract_is_sam3_text_prompt_only():
    props = TOOLS["segment_dataset"].parameters_schema()["properties"]
    assert set(props) == {"dataset", "text_prompt", "class_name", "confidence"}
    assert set(TOOLS["segment_dataset"].parameters_schema()["required"]) == {"dataset", "text_prompt"}
    assert "SAM3" in TOOLS["segment_dataset"].description
    assert _codes("segment_dataset", {"dataset": "cats", "text_prompt": "cat", "model": "sam2"}) == {
        "unsupported_argument"}
    assert _codes("segment_dataset", {"dataset": "cats"}) == {"missing_argument"}
    assert _codes("segment_dataset", {"dataset": "cats", "text_prompt": "  "}) == {"empty_argument"}


def test_export_is_detection_only():
    props = TOOLS["export_yolo"].parameters_schema()["properties"]
    assert set(props) == {"dataset", "val_split"}
    assert _codes("export_yolo", {"dataset": "warehouse", "task": "segment"}) == {"unsupported_argument"}
    assert _codes("export_yolo", {"dataset": "warehouse", "format": "coco"}) == {"unsupported_argument"}


def test_train_models_are_detection_only():
    assert YOLO_DETECTION_MODELS == ("yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x")
    enum = TOOLS["train_yolo"].parameters_schema()["properties"]["model"]["enum"]
    assert not any(m.endswith("-seg") for m in enum)
    for m in ("yolo11n-seg", "yolov8n", "yolo11z"):
        assert _codes("train_yolo", {"dataset": "parking", "model": m}) == {"invalid_enum_value"}


@pytest.mark.parametrize("tool,args", [
    ("train_yolo", {"dataset": "parking", "epochs": 0}),
    ("train_yolo", {"dataset": "parking", "epochs": 1001}),
    ("train_yolo", {"dataset": "parking", "batch_size": 0}),
    ("train_yolo", {"dataset": "parking", "batch_size": 512}),
    ("train_yolo", {"dataset": "parking", "imgsz": 16}),
    ("train_yolo", {"dataset": "parking", "imgsz": 4096}),
    ("segment_dataset", {"dataset": "cats", "text_prompt": "cat", "confidence": 0}),
    ("segment_dataset", {"dataset": "cats", "text_prompt": "cat", "confidence": 1.5}),
    ("export_yolo", {"dataset": "warehouse", "val_split": 0}),
    ("export_yolo", {"dataset": "warehouse", "val_split": 0.8}),
])
def test_out_of_range_numbers(tool, args):
    assert _codes(tool, args) == {"out_of_range"}


@pytest.mark.parametrize("tool,args", [
    ("train_yolo", {"dataset": "parking", "epochs": 1, "batch_size": 1, "imgsz": 32}),
    ("train_yolo", {"dataset": "parking", "epochs": 1000, "batch_size": 256, "imgsz": 2048}),
    ("export_yolo", {"dataset": "warehouse", "val_split": 0.5}),
])
def test_boundary_values_are_valid(tool, args):
    assert _codes(tool, args) == set()


def test_wrong_type_and_missing():
    assert _codes("train_yolo", {"dataset": "parking", "epochs": "many"}) == {"wrong_type"}
    assert _codes("get_job_status", {}) == {"missing_argument"}
    assert _codes("create_labeling_project", {"dataset": "cats"}) == {"missing_argument"}


def test_normalization_is_format_only(ws):
    assert normalize_arguments("export_yolo", {"dataset": "./data/warehouse/", "val_split": "10%"}, ws) == {
        "dataset": "warehouse", "val_split": 0.1}
    assert normalize_arguments("train_yolo", {"dataset": "parking", "epochs": "80", "imgsz": None}, ws) == {
        "dataset": "parking", "epochs": 80}
    # values are never clamped into range
    assert normalize_arguments("train_yolo", {"dataset": "parking", "epochs": 5000}, ws)["epochs"] == 5000


# ------------------------------------------------------------------ simulated runtime


def test_segment_export_train_state_machine(ws):
    rt = MockToolRuntime(ws)
    seg = rt.call("segment_dataset", {"dataset": "cats", "text_prompt": "cat"})
    assert seg.ok and ws.datasets["cats"].state == "annotated"
    assert seg.output["backend"].startswith("SAM3") and seg.output["text_prompt"] == "cat"
    assert seg.output["class_name"] == "cat" and seg.output["num_images"] == 240
    exp = rt.call("export_yolo", {"dataset": "cats"})
    assert exp.ok and ws.datasets["cats"].state == "yolo"
    assert exp.output["format"] == "yolo-detection" and exp.output["data_yaml"].endswith("data.yaml")
    assert exp.output["num_train"] + exp.output["num_val"] == 240
    tr = rt.call("train_yolo", {"dataset": "cats", "model": "yolo11s", "epochs": 5})
    assert tr.ok and tr.output["job_id"] in ws.jobs and tr.output["status"] == "queued"
    assert "weights" in tr.output["weights_path"]
    assert ws.datasets["cats"].state == "yolo"  # training does not change the dataset state


def test_invalid_state_at_runtime(ws):
    rt = MockToolRuntime(ws)
    assert not rt.call("export_yolo", {"dataset": "cats"}).ok  # raw
    assert not rt.call("train_yolo", {"dataset": "warehouse"}).ok  # annotated
    assert ws.datasets["warehouse"].state == "annotated"


def test_nonexistent_references_at_runtime(ws):
    rt = MockToolRuntime(ws)
    assert not rt.call("train_yolo", {"dataset": "birds"}).ok
    assert not rt.call("get_job_status", {"job_id": "train-9999"}).ok
    assert not rt.call("does_not_exist", {}).ok


def test_job_status_and_labeling(ws):
    rt = MockToolRuntime(ws)
    st = rt.call("get_job_status", {"job_id": "train-0041"})
    assert st.ok and st.output["status"] == "completed" and "metrics" in st.output
    lb = rt.call("create_labeling_project", {"project_name": "qa", "dataset": "cats"})
    assert lb.ok and lb.output["project_id"] in ws.projects


def test_resegmenting_marks_dataset_annotated_again(ws):
    rt = MockToolRuntime(ws)
    assert rt.call("segment_dataset", {"dataset": "parking", "text_prompt": "truck"}).ok
    assert ws.datasets["parking"].state == "annotated"
