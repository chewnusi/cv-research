from cvorch.tools import TOOLS, MockToolRuntime, openai_tools
from cvorch.workspace import default_workspace, normalize_dataset_ref


def test_normalize_dataset_ref():
    for ref in ("cats", "./data/cats/", "data/cats", "datasets/cats", "Cats", "data/cats/annotations"):
        assert normalize_dataset_ref(ref) == "cats"


def test_runtime_state_transitions():
    ws = default_workspace()
    rt = MockToolRuntime(ws)
    assert rt.call("segment_dataset", {"dataset": "cats"}).ok
    assert ws.datasets["cats"].state == "annotated"
    assert rt.call("export_yolo", {"dataset": "cats"}).ok
    assert ws.datasets["cats"].state == "yolo"
    rec = rt.call("train_yolo", {"dataset": "cats", "epochs": 10})
    assert rec.ok and rec.output["job_id"] in ws.jobs


def test_runtime_preconditions_fail_at_call_time():
    rt = MockToolRuntime(default_workspace())
    assert not rt.call("train_yolo", {"dataset": "dogs"}).ok
    assert not rt.call("export_yolo", {"dataset": "dogs"}).ok
    assert not rt.call("segment_dataset", {"dataset": "birds"}).ok
    assert not rt.call("get_job_status", {"job_id": "nope"}).ok
    assert not rt.call("delete_dataset", {"dataset": "cats"}).ok


def test_arg_validation_flag():
    strict = MockToolRuntime(default_workspace(), arg_validation=True)
    assert not strict.call("train_yolo", {"dataset": "parking", "epochs": -10}).ok
    loose = MockToolRuntime(default_workspace(), arg_validation=False)
    assert loose.call("train_yolo", {"dataset": "parking", "epochs": -10}).ok


def test_openai_tool_definitions():
    defs = openai_tools()
    assert {d["function"]["name"] for d in defs} == set(TOOLS)
    train = next(d for d in defs if d["function"]["name"] == "train_yolo")["function"]
    assert train["parameters"]["required"] == ["dataset"]
    assert train["parameters"]["properties"]["epochs"]["minimum"] == 1
