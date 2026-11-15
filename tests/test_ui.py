"""Streamlit UI flow with a scripted LLM (no API)."""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from cvorch.llm import ScriptedClient, plan_turn, text_turn, tool_turn  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "src" / "cvorch" / "ui" / "app.py")
SEG = ("segment_dataset", {"dataset": "cats", "text_prompt": "cat"})
EXP = ("export_yolo", {"dataset": "cats"})
BAD = ("train_yolo", {"dataset": "cats", "epochs": 0})


def _app(monkeypatch, responder):
    monkeypatch.setattr("cvorch.llm.make_client", lambda settings: ScriptedClient(responder))
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    return at


def _button(at, label):
    found = [b for b in at.button if b.label == label]
    return found[0] if found else None


def _submit(at, text):
    at.text_area[0].input(text)
    at.run()
    _button(at, "Submit").click()
    at.run()


def test_planned_requires_confirmation(monkeypatch):
    at = _app(monkeypatch, lambda m, kind: plan_turn("ok", [SEG, EXP]))
    _submit(at, 'segment cats for "cat" and export them')
    assert not at.exception
    assert at.session_state.ws.datasets["cats"].state == "raw"  # nothing executed yet
    _button(at, "Execute plan").click()
    at.run()
    assert at.session_state.ws.datasets["cats"].state == "yolo"


def test_invalid_plan_has_no_execute_button(monkeypatch):
    at = _app(monkeypatch, lambda m, kind: plan_turn("ok", [SEG, EXP, BAD]))
    _submit(at, "late failure")
    assert _button(at, "Execute plan") is None
    assert any("blocked" in e.value for e in at.error)
    assert at.session_state.ws.datasets["cats"].state == "raw"


def test_baseline_executes_directly(monkeypatch):
    def responder(messages, kind):
        return text_turn("done") if messages[-1].type == "tool" else tool_turn(SEG, EXP, BAD)
    at = _app(monkeypatch, responder)
    at.sidebar.selectbox[0].select("baseline")
    at.run()
    _submit(at, "late failure")
    assert not at.exception
    assert at.session_state.ws.datasets["cats"].state == "yolo"  # partial side effects are visible
    assert any("cats: raw -> yolo" in md.value for md in at.markdown)


def test_cancel_button_stops_a_running_job(monkeypatch):
    at = _app(monkeypatch, lambda m, kind: plan_turn("ok", []))
    buttons = [b for b in at.button if b.key == "cancel-train-0042"]
    assert len(buttons) == 1 and not [b for b in at.button if b.key == "cancel-train-0041"]
    buttons[0].click()
    at.run()
    assert not at.exception
    assert at.session_state.ws.jobs["train-0042"].status == "cancelled"
    assert not [b for b in at.button if b.key == "cancel-train-0042"]


def test_compare_tab_runs_both_strategies_on_separate_workspaces(monkeypatch):
    def responder(messages, kind):
        if kind == "plan":
            return plan_turn("ok", [SEG, EXP, BAD])
        return text_turn("failed") if messages[-1].type == "tool" else tool_turn(SEG, EXP, BAD)
    at = _app(monkeypatch, responder)
    at.text_area(key="compare_prompt").input('Segment cats for "cat", export to YOLO and train for 0 epochs.')
    at.run()
    at.button(key="compare_run").click()
    at.run()
    assert not at.exception
    cmp = at.session_state.compare
    assert [c.tool for c in cmp["baseline"]["res"].executed] == ["segment_dataset", "export_yolo"]
    assert cmp["planned"]["res"].blocked_by_validator and cmp["planned"]["res"].calls == []
    assert cmp["planned"]["after"]["datasets"]["cats"]["state"] == "raw"
    assert at.session_state.ws.datasets["cats"].state == "raw"  # the Demo workspace is untouched


def test_attempt_history_shows_rejection_and_repair():
    import importlib.util
    from conftest import make_settings
    from cvorch.orchestrators import PlannedOrchestrator
    from cvorch.workspace import default_workspace
    spec = importlib.util.spec_from_file_location("app", APP)
    app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app)
    sub = plan_turn("ok", [("train_yolo", {"dataset": "parking", "model": "yolo11n"})])
    llm = ScriptedClient([sub, plan_turn("unsupported", message="no such model")])
    res = PlannedOrchestrator(llm, make_settings()).run("Train yolov8n on parking.", default_workspace())
    rows = app.attempt_history(res)
    assert [r["plan status"] for r in rows] == ["ok", "unsupported"]
    assert rows[0]["validator decision"].startswith("rejected: [request_value_not_used]")
    assert app.validator_summary(res) == "1 plan(s) rejected; final: refused by planner"
