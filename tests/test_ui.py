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
    _submit(at, "segment and export cats")
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
