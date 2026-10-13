"""The LangChain/OpenAI adapter, exercised offline: ChatOpenAI talks to an httpx MockTransport."""
import json

import httpx
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from conftest import make_settings
from cvorch.llm import LangChainOpenAIClient
from cvorch.orchestrators import BaselineOrchestrator, PlannedOrchestrator
from cvorch.tools import openai_tools


def _completion(message: dict) -> dict:
    return {"id": "x", "object": "chat.completion", "created": 0, "model": "gpt-test",
            "choices": [{"index": 0, "message": {"role": "assistant", **message}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}}


def _client(responses: list[dict], requests: list[dict], settings=None):
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=_completion(responses.pop(0)))

    settings = settings or make_settings(llm_temperature=0.0, llm_seed=42)
    chat = ChatOpenAI(model="gpt-test", api_key="test", temperature=settings.llm_temperature, seed=settings.llm_seed,
                      max_retries=0, http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    return LangChainOpenAIClient(settings, chat_model=chat)


def test_tool_calling_turn():
    reqs: list[dict] = []
    llm = _client([{"content": None, "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "train_yolo", "arguments": '{"dataset": "parking"}'}},
        {"id": "c2", "type": "function", "function": {"name": "get_job_status", "arguments": "{bad"}},
    ]}], reqs)
    turn = llm.invoke_tools([SystemMessage("s"), HumanMessage("u")], openai_tools())
    assert [(c.id, c.name, c.args) for c in turn.tool_calls] == [("c1", "train_yolo", {"dataset": "parking"}),
                                                                ("c2", "get_job_status", None)]
    assert (turn.input_tokens, turn.output_tokens) == (11, 7)
    body = reqs[0]
    assert {t["function"]["name"] for t in body["tools"]} == {t["function"]["name"] for t in openai_tools()}
    assert body["temperature"] == 0.0 and body["seed"] == 42


def test_plan_turn_uses_non_strict_json_schema_by_default():
    reqs: list[dict] = []
    plan = {"status": "ok", "message": "", "steps": [{"tool": "train_yolo", "arguments": {"dataset": "parking"}}]}
    llm = _client([{"content": json.dumps(plan)}], reqs)
    turn = llm.invoke_plan([HumanMessage("u")], "json_schema")
    assert turn.data == plan and turn.parse_error is None
    rf = reqs[0]["response_format"]
    assert rf["type"] == "json_schema" and rf["json_schema"].get("strict") is False


def test_plan_turn_strict_ablation_sends_strict_schema():
    reqs: list[dict] = []
    llm = _client([{"content": json.dumps({"status": "unsupported", "message": "no", "steps": []})}], reqs)
    llm.invoke_plan([HumanMessage("u")], "strict_json_schema")
    assert reqs[0]["response_format"]["json_schema"]["strict"] is True


def test_unparseable_plan_is_reported():
    llm = _client([{"content": "this is not json"}], [])
    turn = llm.invoke_plan([HumanMessage("u")], "json_schema")
    assert turn.data is None and turn.parse_error


def test_end_to_end_with_langchain_adapter(ws):
    reqs: list[dict] = []
    llm = _client([
        {"content": None, "tool_calls": [{"id": "c1", "type": "function", "function": {
            "name": "train_yolo", "arguments": '{"dataset": "parking", "epochs": 5}'}}]},
        {"content": "started"},
    ], reqs)
    res = BaselineOrchestrator(llm, make_settings("baseline")).run("train parking 5 epochs", ws)
    assert [c.tool for c in res.executed] == ["train_yolo"] and res.final_message == "started"
    assert reqs[1]["messages"][-1]["role"] == "tool" and reqs[1]["messages"][-1]["tool_call_id"] == "c1"
    assert res.input_tokens == 22

    plan = {"status": "ok", "message": "", "steps": [{"tool": "get_job_status", "arguments": {"job_id": "train-0041"}}]}
    res = PlannedOrchestrator(_client([{"content": json.dumps(plan)}], []), make_settings()).run("status", ws)
    assert [c.tool for c in res.executed] == ["get_job_status"]
