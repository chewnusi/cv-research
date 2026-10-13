import json

import pytest

from cvorch.config import Settings
from cvorch.llm import LLMResponse, ToolCall


@pytest.fixture
def settings():
    return Settings(mode="planned", llm_model="scripted")


def plan_response(status="ok", steps=(), message=""):
    return LLMResponse(content=json.dumps({
        "status": status, "message": message,
        "steps": [{"tool": t, "arguments": a} for t, a in steps],
    }))


def tool_calls_response(*calls):
    return LLMResponse(content=None, tool_calls=[
        ToolCall(id=f"call_{i}", name=name, arguments=json.dumps(args)) for i, (name, args) in enumerate(calls)
    ])


def text_response(text):
    return LLMResponse(content=text)
