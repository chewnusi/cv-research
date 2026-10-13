"""LLM access through LangChain.

LangChain is used only for model interaction: tool binding/tool calling (baseline) and structured
output (planned). Validation, execution, workspace simulation and scoring are plain Python elsewhere.

`LLMClient` is the small interface the orchestrators depend on:
  - `invoke_tools`: one tool-calling turn of a ChatOpenAI model with bound tools
  - `invoke_plan`:  one structured-output turn returning the raw plan object
`LangChainOpenAIClient` implements it with ChatOpenAI; `ScriptedClient` implements it deterministically
for tests (no API key, no network).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Protocol

from langchain_core.messages import AIMessage, BaseMessage

from .config import Settings, StructuredOutput
from .plan import plan_json_schema, strict_plan_json_schema


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict[str, Any] | None  # None if the model produced arguments that are not a JSON object
    raw_args: str | None = None
    error: str | None = None


@dataclass
class ToolTurn:
    message: AIMessage  # appended to the conversation as-is
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0

    @property
    def content(self) -> str:
        c = self.message.content
        return c if isinstance(c, str) else json.dumps(c)


@dataclass
class PlanTurn:
    data: Any  # structured output as returned by the model (normally a dict), None if unparseable
    raw_text: str | None = None
    parse_error: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0


class LLMClient(Protocol):
    model_name: str

    def invoke_tools(self, messages: list[BaseMessage], tools: list[dict]) -> ToolTurn: ...

    def invoke_plan(self, messages: list[BaseMessage], output: StructuredOutput) -> PlanTurn: ...


def _usage(msg: Any) -> tuple[int, int]:
    u = getattr(msg, "usage_metadata", None) or {}
    return int(u.get("input_tokens", 0) or 0), int(u.get("output_tokens", 0) or 0)


def _ordered_tool_calls(ai: AIMessage) -> list[ToolCall]:
    calls = [ToolCall(tc["id"] or f"call_{i}", tc["name"], tc["args"] if isinstance(tc["args"], dict) else None)
             for i, tc in enumerate(ai.tool_calls)]
    calls += [ToolCall(tc.get("id") or f"invalid_{i}", tc.get("name") or "", None, raw_args=tc.get("args"),
                       error=tc.get("error"))
              for i, tc in enumerate(ai.invalid_tool_calls)]
    raw = ai.additional_kwargs.get("tool_calls") or []
    order = {tc.get("id"): i for i, tc in enumerate(raw)}
    return sorted(calls, key=lambda c: order.get(c.id, len(order)))


class LangChainOpenAIClient:
    def __init__(self, settings: Settings, chat_model: Any | None = None):
        if chat_model is None:
            from langchain_openai import ChatOpenAI

            chat_model = ChatOpenAI(
                model=settings.llm_model,
                temperature=settings.llm_temperature,
                seed=settings.llm_seed,
                api_key=settings.llm_api_key,
                base_url=settings.llm_base_url,
                timeout=settings.llm_timeout_s,
                max_retries=settings.llm_max_retries,
                http_socket_options=(),
            )
        self.chat = chat_model
        self.model_name = settings.llm_model
        self._bound: dict[str, Any] = {}

    def invoke_tools(self, messages: list[BaseMessage], tools: list[dict]) -> ToolTurn:
        key = json.dumps(tools, sort_keys=True)
        if key not in self._bound:
            self._bound[key] = self.chat.bind_tools(tools)
        t0 = time.perf_counter()
        ai = self._bound[key].invoke(messages)
        latency = time.perf_counter() - t0
        inp, out = _usage(ai)
        return ToolTurn(ai, _ordered_tool_calls(ai), inp, out, latency)

    def invoke_plan(self, messages: list[BaseMessage], output: StructuredOutput) -> PlanTurn:
        if output == "strict_json_schema":
            runnable = self.chat.with_structured_output(
                strict_plan_json_schema(), method="json_schema", strict=True, include_raw=True)
        elif output == "json_schema":
            runnable = self.chat.with_structured_output(
                plan_json_schema(), method="json_schema", strict=False, include_raw=True)
        else:
            runnable = self.chat.with_structured_output(plan_json_schema(), method="json_mode", include_raw=True)
        t0 = time.perf_counter()
        out = runnable.invoke(messages)
        latency = time.perf_counter() - t0
        raw = out.get("raw")
        inp, outp = _usage(raw)
        err = out.get("parsing_error")
        content = getattr(raw, "content", None)
        return PlanTurn(out.get("parsed"), content if isinstance(content, str) else None,
                        None if err is None else f"{type(err).__name__}: {err}", inp, outp, latency)


# --------------------------------------------------------------------------- deterministic test double

Kind = Literal["tools", "plan"]


class ScriptedClient:
    """Returns pre-defined turns in order, or computes them from the conversation with a callable
    `fn(messages, kind) -> ToolTurn | PlanTurn`. Records every request."""

    def __init__(self, script: list[ToolTurn | PlanTurn] | Callable[[list[BaseMessage], Kind], ToolTurn | PlanTurn]):
        self.model_name = "scripted"
        self._script = script
        self.requests: list[tuple[Kind, list[BaseMessage]]] = []

    def _next(self, messages: list[BaseMessage], kind: Kind):
        self.requests.append((kind, list(messages)))
        if callable(self._script):
            return self._script(messages, kind)
        if not self._script:
            raise RuntimeError("ScriptedClient ran out of responses")
        return self._script.pop(0)

    def invoke_tools(self, messages, tools) -> ToolTurn:
        turn = self._next(messages, "tools")
        assert isinstance(turn, ToolTurn), "scripted tool turn expected"
        return turn

    def invoke_plan(self, messages, output) -> PlanTurn:
        turn = self._next(messages, "plan")
        assert isinstance(turn, PlanTurn), "scripted plan turn expected"
        return turn


def tool_turn(*calls: tuple[str, dict], content: str = "") -> ToolTurn:
    """Scripted assistant turn that calls the given tools."""
    lc_calls = [{"name": n, "args": a, "id": f"call_{i}", "type": "tool_call"} for i, (n, a) in enumerate(calls)]
    return ToolTurn(AIMessage(content=content, tool_calls=lc_calls),
                    [ToolCall(c["id"], c["name"], c["args"]) for c in lc_calls])


def text_turn(text: str) -> ToolTurn:
    """Scripted assistant turn with a text reply and no tool calls."""
    return ToolTurn(AIMessage(content=text))


def plan_turn(status: str = "ok", steps: list[tuple[str, dict]] | tuple = (), message: str = "") -> PlanTurn:
    """Scripted planner turn."""
    data = {"status": status, "message": message, "steps": [{"tool": t, "arguments": a} for t, a in steps]}
    return PlanTurn(data, raw_text=json.dumps(data))


def make_client(settings: Settings) -> LLMClient:
    return LangChainOpenAIClient(settings)
