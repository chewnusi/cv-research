from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from .config import Settings


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # raw JSON string as produced by the model


@dataclass
class LLMResponse:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0

    def assistant_message(self) -> dict:
        msg: dict[str, Any] = {"role": "assistant", "content": self.content or ""}
        if self.tool_calls:
            msg["tool_calls"] = [
                {"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": tc.arguments}}
                for tc in self.tool_calls
            ]
        return msg


class LLMClient(Protocol):
    model: str

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        response_format: dict | None = None,
    ) -> LLMResponse: ...


class OpenAICompatibleClient:
    def __init__(self, settings: Settings, max_retries: int = 3):
        from openai import OpenAI

        self.model = settings.llm_model
        self.temperature = settings.llm_temperature
        self.seed = settings.llm_seed
        self.max_retries = max_retries
        self._client = OpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key or "not-needed",
            timeout=settings.llm_timeout_s,
            max_retries=0,
        )

    def chat(self, messages, tools=None, response_format=None) -> LLMResponse:
        kwargs: dict[str, Any] = {"model": self.model, "messages": messages, "temperature": self.temperature}
        if self.seed is not None:
            kwargs["seed"] = self.seed
        if tools:
            kwargs["tools"] = tools
        if response_format:
            kwargs["response_format"] = response_format

        from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError

        for attempt in range(self.max_retries + 1):
            t0 = time.perf_counter()
            try:
                resp = self._client.chat.completions.create(**kwargs)
                break
            except (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError):
                if attempt == self.max_retries:
                    raise
                time.sleep(2 ** attempt)
        latency = time.perf_counter() - t0

        msg = resp.choices[0].message
        calls = [
            ToolCall(tc.id, tc.function.name, tc.function.arguments or "{}")
            for tc in (msg.tool_calls or [])
        ]
        usage = resp.usage
        return LLMResponse(
            content=msg.content,
            tool_calls=calls,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            latency_s=latency,
        )


class ScriptedClient:
    """Deterministic stand-in for tests: returns pre-defined responses in order,
    or computes them from the conversation with a callable."""

    def __init__(self, script: list[LLMResponse] | Callable[[list[dict], list[dict] | None], LLMResponse]):
        self.model = "scripted"
        self._script = script
        self.requests: list[list[dict]] = []

    def chat(self, messages, tools=None, response_format=None) -> LLMResponse:
        self.requests.append(json.loads(json.dumps(messages)))
        if callable(self._script):
            return self._script(messages, tools)
        if not self._script:
            raise RuntimeError("ScriptedClient ran out of responses")
        return self._script.pop(0)


def make_client(settings: Settings) -> LLMClient:
    return OpenAICompatibleClient(settings)
