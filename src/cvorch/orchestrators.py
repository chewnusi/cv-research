"""The two orchestration approaches under comparison.

A: BaselineOrchestrator  - LLM -> direct tool calling (each call is executed immediately)
B: PlannedOrchestrator   - LLM -> structured plan -> deterministic validation -> execution
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from .config import Settings
from .llm import LLMClient, LLMResponse
from .normalize import normalize_plan
from .plan import Plan, plan_json_schema
from .prompts import baseline_system_prompt, planner_system_prompt, repair_prompt
from .tools import MockToolRuntime, ToolCallRecord, openai_tools
from .validator import complete_prerequisites, validate_plan
from .workspace import Workspace


@dataclass
class RunResult:
    mode: str
    prompt: str
    # Every tool call that reached the tool runtime (successful or not), in order.
    calls: list[ToolCallRecord] = field(default_factory=list)
    final_message: str = ""
    # planned-mode details
    plan: dict | None = None
    plan_status: str | None = None
    plan_attempts: list[dict] = field(default_factory=list)
    validation_issues: list[dict] = field(default_factory=list)
    blocked_by_validator: bool = False
    # bookkeeping
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_latency_s: float = 0.0
    total_latency_s: float = 0.0
    error: str | None = None

    @property
    def executed(self) -> list[ToolCallRecord]:
        return [c for c in self.calls if c.ok]

    def account(self, r: LLMResponse) -> None:
        self.llm_calls += 1
        self.prompt_tokens += r.prompt_tokens
        self.completion_tokens += r.completion_tokens
        self.llm_latency_s += r.latency_s

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "prompt": self.prompt,
            "calls": [c.to_dict() for c in self.calls],
            "final_message": self.final_message,
            "plan": self.plan,
            "plan_status": self.plan_status,
            "plan_attempts": self.plan_attempts,
            "validation_issues": self.validation_issues,
            "blocked_by_validator": self.blocked_by_validator,
            "llm_calls": self.llm_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "llm_latency_s": round(self.llm_latency_s, 4),
            "total_latency_s": round(self.total_latency_s, 4),
            "error": self.error,
        }


def _parse_json_args(raw: str) -> dict[str, Any] | None:
    try:
        val = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        return None
    return val if isinstance(val, dict) else None


class BaselineOrchestrator:
    mode = "baseline"

    def __init__(self, llm: LLMClient, settings: Settings):
        self.llm = llm
        self.settings = settings

    def run(self, prompt: str, workspace: Workspace) -> RunResult:
        res = RunResult(self.mode, prompt)
        runtime = MockToolRuntime(workspace, arg_validation=self.settings.tool_arg_validation)
        messages: list[dict] = [
            {"role": "system", "content": baseline_system_prompt(workspace)},
            {"role": "user", "content": prompt},
        ]
        tools = openai_tools()
        t0 = time.perf_counter()
        try:
            for _ in range(self.settings.baseline_max_rounds):
                r = self.llm.chat(messages, tools=tools)
                res.account(r)
                messages.append(r.assistant_message())
                if not r.tool_calls:
                    res.final_message = r.content or ""
                    break
                for tc in r.tool_calls:
                    args = _parse_json_args(tc.arguments)
                    if args is None:
                        rec = ToolCallRecord(tc.name, {"_raw": tc.arguments}, False, error="arguments are not valid JSON")
                        runtime.calls.append(rec)
                    else:
                        rec = runtime.call(tc.name, args)
                    payload = rec.output if rec.ok else {"error": rec.error}
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(payload)})
            else:
                res.error = "max_rounds_exceeded"
        except Exception as e:  # noqa: BLE001 - recorded for the benchmark
            res.error = f"{type(e).__name__}: {e}"
        res.calls = runtime.calls
        res.total_latency_s = time.perf_counter() - t0
        return res


class PlannedOrchestrator:
    mode = "planned"

    def __init__(self, llm: LLMClient, settings: Settings):
        self.llm = llm
        self.settings = settings

    def _response_format(self) -> dict:
        if self.settings.planner_response_format == "json_schema":
            return {"type": "json_schema",
                    "json_schema": {"name": "plan", "schema": plan_json_schema(), "strict": True}}
        return {"type": "json_object"}

    def make_plan(self, prompt: str, workspace: Workspace, res: RunResult) -> Plan | None:
        """LLM -> plan, with optional validator-feedback repair rounds. Fills validation info in `res`."""
        messages: list[dict] = [
            {"role": "system", "content": planner_system_prompt(
                workspace,
                arguments_as_pairs=self.settings.planner_response_format == "json_schema",
                auto_prerequisites=self.settings.planner_auto_prerequisites,
            )},
            {"role": "user", "content": prompt},
        ]
        plan: Plan | None = None
        for attempt in range(self.settings.planner_repair_attempts + 1):
            r = self.llm.chat(messages, response_format=self._response_format())
            res.account(r)
            messages.append({"role": "assistant", "content": r.content or ""})
            attempt_log: dict[str, Any] = {"raw": r.content}
            res.plan_attempts.append(attempt_log)

            try:
                plan = normalize_plan(Plan.model_validate_json(_strip_fences(r.content or "")), workspace)
            except ValidationError as e:
                plan = None
                issues = [f"[plan_parse_error] plan: response is not a valid plan object: {e.errors()[0]['msg']}"]
                res.validation_issues = [{"step": None, "code": "plan_parse_error", "message": issues[0]}]
            else:
                if self.settings.planner_auto_prerequisites:
                    plan, inserted = complete_prerequisites(plan, workspace)
                    attempt_log["inserted_prerequisites"] = inserted
                attempt_log["plan"] = plan.model_dump()
                if not self.settings.planner_validate:
                    res.validation_issues = []
                    return plan
                vr = validate_plan(plan, workspace)
                res.validation_issues = [i.to_dict() for i in vr.issues]
                if vr.ok:
                    return plan
                issues = [str(i) for i in vr.issues]
            attempt_log["issues"] = issues
            messages.append({"role": "user", "content": repair_prompt(issues)})
        return plan

    def run(self, prompt: str, workspace: Workspace) -> RunResult:
        res = RunResult(self.mode, prompt)
        runtime = MockToolRuntime(workspace, arg_validation=self.settings.tool_arg_validation)
        t0 = time.perf_counter()
        try:
            plan = self.make_plan(prompt, workspace, res)
            if plan is not None:
                res.plan = plan.model_dump()
                res.plan_status = plan.status
                res.final_message = plan.message
            if plan is None or res.validation_issues:
                res.blocked_by_validator = True
                res.plan_status = res.plan_status or "invalid"
            elif plan.status == "ok":
                for step in plan.steps:
                    rec = runtime.call(step.tool, step.arguments)
                    if not rec.ok:
                        break
        except Exception as e:  # noqa: BLE001
            res.error = f"{type(e).__name__}: {e}"
        res.calls = runtime.calls
        res.total_latency_s = time.perf_counter() - t0
        return res


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else ""
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    return t.strip()


def make_orchestrator(llm: LLMClient, settings: Settings):
    return BaselineOrchestrator(llm, settings) if settings.mode == "baseline" else PlannedOrchestrator(llm, settings)
