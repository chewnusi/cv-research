"""The two orchestration approaches under comparison.

A: BaselineOrchestrator - LLM -> direct tool calling. Every call is executed immediately; its result is
   fed back to the LLM, which continues the workflow. No global validation, so a late failure can leave
   earlier steps already executed (partial side effects).
B: PlannedOrchestrator  - LLM -> complete structured ExecutionPlan -> deterministic full-plan validation
   -> execution. Any issue blocks the whole plan; nothing executes.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import ValidationError

from .config import Settings
from .llm import LLMClient, PlanTurn, ToolTurn
from .normalize import normalize_plan
from .plan import ExecutionPlan
from .prompts import baseline_system_prompt, planner_system_prompt, repair_prompt
from .tools import MockToolRuntime, ToolCallRecord, openai_tools
from .validator import complete_prerequisites, validate_plan
from .workspace import Workspace


@dataclass
class RunResult:
    mode: str
    profile: str
    prompt: str
    # Every tool call that reached the tool runtime (successful or not), in order.
    calls: list[ToolCallRecord] = field(default_factory=list)
    final_message: str = ""
    # planned-mode details
    plan: dict | None = None
    plan_status: str | None = None
    plan_attempts: list[dict] = field(default_factory=list)
    plan_parse_error: bool = False
    inserted_prerequisites: int = 0
    validation_issues: list[dict] = field(default_factory=list)
    blocked_by_validator: bool = False
    # bookkeeping
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    llm_latency_s: float = 0.0
    total_latency_s: float = 0.0
    error: str | None = None

    @property
    def executed(self) -> list[ToolCallRecord]:
        return [c for c in self.calls if c.ok]

    @property
    def executable(self) -> bool:
        """Planned mode: the plan may be executed (parsed, status ok, not blocked)."""
        return (self.plan is not None and self.plan_status == "ok" and not self.blocked_by_validator
                and not self.plan_parse_error and self.error is None)

    def account(self, turn: ToolTurn | PlanTurn) -> None:
        self.llm_calls += 1
        self.input_tokens += turn.input_tokens
        self.output_tokens += turn.output_tokens
        self.llm_latency_s += turn.latency_s

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "profile": self.profile,
            "prompt": self.prompt,
            "calls": [c.to_dict() for c in self.calls],
            "final_message": self.final_message,
            "plan": self.plan,
            "plan_status": self.plan_status,
            "plan_attempts": self.plan_attempts,
            "plan_parse_error": self.plan_parse_error,
            "inserted_prerequisites": self.inserted_prerequisites,
            "validation_issues": self.validation_issues,
            "blocked_by_validator": self.blocked_by_validator,
            "llm_calls": self.llm_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "llm_latency_s": round(self.llm_latency_s, 4),
            "total_latency_s": round(self.total_latency_s, 4),
            "error": self.error,
        }


OnCall = Callable[[ToolCallRecord], None]


class BaselineOrchestrator:
    mode = "baseline"

    def __init__(self, llm: LLMClient, settings: Settings):
        self.llm = llm
        self.settings = settings

    def run(self, prompt: str, workspace: Workspace, on_call: OnCall | None = None) -> RunResult:
        res = RunResult(self.mode, self.settings.profile, prompt)
        runtime = MockToolRuntime(workspace, arg_validation=self.settings.tool_arg_validation)
        messages: list[BaseMessage] = [SystemMessage(baseline_system_prompt(workspace)), HumanMessage(prompt)]
        tools = openai_tools()
        t0 = time.perf_counter()
        try:
            for _ in range(self.settings.baseline_max_rounds):
                turn = self.llm.invoke_tools(messages, tools)
                res.account(turn)
                messages.append(turn.message)
                if not turn.tool_calls:
                    res.final_message = turn.content
                    break
                for tc in turn.tool_calls:
                    if tc.args is None:
                        rec = ToolCallRecord(tc.name, {"_raw": tc.raw_args}, False,
                                             error="arguments are not a valid JSON object")
                        runtime.calls.append(rec)
                    else:
                        rec = runtime.call(tc.name, tc.args)
                    if on_call:
                        on_call(rec)
                    payload = rec.output if rec.ok else {"error": rec.error}
                    messages.append(ToolMessage(json.dumps(payload), tool_call_id=tc.id))
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

    def _parse(self, turn: PlanTurn) -> tuple[ExecutionPlan | None, str | None]:
        if turn.data is None:
            return None, turn.parse_error or "no structured output returned"
        try:
            return ExecutionPlan.model_validate(turn.data), None
        except ValidationError as e:
            first = e.errors()[0]
            return None, f"{'.'.join(map(str, first['loc']))}: {first['msg']}"

    def prepare(self, prompt: str, workspace: Workspace) -> RunResult:
        """LLM -> plan -> normalization -> (ablation: prerequisite insertion) -> full-plan validation.
        Nothing is executed."""
        res = RunResult(self.mode, self.settings.profile, prompt)
        s = self.settings
        messages: list[BaseMessage] = [
            SystemMessage(planner_system_prompt(workspace, arguments_as_pairs=s.planner_structured_output
                                                == "strict_json_schema",
                                                auto_prerequisites=s.planner_auto_prerequisites)),
            HumanMessage(prompt),
        ]
        t0 = time.perf_counter()
        try:
            for attempt in range(s.planner_repair_attempts + 1):
                turn = self.llm.invoke_plan(messages, s.planner_structured_output)
                res.account(turn)
                log: dict[str, Any] = {"raw": turn.data if turn.data is not None else turn.raw_text}
                res.plan_attempts.append(log)
                plan, parse_error = self._parse(turn)

                if plan is None:
                    res.plan, res.plan_status, res.plan_parse_error = None, "parse_error", True
                    res.validation_issues, res.blocked_by_validator = [], False
                    issues = [f"[plan_parse_error] plan: {parse_error}"]
                    log["parse_error"] = parse_error
                else:
                    plan = normalize_plan(plan, workspace)
                    if s.planner_auto_prerequisites:
                        plan, inserted = complete_prerequisites(plan, workspace)
                        res.inserted_prerequisites = inserted
                        log["inserted_prerequisites"] = inserted
                    res.plan, res.plan_status, res.plan_parse_error = plan.model_dump(), plan.status, False
                    res.final_message = plan.message
                    log["plan"] = res.plan
                    if not s.planner_validate:
                        break
                    vr = validate_plan(plan, workspace)
                    res.validation_issues = vr.to_dicts()
                    res.blocked_by_validator = not vr.ok
                    if vr.ok:
                        break
                    issues = [str(i) for i in vr.issues]
                log["issues"] = issues
                if attempt < s.planner_repair_attempts:
                    raw = log["raw"] if isinstance(log["raw"], str) else json.dumps(log["raw"])
                    messages += [AIMessage(raw or ""), HumanMessage(repair_prompt(issues))]
        except Exception as e:  # noqa: BLE001
            res.error = f"{type(e).__name__}: {e}"
        res.total_latency_s = time.perf_counter() - t0
        return res

    def execute(self, res: RunResult, workspace: Workspace, on_call: OnCall | None = None) -> RunResult:
        """Execute a prepared plan. With validation on, the plan is re-validated against the current
        workspace first; any issue blocks the whole plan and nothing is executed."""
        if not res.executable:
            return res
        plan = ExecutionPlan.model_validate(res.plan)
        t0 = time.perf_counter()
        if self.settings.planner_validate:
            vr = validate_plan(plan, workspace)
            if not vr.ok:
                res.validation_issues, res.blocked_by_validator = vr.to_dicts(), True
                return res
        runtime = MockToolRuntime(workspace, arg_validation=self.settings.tool_arg_validation)
        for step in plan.steps:
            rec = runtime.call(step.tool, step.arguments)
            if on_call:
                on_call(rec)
            if not rec.ok:
                break
        res.calls = runtime.calls
        res.total_latency_s += time.perf_counter() - t0
        return res

    def run(self, prompt: str, workspace: Workspace, on_call: OnCall | None = None) -> RunResult:
        return self.execute(self.prepare(prompt, workspace), workspace, on_call)


def make_orchestrator(llm: LLMClient, settings: Settings):
    return BaselineOrchestrator(llm, settings) if settings.mode == "baseline" else PlannedOrchestrator(llm, settings)
