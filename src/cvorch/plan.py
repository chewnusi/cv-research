"""The structured execution plan produced by the LLM in the proposed method.

The LLM decides *what* workflow is requested; deterministic code (validator.py) decides whether it is valid.
Steps are executed sequentially, so no explicit dependency graph is needed.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from .tools import TOOLS

PlanStatus = Literal["ok", "unsupported", "needs_clarification"]


class PlanStep(BaseModel):
    """One tool invocation of the workflow."""
    tool: str = Field(description="Name of one of the available tools.")
    arguments: dict[str, Any] = Field(default_factory=dict, description="Tool arguments as a JSON object.")

    @field_validator("arguments", mode="before")
    @classmethod
    def _pairs_to_dict(cls, v: Any) -> Any:
        """The strict structured-output ablation encodes arguments as [{"name": ..., "value": ...}]."""
        if isinstance(v, list) and all(isinstance(p, dict) and "name" in p for p in v):
            return {p["name"]: p.get("value") for p in v}
        return v


class ExecutionPlan(BaseModel):
    """The complete intended workflow for a request, produced before anything is executed."""
    status: PlanStatus = Field(description="'ok' if the request can be fulfilled with the tools, otherwise "
                                           "'unsupported' or 'needs_clarification'.")
    message: str = Field("", description="Short explanation for the user.")
    steps: list[PlanStep] = Field(default_factory=list, description="Steps in execution order; empty unless status is 'ok'.")


def plan_json_schema() -> dict:
    """Schema sent to the provider for the primary condition (non-strict guidance; we parse and validate)."""
    return ExecutionPlan.model_json_schema()


_VALUE = {"anyOf": [
    {"type": "string"}, {"type": "number"}, {"type": "boolean"},
    {"type": "array", "items": {"type": "string"}},
]}


def strict_plan_json_schema() -> dict:
    """Strict JSON schema for the provider-side constrained-decoding ablation.

    The decoder enforces structure only: valid tool names and, per tool, valid argument names.
    Arguments are name/value pairs so unspecified optional arguments can simply be left out
    (strict mode would otherwise force a value for every property). Value constraints stay with the
    validator: if the decoder enforced them, "-10 epochs" would be silently coerced to a valid value.
    """
    step_variants = [
        {
            "type": "object",
            "properties": {
                "tool": {"type": "string", "enum": [name]},
                "arguments": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "enum": list(spec.args_model.model_fields)},
                            "value": _VALUE,
                        },
                        "required": ["name", "value"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["tool", "arguments"],
            "additionalProperties": False,
        }
        for name, spec in TOOLS.items()
    ]
    return {
        "title": "ExecutionPlan",
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["ok", "unsupported", "needs_clarification"]},
            "message": {"type": "string"},
            "steps": {"type": "array", "items": {"anyOf": step_variants}},
        },
        "required": ["status", "message", "steps"],
        "additionalProperties": False,
    }
