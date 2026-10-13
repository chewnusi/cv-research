from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from .tools import TOOLS

PlanStatus = Literal["ok", "unsupported", "needs_clarification"]


class PlanStep(BaseModel):
    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)

    @field_validator("arguments", mode="before")
    @classmethod
    def _pairs_to_dict(cls, v: Any) -> Any:
        """Strict structured output encodes arguments as [{"name": ..., "value": ...}]."""
        if isinstance(v, list) and all(isinstance(p, dict) and "name" in p for p in v):
            return {p["name"]: p.get("value") for p in v}
        return v


class Plan(BaseModel):
    status: PlanStatus
    message: str = ""
    steps: list[PlanStep] = Field(default_factory=list)


_VALUE = {"anyOf": [
    {"type": "string"}, {"type": "number"}, {"type": "boolean"},
    {"type": "array", "items": {"type": "string"}},
]}


def plan_json_schema() -> dict:
    """Strict JSON schema for provider-side structured output (PLANNER_RESPONSE_FORMAT=json_schema).

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
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["ok", "unsupported", "needs_clarification"]},
            "message": {"type": "string"},
            "steps": {"type": "array", "items": {"anyOf": step_variants}},
        },
        "required": ["status", "message", "steps"],
        "additionalProperties": False,
    }
