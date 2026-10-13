"""Deterministic argument normalization applied to a plan before validation (planned mode).

Only lossless format conversions are performed; values are never clamped or otherwise changed in meaning.
"""
from __future__ import annotations

import re
from typing import Any

from .plan import Plan, PlanStep
from .tools import TOOLS
from .workspace import Workspace, normalize_dataset_ref

_PERCENT = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*%\s*$")
_NUMBER = re.compile(r"^\s*-?\d+(?:\.\d+)?\s*$")


def _field_type(tool: str, key: str) -> type | None:
    spec = TOOLS.get(tool)
    field = spec.args_model.model_fields.get(key) if spec else None
    return field.annotation if field and field.annotation in (int, float) else None


def normalize_value(tool: str, key: str, value: Any, workspace: Workspace) -> Any:
    if key == "dataset" and isinstance(value, str):
        name = normalize_dataset_ref(value)
        return name if name in workspace.datasets else value
    if key == "job_id" and isinstance(value, str):
        return value.strip()
    typ = _field_type(tool, key)
    if typ is float and isinstance(value, str):
        if m := _PERCENT.match(value):
            return float(m.group(1)) / 100
        if _NUMBER.match(value):
            return float(value)
    if typ is int and isinstance(value, str) and _NUMBER.match(value):
        num = float(value)
        return int(num) if num.is_integer() else num
    return value


def normalize_plan(plan: Plan, workspace: Workspace) -> Plan:
    steps = [
        PlanStep(
            tool=s.tool,
            arguments={k: normalize_value(s.tool, k, v, workspace) for k, v in s.arguments.items() if v is not None},
        )
        for s in plan.steps
    ]
    return plan.model_copy(update={"steps": steps})
