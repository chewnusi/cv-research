"""Plan-level argument normalization (planned mode), using the same contract normalization as the tool runtime."""
from __future__ import annotations

from .plan import ExecutionPlan, PlanStep
from .tools import normalize_arguments
from .workspace import Workspace


def normalize_plan(plan: ExecutionPlan, workspace: Workspace) -> ExecutionPlan:
    steps = [PlanStep(tool=s.tool, arguments=normalize_arguments(s.tool, s.arguments, workspace)) for s in plan.steps]
    return plan.model_copy(update={"steps": steps})
