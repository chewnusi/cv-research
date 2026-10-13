"""Deterministic plan validation (no LLM involved).

Rules, applied per step while simulating dataset state across the plan:
  R1 unknown_tool          tool must exist in the registry
  R2 invalid_arguments     arguments must satisfy the tool schema (required, types, ranges, enums, no extras)
  R3 dataset_not_found     referenced dataset must exist in the workspace
  R4 job_not_found         referenced job must exist
  R5 dependency_violation  step precondition must hold given the simulated state after previous steps
                           (export needs an annotated dataset, training needs a YOLO-format dataset)
  R6 empty_plan            status 'ok' must contain at least one step
"""
from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import ValidationError

from .plan import Plan, PlanStep
from .tools import TOOLS, format_validation_error
from .workspace import Workspace, normalize_dataset_ref


@dataclass
class ValidationIssue:
    step: int | None  # 1-based, None for plan-level issues
    code: str
    message: str

    def to_dict(self) -> dict:
        return {"step": self.step, "code": self.code, "message": self.message}

    def __str__(self) -> str:
        where = f"step {self.step}" if self.step is not None else "plan"
        return f"[{self.code}] {where}: {self.message}"


@dataclass
class ValidationResult:
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.issues


_REQUIRED_STATE = {"export_yolo": {"annotated", "yolo"}, "train_yolo": {"yolo"}}
_STATE_AFTER = {"segment_dataset": "annotated", "export_yolo": "yolo"}
_STATE_RANK = {"raw": 0, "annotated": 1, "yolo": 2}


_PREREQUISITES = {"raw": ["segment_dataset", "export_yolo"], "annotated": ["export_yolo"]}


def complete_prerequisites(plan: Plan, workspace: Workspace) -> tuple[Plan, int]:
    """Insert missing segment/export steps before steps whose dataset is not yet in the required state.

    Deterministic dependency resolution: the LLM only has to choose the goal steps. Steps that reference
    unknown tools or datasets are left untouched for the validator to report. Returns (plan, inserted).
    """
    if plan.status != "ok":
        return plan, 0
    state = {name: d.state for name, d in workspace.datasets.items()}
    steps: list[PlanStep] = []
    inserted = 0
    for step in plan.steps:
        name = normalize_dataset_ref(step.arguments.get("dataset")) if step.tool in TOOLS else None
        if name in state:
            required = _REQUIRED_STATE.get(step.tool)
            if required and state[name] not in required:
                needed = _PREREQUISITES[state[name]]
                if step.tool == "export_yolo":
                    needed = [t for t in needed if t != "export_yolo"]
                for tool in needed:
                    steps.append(PlanStep(tool=tool, arguments={"dataset": name}))
                    inserted += 1
                state[name] = "yolo" if "export_yolo" in needed else "annotated"
            after = _STATE_AFTER.get(step.tool)
            if after and _STATE_RANK[after] > _STATE_RANK[state[name]]:
                state[name] = after
        steps.append(step)
    return plan.model_copy(update={"steps": steps}), inserted


def validate_plan(plan: Plan, workspace: Workspace) -> ValidationResult:
    result = ValidationResult()
    if plan.status != "ok":
        return result
    if not plan.steps:
        result.issues.append(ValidationIssue(None, "empty_plan", "status is 'ok' but the plan has no steps"))
        return result

    state = {name: d.state for name, d in workspace.datasets.items()}
    for i, step in enumerate(plan.steps, start=1):
        spec = TOOLS.get(step.tool)
        if spec is None:
            result.issues.append(ValidationIssue(i, "unknown_tool", f"tool '{step.tool}' does not exist"))
            continue
        try:
            args = spec.args_model.model_validate(step.arguments).model_dump()
        except ValidationError as e:
            result.issues.append(ValidationIssue(i, "invalid_arguments", f"{step.tool}: {format_validation_error(e)}"))
            continue

        if "dataset" in args:
            name = normalize_dataset_ref(args["dataset"])
            if name not in state:
                result.issues.append(ValidationIssue(i, "dataset_not_found", f"dataset '{args['dataset']}' does not exist"))
                continue
            required = _REQUIRED_STATE.get(step.tool)
            if required and state[name] not in required:
                result.issues.append(ValidationIssue(
                    i, "dependency_violation",
                    f"{step.tool} requires dataset '{name}' in state {sorted(required)}, "
                    f"but it is '{state[name]}' at this point of the plan",
                ))
                continue
            after = _STATE_AFTER.get(step.tool)
            if after and _STATE_RANK[after] > _STATE_RANK[state[name]]:
                state[name] = after

        if step.tool == "get_job_status" and args["job_id"].strip() not in workspace.jobs:
            result.issues.append(ValidationIssue(i, "job_not_found", f"job '{args['job_id']}' does not exist"))

    return result
