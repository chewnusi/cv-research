"""Deterministic full-plan validation (no LLM involved).

The whole plan is inspected before anything executes; every issue is reported, and a single issue
blocks the entire plan. Dataset state is simulated across the steps (each step is assumed to succeed
for the purpose of checking later steps), so ordering and prerequisites are checked as a workflow.

Issue codes:
  plan level      empty_plan             status 'ok' but no steps
  tool            unknown_tool           tool does not exist
  arguments       missing_argument       required argument absent (e.g. SAM3 text_prompt)
                  unsupported_argument   argument not in the tool contract
                  wrong_type             value of the wrong type (e.g. 2.5 epochs)
                  invalid_enum_value     value outside the allowed set (e.g. model 'faster_rcnn')
                  out_of_range           numeric value outside its range (e.g. epochs -10)
                  empty_argument         empty string where text is required
  references      dataset_not_found      dataset does not exist in the workspace
                  job_not_found          training job does not exist
  workflow        missing_prerequisite   dataset not in the required state and no step in the plan makes it so
                  invalid_order          the required step exists, but only later in the plan
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .plan import ExecutionPlan, PlanStep
from .tools import TOOLS, check_arguments
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

    def to_dicts(self) -> list[dict]:
        return [i.to_dict() for i in self.issues]


REQUIRED_STATE = {"export_yolo": {"annotated", "yolo"}, "train_yolo": {"yolo"}}
STATE_AFTER = {"segment_dataset": "annotated", "export_yolo": "yolo"}
# Which tool, if it appears later in the plan, would have provided the missing state.
_PROVIDER = {"export_yolo": "segment_dataset", "train_yolo": "export_yolo"}


def _dataset_of(step: PlanStep) -> str | None:
    ref = step.arguments.get("dataset")
    return normalize_dataset_ref(ref) if isinstance(ref, str) else None


def validate_plan(plan: ExecutionPlan, workspace: Workspace) -> ValidationResult:
    result = ValidationResult()
    if plan.status != "ok":
        return result
    if not plan.steps:
        result.issues.append(ValidationIssue(None, "empty_plan", "status is 'ok' but the plan has no steps"))
        return result

    state = {name: d.state for name, d in workspace.datasets.items()}
    for i, step in enumerate(plan.steps, start=1):
        if step.tool not in TOOLS:
            result.issues.append(ValidationIssue(i, "unknown_tool", f"tool '{step.tool}' does not exist"))
            continue

        _, arg_errors = check_arguments(step.tool, step.arguments)
        for e in arg_errors:
            result.issues.append(ValidationIssue(i, e.code, f"{step.tool}.{e.field}: {e.message}"))

        name = _dataset_of(step)
        if "dataset" in step.arguments and isinstance(step.arguments["dataset"], str):
            if name not in state:
                result.issues.append(ValidationIssue(
                    i, "dataset_not_found", f"dataset '{step.arguments['dataset']}' does not exist"))
                continue
            required = REQUIRED_STATE.get(step.tool)
            if required and state[name] not in required:
                provider = _PROVIDER[step.tool]
                later = any(s.tool == provider and _dataset_of(s) == name for s in plan.steps[i:])
                code = "invalid_order" if later else "missing_prerequisite"
                hint = f"{provider} appears only later in the plan" if later else f"no earlier {provider} step"
                result.issues.append(ValidationIssue(
                    i, code,
                    f"{step.tool} requires dataset '{name}' in state {sorted(required)}, but it is "
                    f"'{state[name]}' at this point of the plan ({hint})",
                ))
            if step.tool in STATE_AFTER:
                state[name] = STATE_AFTER[step.tool]  # type: ignore[assignment]

        job_id = step.arguments.get("job_id")
        if step.tool == "get_job_status" and isinstance(job_id, str) and job_id.strip() not in workspace.jobs:
            result.issues.append(ValidationIssue(i, "job_not_found", f"job '{job_id}' does not exist"))

    return result


def complete_prerequisites(plan: ExecutionPlan, workspace: Workspace) -> tuple[ExecutionPlan, int]:
    """Ablation only: insert a missing export_yolo step before steps that need a YOLO-format dataset.

    Only annotated datasets can be completed this way. A raw dataset would need segmentation, whose
    text prompt must come from the user, so it is left for the validator to report.
    Returns (plan, number of inserted steps).
    """
    if plan.status != "ok":
        return plan, 0
    state = {name: d.state for name, d in workspace.datasets.items()}
    steps: list[PlanStep] = []
    inserted = 0
    for step in plan.steps:
        name = _dataset_of(step) if step.tool in TOOLS else None
        if name in state:
            if step.tool == "train_yolo" and state[name] == "annotated":
                steps.append(PlanStep(tool="export_yolo", arguments={"dataset": name}))
                inserted += 1
                state[name] = "yolo"
            if step.tool in STATE_AFTER:
                state[name] = STATE_AFTER[step.tool]  # type: ignore[assignment]
        steps.append(step)
    return plan.model_copy(update={"steps": steps}), inserted
