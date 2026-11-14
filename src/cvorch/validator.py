"""Deterministic full-plan validation (no LLM involved).

The whole plan is inspected before anything executes; every issue is reported, and a single issue
blocks the entire plan. The workspace is simulated across the steps (each step is assumed to succeed for
the purpose of checking later steps): dataset states, projects created by the plan, job cancellations and
the GPU quota. Ordering and prerequisites are therefore checked as a workflow.

When the user's request is given, the plan is additionally checked against it (explicit-request policy,
see grounding.py): every value a step uses must be stated in the request, and explicit values in the
request must not be dropped.

Issue codes:
  plan level      empty_plan              status 'ok' but no steps
  tool            unknown_tool            tool does not exist
  arguments       missing_argument        required argument absent (e.g. SAM3 text_prompt)
                  unsupported_argument    argument not in the tool contract
                  wrong_type              value of the wrong type (e.g. 2.5 epochs)
                  invalid_enum_value      value outside the allowed set (e.g. model 'faster_rcnn')
                  out_of_range            numeric value outside its range (e.g. epochs -10)
                  empty_argument          empty string where text is required
  references      dataset_not_found       not a project name, project cloud folder or local dataset path
                  job_not_found           training job does not exist
                  project_exists          create_labeling_project with a name that is already taken
  workflow        missing_prerequisite    dataset not in the required state and no step in the plan makes it so
                  invalid_order           the required step exists, but only later in the plan
                  job_not_cancellable     cancel_job on a completed, failed or already cancelled job
                  quota_exceeded          more queued/running training jobs than the GPU quota allows
  request         ungrounded_value        a value the request does not state (dataset, model, prompt, job,
                                          project name or non-default number)
                  request_value_not_used  a model, format or job id named in the request is not used
  repair          repair_changed_request  a repaired plan changed or dropped a value the user asked for that
                                          the validator had rejected (repair may only fix the LLM's own errors)
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .grounding import mentions, number_grounded, phrase_grounded, unused_request_values, without_references
from .plan import ExecutionPlan, PlanStep
from .tools import TOOLS, YOLO_DETECTION_MODELS, _numeric_type, check_arguments
from .workspace import ACTIVE_JOB_STATUSES, CLOUD_ROOT, Workspace, clean_ref, cloud_folder_for


@dataclass
class ValidationIssue:
    step: int | None  # 1-based, None for plan-level issues
    code: str
    message: str
    field: str | None = None

    def to_dict(self) -> dict:
        return {"step": self.step, "code": self.code, "message": self.message, "field": self.field}

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
VALUE_ISSUE_CODES = {"out_of_range", "invalid_enum_value", "wrong_type", "empty_argument"}


class WorkspaceSim:
    """Copy of the parts of the workspace the validator simulates over a plan."""

    def __init__(self, ws: Workspace):
        self.quota = ws.gpu_quota
        self.state = {d.id: d.state for d in ws.datasets.values()}
        self.projects = {p.name.lower(): p.dataset for p in ws.projects.values()}
        self.local = {d.local_path.rstrip("/"): d.id for d in ws.datasets.values() if d.local_path}
        self.jobs = {j: job.status for j, job in ws.jobs.items()}

    def _project_by_url(self, s: str) -> str | None:
        if not s.lower().startswith(CLOUD_ROOT + "/"):
            return None
        return next((n for n in self.projects if cloud_folder_for(n).rstrip("/") == s.lower()), None)

    def resolve(self, ref: object) -> str | None:
        s = clean_ref(ref)
        if s is None:
            return None
        if s.lower() in self.projects:
            return self.projects[s.lower()]
        if (name := self._project_by_url(s)) is not None:
            return self.projects[name]
        return self.local.get(s)

    def reference_forms(self, ref: object) -> list[str]:
        """Equivalent spellings of a valid reference: a project's name and its cloud folder."""
        s = clean_ref(ref)
        if s is None:
            return []
        name = s.lower() if s.lower() in self.projects else self._project_by_url(s)
        return [s] if name is None else [s, name, cloud_folder_for(name)]

    def active(self) -> int:
        return sum(1 for st in self.jobs.values() if st in ACTIVE_JOB_STATUSES)


def _grounding_issues(i: int, step: PlanStep, request: str, sim: WorkspaceSim) -> list[ValidationIssue]:
    out: list[ValidationIssue] = []
    args = step.arguments
    defaults = TOOLS[step.tool].defaults()

    def is_default(key: str) -> bool:
        return key in defaults and args[key] == defaults[key]

    def flag(key: str, what: str) -> None:
        out.append(ValidationIssue(i, "ungrounded_value",
                                   f"{step.tool}.{key}={args[key]!r}: {what} is not stated in the request", key))

    ref = args.get("dataset")
    if isinstance(ref, str) and not any(mentions(request, f) for f in sim.reference_forms(ref)):
        flag("dataset", "this dataset reference (project name or local path)")
    prompt_text = without_references(request, sim.reference_forms(ref)) if isinstance(ref, str) else request
    for key in ("text_prompt", "class_name"):
        if isinstance(args.get(key), str) and args[key].strip() and not phrase_grounded(prompt_text, args[key]):
            flag(key, "this SAM3 text" if key == "text_prompt" else "this class name")
    for key in ("model", "job_id", "project_name"):
        if (isinstance(args.get(key), str) and args[key].strip() and not is_default(key)
                and not mentions(request, args[key])):
            flag(key, f"this {key.replace('_', ' ')}")
    for key, value in args.items():
        if _numeric_type(step.tool, key) is None or isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if is_default(key):
            continue
        if not number_grounded(request, value):
            flag(key, "this number")
    return out


def validate_plan(plan: ExecutionPlan, workspace: Workspace, request: str | None = None,
                  check_coverage: bool = True) -> ValidationResult:
    """Validate a whole plan. With `request`, also apply the explicit-request (grounding) checks;
    `check_coverage` adds the plan-level check that explicit values of the request are not dropped."""
    result = ValidationResult()
    if plan.status != "ok":
        return result
    if not plan.steps:
        result.issues.append(ValidationIssue(None, "empty_plan", "status is 'ok' but the plan has no steps"))
        return result

    sim = WorkspaceSim(workspace)
    add = result.issues.append
    for i, step in enumerate(plan.steps, start=1):
        if step.tool not in TOOLS:
            add(ValidationIssue(i, "unknown_tool", f"tool '{step.tool}' does not exist"))
            continue

        _, arg_errors = check_arguments(step.tool, step.arguments)
        for e in arg_errors:
            add(ValidationIssue(i, e.code, f"{step.tool}.{e.field}: {e.message}", e.field))
        if request is not None:
            result.issues.extend(_grounding_issues(i, step, request, sim))

        ref = step.arguments.get("dataset")
        if isinstance(ref, str):
            did = sim.resolve(ref)
            if did is None:
                add(ValidationIssue(i, "dataset_not_found",
                                    f"dataset '{ref}' is not a project name or the exact path of a local dataset",
                                    "dataset"))
                continue
            required = REQUIRED_STATE.get(step.tool)
            if required and sim.state[did] not in required:
                provider = _PROVIDER[step.tool]
                later = any(s.tool == provider and sim.resolve(s.arguments.get("dataset")) == did
                            for s in plan.steps[i:])
                code = "invalid_order" if later else "missing_prerequisite"
                hint = f"{provider} appears only later in the plan" if later else f"no earlier {provider} step"
                add(ValidationIssue(
                    i, code, f"{step.tool} requires dataset '{ref}' in state {sorted(required)}, but it is "
                             f"'{sim.state[did]}' at this point of the plan ({hint})", "dataset"))
            if step.tool in STATE_AFTER:
                sim.state[did] = STATE_AFTER[step.tool]  # type: ignore[assignment]
            if step.tool == "create_labeling_project":
                name = step.arguments.get("project_name")
                if isinstance(name, str) and name.strip():
                    if name.strip().lower() in sim.projects:
                        add(ValidationIssue(i, "project_exists", f"project '{name}' already exists", "project_name"))
                    else:
                        sim.projects[name.strip().lower()] = did
            if step.tool == "train_yolo":
                sim.jobs[f"__planned_{i}"] = "queued"
                if sim.active() > sim.quota:
                    later = any(s.tool == "cancel_job" for s in plan.steps[i:])
                    hint = " (a cancel_job step appears only later in the plan)" if later else ""
                    add(ValidationIssue(i, "quota_exceeded",
                                        f"starting this job would make {sim.active()} queued/running training jobs; "
                                        f"the GPU quota is {sim.quota}{hint}"))

        job_id = step.arguments.get("job_id")
        if step.tool in ("get_job_status", "cancel_job") and isinstance(job_id, str):
            jid = job_id.strip()
            if jid not in sim.jobs:
                add(ValidationIssue(i, "job_not_found", f"job '{job_id}' does not exist", "job_id"))
            elif step.tool == "cancel_job":
                if sim.jobs[jid] not in ACTIVE_JOB_STATUSES:
                    add(ValidationIssue(i, "job_not_cancellable",
                                        f"job '{jid}' is {sim.jobs[jid]} at this point; only queued or running jobs "
                                        "can be cancelled", "job_id"))
                sim.jobs[jid] = "cancelled"

    if request is not None and check_coverage:
        used_models = [str(s.arguments.get("model") or TOOLS["train_yolo"].defaults()["model"])
                       for s in plan.steps if s.tool == "train_yolo"]
        used_jobs = [str(s.arguments.get("job_id")).strip() for s in plan.steps if "job_id" in s.arguments]
        for msg in unused_request_values(request, used_models, used_jobs, YOLO_DETECTION_MODELS):
            add(ValidationIssue(None, "request_value_not_used", msg))
    return result


def complete_prerequisites(plan: ExecutionPlan, workspace: Workspace) -> tuple[ExecutionPlan, int]:
    """Ablation only: insert a missing export_yolo step before steps that need a YOLO-format dataset.

    Only annotated datasets can be completed this way. A raw dataset would need segmentation, whose
    text prompt must come from the user, so it is left for the validator to report.
    Returns (plan, number of inserted steps).
    """
    if plan.status != "ok":
        return plan, 0
    sim = WorkspaceSim(workspace)
    steps: list[PlanStep] = []
    inserted = 0
    for step in plan.steps:
        did = sim.resolve(step.arguments.get("dataset")) if step.tool in TOOLS else None
        if did is not None:
            if step.tool == "train_yolo" and sim.state[did] == "annotated":
                steps.append(PlanStep(tool="export_yolo", arguments={"dataset": step.arguments["dataset"]}))
                inserted += 1
                sim.state[did] = "yolo"
            if step.tool in STATE_AFTER:
                sim.state[did] = STATE_AFTER[step.tool]  # type: ignore[assignment]
            if step.tool == "create_labeling_project" and isinstance(step.arguments.get("project_name"), str):
                sim.projects.setdefault(step.arguments["project_name"].strip().lower(), did)
        steps.append(step)
    return plan.model_copy(update={"steps": steps}), inserted


def _step_key(step: PlanStep) -> tuple[str, str]:
    ref = step.arguments.get("dataset", step.arguments.get("job_id"))
    return step.tool, (clean_ref(ref) or "").lower()


def repair_guard_issues(previous: ExecutionPlan, previous_issues: list[ValidationIssue], repaired: ExecutionPlan,
                        request: str) -> list[ValidationIssue]:
    """Values the user explicitly asked for and the validator rejected (e.g. "-5 epochs") must not be changed
    or dropped by a repair, and steps with unknown tools must not simply disappear: such requests have to
    be refused, not silently rewritten into a different valid request."""
    if repaired.status != "ok":
        return []
    out: list[ValidationIssue] = []
    for issue in previous_issues:
        if issue.code not in VALUE_ISSUE_CODES or issue.step is None or not issue.field:
            continue
        old = previous.steps[issue.step - 1]
        value = old.arguments.get(issue.field)
        stated = mentions(request, value) if isinstance(value, str) else number_grounded(request, value)
        if value is None or not stated:
            continue  # the rejected value was the LLM's own invention; repair may change it
        same = [s for s in repaired.steps if _step_key(s) == _step_key(old)]
        if not same or any(s.arguments.get(issue.field) != value for s in same):
            out.append(ValidationIssue(None, "repair_changed_request",
                                       f"{old.tool}.{issue.field}={value!r} was requested by the user and rejected; "
                                       "the repair changed or dropped it instead of refusing the request",
                                       issue.field))
    if any(i.code == "unknown_tool" for i in previous_issues) and len(repaired.steps) < len(previous.steps):
        out.append(ValidationIssue(None, "repair_changed_request",
                                   "a step with an unavailable tool was dropped by the repair instead of refusing "
                                   "the request"))
    return out
