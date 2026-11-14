"""Tool contracts (shared by both orchestration modes) and the simulated tool runtime.

The contracts describe the intended real system:

    segment_dataset  SAM3 text-prompt segmentation of every image of a dataset
    export_yolo      SAM3 masks -> bounding boxes -> YOLO *detection* dataset
    train_yolo       Ultralytics YOLO *detection* training (creates a job)
    get_job_status   status/metrics of a training job
    cancel_job       cancel a queued or running training job (frees a GPU slot)
    create_labeling_project  project in a labeling platform with a cloud folder for the dataset (mock; no CVAT)

Nothing is actually executed: the runtime only updates the simulated workspace and returns
plausible metadata. The research subject is orchestration, not CV quality.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Annotated, Any, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError

from .workspace import ACTIVE_JOB_STATUSES, Job, Workspace, clean_ref

YOLO_DETECTION_MODELS = ("yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x")

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
DATASET_REF = "Dataset reference: a project name or the exact local path of a local dataset"


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SegmentDatasetArgs(_Args):
    """Segment every image of a dataset with SAM3 using a text prompt describing the target objects.
    Works on a dataset in any state (re-segmenting replaces earlier annotations); the dataset becomes
    'annotated'. The text prompt must come from the user's request."""
    dataset: str = Field(description=f"{DATASET_REF}, e.g. 'cats' or '/data/local/dock_cam'.")
    text_prompt: NonEmptyStr = Field(description="Text description of the objects to segment, e.g. 'car'.")
    class_name: NonEmptyStr | None = Field(
        None, description="Class label for the resulting annotations. Defaults to the text prompt.")
    confidence: float = Field(0.5, gt=0.0, lt=1.0, description="Mask confidence threshold, between 0 and 1.")


class ExportYoloArgs(_Args):
    """Convert the SAM3 annotations of a dataset into a YOLO object-detection dataset
    (masks -> bounding boxes; images/, labels/, data.yaml) with a train/val split.
    Needs the dataset in 'annotated' or 'yolo' state (a 'yolo' dataset can be re-exported) when the step
    runs; it becomes 'yolo'."""
    dataset: str = Field(description=f"{DATASET_REF}; in 'annotated' or 'yolo' state when this step runs.")
    val_split: float = Field(0.2, gt=0.0, le=0.5, description="Fraction of images used for validation.")


class TrainYoloArgs(_Args):
    """Start training an Ultralytics YOLO object-detection model on a dataset in YOLO format.
    Needs the dataset in 'yolo' state when the step runs. Returns a job id. Several training jobs may run
    at the same time, on the same or different datasets, up to the GPU quota (queued + running jobs); the call
    fails when the quota is full."""
    dataset: str = Field(description=f"{DATASET_REF}; in 'yolo' state when this step runs.")
    model: Literal[YOLO_DETECTION_MODELS] = Field("yolo11n", description="YOLO11 detection model variant.")  # type: ignore[valid-type]
    epochs: int = Field(100, ge=1, le=1000, description="Number of training epochs.")
    batch_size: int = Field(16, ge=1, le=256, description="Batch size.")
    imgsz: int = Field(640, ge=32, le=2048, description="Training image size in pixels.")


class GetJobStatusArgs(_Args):
    """Get the status and metrics of a training job."""
    job_id: str = Field(description="Training job id, e.g. 'train-0041'.")


class CancelJobArgs(_Args):
    """Cancel a queued or running training job, freeing its GPU slot. Completed, failed or already cancelled
    jobs cannot be cancelled. Cancelling cannot be undone."""
    job_id: str = Field(description="Training job id, e.g. 'train-0042'.")


class CreateLabelingProjectArgs(_Args):
    """Create a project in the labeling platform for manual review/correction of a dataset (simulated). The
    dataset is uploaded to the new project's cloud folder; afterwards the project name also refers to it.
    The project name must not be taken by an existing project."""
    project_name: NonEmptyStr = Field(description="Name of the new project.")
    dataset: str = Field(description=f"{DATASET_REF}.")


@dataclass(frozen=True)
class ToolSpec:
    name: str
    args_model: type[_Args]

    @property
    def description(self) -> str:
        return " ".join((self.args_model.__doc__ or "").split())

    def parameters_schema(self) -> dict:
        schema = self.args_model.model_json_schema()
        schema.pop("title", None)
        schema.pop("description", None)
        for prop in schema.get("properties", {}).values():
            prop.pop("title", None)
        return schema

    def openai_tool(self) -> dict:
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": self.parameters_schema()},
        }

    def defaults(self) -> dict[str, Any]:
        return {k: f.default for k, f in self.args_model.model_fields.items() if not f.is_required()}


TOOLS: dict[str, ToolSpec] = {
    s.name: s
    for s in [
        ToolSpec("segment_dataset", SegmentDatasetArgs),
        ToolSpec("export_yolo", ExportYoloArgs),
        ToolSpec("train_yolo", TrainYoloArgs),
        ToolSpec("get_job_status", GetJobStatusArgs),
        ToolSpec("cancel_job", CancelJobArgs),
        ToolSpec("create_labeling_project", CreateLabelingProjectArgs),
    ]
}


def openai_tools() -> list[dict]:
    return [t.openai_tool() for t in TOOLS.values()]


def tools_catalog_text() -> str:
    """The same tool contracts the baseline receives through tool binding, rendered for a prompt."""
    parts = []
    for t in TOOLS.values():
        parts.append(f"### {t.name}\n{t.description}\nParameters (JSON schema): {json.dumps(t.parameters_schema())}")
    return "\n\n".join(parts)


# --------------------------------------------------------------------------- argument normalization

_PERCENT = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*%\s*$")
_NUMBER = re.compile(r"^\s*-?\d+(?:\.\d+)?\s*$")


def _numeric_type(tool: str, key: str) -> type | None:
    spec = TOOLS.get(tool)
    f = spec.args_model.model_fields.get(key) if spec else None
    return f.annotation if f is not None and f.annotation in (int, float) else None


def normalize_value(tool: str, key: str, value: Any, workspace: Workspace | None = None) -> Any:
    """Lossless format conversion only; values are never clamped or changed in meaning."""
    if key == "dataset" and isinstance(value, str):
        return clean_ref(value) or value
    if key in ("job_id", "project_name") and isinstance(value, str):
        return value.strip()
    if key == "model" and isinstance(value, str):
        return value.strip().lower()
    typ = _numeric_type(tool, key)
    if typ is float and isinstance(value, str):
        if m := _PERCENT.match(value):
            return float(m.group(1)) / 100
        if _NUMBER.match(value):
            return float(value)
    if typ is int and isinstance(value, str) and _NUMBER.match(value):
        num = float(value)
        return int(num) if num.is_integer() else num
    return value


def normalize_arguments(tool: str, args: dict[str, Any], workspace: Workspace | None = None) -> dict[str, Any]:
    """Drop nulls ("not specified") and normalize formats."""
    return {k: normalize_value(tool, k, v, workspace) for k, v in args.items() if v is not None}


# --------------------------------------------------------------------------- argument errors

_ERROR_CODES = {
    "missing": "missing_argument",
    "extra_forbidden": "unsupported_argument",
    "literal_error": "invalid_enum_value",
    "enum": "invalid_enum_value",
    "greater_than": "out_of_range",
    "greater_than_equal": "out_of_range",
    "less_than": "out_of_range",
    "less_than_equal": "out_of_range",
    "string_too_short": "empty_argument",
}


@dataclass(frozen=True)
class ArgumentError:
    code: str  # missing_argument | unsupported_argument | wrong_type | invalid_enum_value | out_of_range | empty_argument
    field: str
    message: str


def argument_errors(e: ValidationError) -> list[ArgumentError]:
    out = []
    for err in e.errors():
        loc = ".".join(str(p) for p in err["loc"]) or "<root>"
        out.append(ArgumentError(_ERROR_CODES.get(err["type"], "wrong_type"), loc, err["msg"]))
    return out


def check_arguments(tool: str, args: dict[str, Any]) -> tuple[dict[str, Any] | None, list[ArgumentError]]:
    """Validate arguments against the tool contract. Returns (parsed args with defaults, errors)."""
    try:
        return TOOLS[tool].args_model.model_validate(args).model_dump(), []
    except ValidationError as e:
        return None, argument_errors(e)


def format_argument_errors(errors: list[ArgumentError]) -> str:
    return "; ".join(f"{e.field}: {e.message}" for e in errors)


# --------------------------------------------------------------------------- runtime


class ToolError(Exception):
    pass


@dataclass
class ToolCallRecord:
    tool: str
    args: dict[str, Any]
    ok: bool
    output: dict[str, Any] | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        return {"tool": self.tool, "args": self.args, "ok": self.ok, "output": self.output, "error": self.error}


class MockToolRuntime:
    """Executes tool calls against a simulated workspace and records every call.

    Preconditions that a real implementation would hit at run time (missing dataset, wrong dataset
    state, unknown job) are always enforced. Argument contract checks (types, ranges, enums) are
    enforced only when `arg_validation` is on, as a typical tool framework would do.
    """

    def __init__(self, workspace: Workspace, arg_validation: bool = True):
        self.ws = workspace
        self.arg_validation = arg_validation
        self.calls: list[ToolCallRecord] = []
        self._handlers: dict[str, Callable[[dict], dict]] = {
            "segment_dataset": self._segment,
            "export_yolo": self._export,
            "train_yolo": self._train,
            "get_job_status": self._status,
            "cancel_job": self._cancel,
            "create_labeling_project": self._labeling,
        }

    def call(self, tool: str, args: dict[str, Any]) -> ToolCallRecord:
        try:
            if tool not in TOOLS:
                raise ToolError(f"unknown tool '{tool}'")
            if not isinstance(args, dict):
                raise ToolError("arguments must be a JSON object")
            output = self._handlers[tool](self._parse(tool, args))
            rec = ToolCallRecord(tool, args, True, output=output)
        except ToolError as e:
            rec = ToolCallRecord(tool, args, False, error=str(e))
        self.calls.append(rec)
        return rec

    def _parse(self, tool: str, args: dict) -> dict:
        args = normalize_arguments(tool, args, self.ws)
        spec = TOOLS[tool]
        if self.arg_validation:
            parsed, errors = check_arguments(tool, args)
            if errors:
                raise ToolError(f"invalid arguments: {format_argument_errors(errors)}")
            return parsed  # type: ignore[return-value]
        merged = {**spec.defaults(), **args}
        missing = [k for k, f in spec.args_model.model_fields.items() if f.is_required() and k not in merged]
        if missing:
            raise ToolError(f"missing required arguments: {missing}")
        return merged

    def _dataset(self, ref: Any):
        d = self.ws.resolve_dataset(ref)
        if d is None:
            raise ToolError(f"dataset '{ref}' not found (use a project name or the exact local path of a local "
                            "dataset)")
        return d

    def _segment(self, a: dict) -> dict:
        d = self._dataset(a["dataset"])
        class_name = a.get("class_name") or a.get("text_prompt")
        d.state = "annotated"
        d.classes = [str(class_name)]
        return {"status": "completed", "backend": "SAM3 (simulated)", "annotation_path": f"{d.path}/annotations",
                "num_images": d.num_images, "text_prompt": a.get("text_prompt"), "class_name": class_name,
                "confidence": a.get("confidence")}

    def _export(self, a: dict) -> dict:
        d = self._dataset(a["dataset"])
        if d.state == "raw":
            raise ToolError(f"dataset '{a['dataset']}' has no annotations; segment it first")
        d.state = "yolo"
        val = a.get("val_split")
        num_val = round(d.num_images * val) if isinstance(val, (int, float)) else None
        return {"status": "completed", "format": "yolo-detection", "dataset_path": f"{d.path}/yolo",
                "data_yaml": f"{d.path}/yolo/data.yaml", "classes": d.classes,
                "num_train": d.num_images - num_val if num_val is not None else None, "num_val": num_val}

    def _train(self, a: dict) -> dict:
        d = self._dataset(a["dataset"])
        if d.state != "yolo":
            raise ToolError(f"dataset '{a['dataset']}' is not in YOLO format (state={d.state}); export it first")
        active = self.ws.active_jobs()
        if len(active) >= self.ws.gpu_quota:
            raise ToolError(f"GPU quota exceeded: {len(active)} of {self.ws.gpu_quota} training slots are in use "
                            f"({', '.join(active)})")
        job_id = self.ws.next_job_id()
        self.ws.jobs[job_id] = Job(job_id, d.id, str(a.get("model")), a.get("epochs"), "queued")  # type: ignore[arg-type]
        return {"status": "queued", "job_id": job_id, "model": a.get("model"), "epochs": a.get("epochs"),
                "weights_path": f"runs/{job_id}/weights/best.pt"}

    def _status(self, a: dict) -> dict:
        job = self.ws.jobs.get(str(a["job_id"]).strip())
        if job is None:
            raise ToolError(f"job '{a['job_id']}' not found")
        metrics = {"mAP50": 0.71, "mAP50-95": 0.48} if job.status == "completed" else None
        return {"job_id": job.job_id, "status": job.status, "dataset": job.dataset, "model": job.model,
                "metrics": metrics}

    def _cancel(self, a: dict) -> dict:
        job = self.ws.jobs.get(str(a["job_id"]).strip())
        if job is None:
            raise ToolError(f"job '{a['job_id']}' not found")
        if job.status not in ACTIVE_JOB_STATUSES:
            raise ToolError(f"job '{job.job_id}' is {job.status} and cannot be cancelled")
        previous, job.status = job.status, "cancelled"
        return {"job_id": job.job_id, "status": "cancelled", "previous_status": previous,
                "active_jobs": len(self.ws.active_jobs()), "gpu_quota": self.ws.gpu_quota}

    def _labeling(self, a: dict) -> dict:
        d = self._dataset(a["dataset"])
        name = str(a["project_name"]).strip()
        if self.ws.project(name) is not None:
            raise ToolError(f"project '{name}' already exists")
        p = self.ws.add_project(name, d.id)
        return {"status": "created", "project_id": p.project_id, "project_name": p.name,
                "cloud_folder": p.cloud_folder, "uploaded_images": d.num_images, "backend": "MockLabelingService"}
