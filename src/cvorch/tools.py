"""Tool contracts (shared by both orchestration modes) and the mock tool runtime."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .workspace import Workspace

SEGMENTATION_MODELS = ("sam2", "yolo11n-seg", "yolo11s-seg")
YOLO_MODELS = (
    "yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x",
    "yolo11n-seg", "yolo11s-seg", "yolo11m-seg",
)


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SegmentDatasetArgs(_Args):
    """Automatically segment all images of a dataset and save the annotations."""
    dataset: str = Field(description="Dataset name, e.g. 'cats'.")
    model: Literal[SEGMENTATION_MODELS] = Field("sam2", description="Segmentation model.")  # type: ignore[valid-type]
    classes: list[str] | None = Field(None, description="Optional list of object classes to segment (text prompts).")
    confidence: float = Field(0.5, gt=0.0, lt=1.0, description="Mask confidence threshold, between 0 and 1.")


class ExportYoloArgs(_Args):
    """Convert an annotated dataset into YOLO format (images/, labels/, data.yaml) with a train/val split."""
    dataset: str = Field(description="Dataset name. The dataset must already be annotated.")
    val_split: float = Field(0.2, gt=0.0, le=0.5, description="Fraction of images used for validation.")
    task: Literal["detect", "segment"] = Field("segment", description="Export boxes (detect) or polygons (segment).")


class TrainYoloArgs(_Args):
    """Start training a YOLO model on a dataset in YOLO format. Returns a job id."""
    dataset: str = Field(description="Dataset name. The dataset must already be in YOLO format.")
    model: Literal[YOLO_MODELS] = Field("yolo11n", description="YOLO model variant.")  # type: ignore[valid-type]
    epochs: int = Field(100, ge=1, le=1000, description="Number of training epochs.")
    batch_size: int = Field(16, ge=1, le=256, description="Batch size.")
    imgsz: int = Field(640, ge=32, le=2048, description="Training image size in pixels.")


class GetJobStatusArgs(_Args):
    """Get the status and metrics of a training job."""
    job_id: str = Field(description="Training job id, e.g. 'train-0041'.")


class CreateLabelingProjectArgs(_Args):
    """Create a project in the labeling platform for manual review/correction of a dataset."""
    project_name: str = Field(min_length=1, description="Name of the labeling project.")
    dataset: str = Field(description="Dataset name to attach to the project.")


@dataclass(frozen=True)
class ToolSpec:
    name: str
    args_model: type[_Args]

    @property
    def description(self) -> str:
        return (self.args_model.__doc__ or "").strip()

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
        return {
            k: f.default for k, f in self.args_model.model_fields.items() if not f.is_required()
        }


TOOLS: dict[str, ToolSpec] = {
    s.name: s
    for s in [
        ToolSpec("segment_dataset", SegmentDatasetArgs),
        ToolSpec("export_yolo", ExportYoloArgs),
        ToolSpec("train_yolo", TrainYoloArgs),
        ToolSpec("get_job_status", GetJobStatusArgs),
        ToolSpec("create_labeling_project", CreateLabelingProjectArgs),
    ]
}


def openai_tools() -> list[dict]:
    return [t.openai_tool() for t in TOOLS.values()]


def tools_catalog_text() -> str:
    """Same tool contracts as the baseline gets via the `tools` API, rendered for a prompt."""
    parts = []
    for t in TOOLS.values():
        parts.append(f"### {t.name}\n{t.description}\nParameters (JSON schema): {json.dumps(t.parameters_schema())}")
    return "\n\n".join(parts)


def format_validation_error(e: ValidationError) -> str:
    return "; ".join(f"{'.'.join(str(p) for p in err['loc']) or '<root>'}: {err['msg']}" for err in e.errors())


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

    Preconditions that a real implementation would hit at run time (missing dataset,
    wrong dataset state, unknown job) are always enforced. Argument type/range checks
    are enforced only when `arg_validation` is on.
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
            "create_labeling_project": self._labeling,
        }

    def call(self, tool: str, args: dict[str, Any]) -> ToolCallRecord:
        try:
            if tool not in TOOLS:
                raise ToolError(f"unknown tool '{tool}'")
            if not isinstance(args, dict):
                raise ToolError("arguments must be a JSON object")
            parsed = self._parse(tool, args)
            output = self._handlers[tool](parsed)
            rec = ToolCallRecord(tool, args, True, output=output)
        except ToolError as e:
            rec = ToolCallRecord(tool, args, False, error=str(e))
        self.calls.append(rec)
        return rec

    def _parse(self, tool: str, args: dict) -> dict:
        spec = TOOLS[tool]
        if self.arg_validation:
            try:
                return spec.args_model.model_validate(args).model_dump()
            except ValidationError as e:
                raise ToolError(f"invalid arguments: {format_validation_error(e)}") from None
        merged = {**spec.defaults(), **args}
        missing = [k for k, f in spec.args_model.model_fields.items() if f.is_required() and k not in merged]
        if missing:
            raise ToolError(f"missing required arguments: {missing}")
        return merged

    def _dataset(self, ref: Any):
        d = self.ws.resolve_dataset(ref if isinstance(ref, str) else None)
        if d is None:
            raise ToolError(f"dataset '{ref}' not found")
        return d

    def _segment(self, a: dict) -> dict:
        d = self._dataset(a["dataset"])
        d.state = "annotated" if d.state == "raw" else d.state
        if a.get("classes"):
            d.classes = list(a["classes"])
        return {"status": "completed", "annotation_path": f"{d.path}/annotations", "num_images": d.num_images,
                "model": a.get("model")}

    def _export(self, a: dict) -> dict:
        d = self._dataset(a["dataset"])
        if d.state == "raw":
            raise ToolError(f"dataset '{d.name}' has no annotations; segment it first")
        d.state = "yolo"
        return {"status": "completed", "dataset_path": f"{d.path}/yolo", "data_yaml": f"{d.path}/yolo/data.yaml"}

    def _train(self, a: dict) -> dict:
        d = self._dataset(a["dataset"])
        if d.state != "yolo":
            raise ToolError(f"dataset '{d.name}' is not in YOLO format (state={d.state}); export it first")
        from .workspace import Job

        job_id = self.ws.next_job_id()
        self.ws.jobs[job_id] = Job(job_id, d.name, str(a.get("model")), a.get("epochs"), "queued")  # type: ignore[arg-type]
        return {"status": "queued", "job_id": job_id, "weights_path": f"runs/{job_id}/weights/best.pt"}

    def _status(self, a: dict) -> dict:
        job = self.ws.jobs.get(str(a["job_id"]).strip())
        if job is None:
            raise ToolError(f"job '{a['job_id']}' not found")
        metrics = {"mAP50": 0.71, "mAP50-95": 0.48} if job.status == "completed" else None
        return {"job_id": job.job_id, "status": job.status, "metrics": metrics}

    def _labeling(self, a: dict) -> dict:
        d = self._dataset(a["dataset"])
        pid = self.ws.next_project_id()
        self.ws.projects[pid] = {"name": a["project_name"], "dataset": d.name}
        return {"status": "created", "project_id": pid, "backend": "MockLabelingService"}
