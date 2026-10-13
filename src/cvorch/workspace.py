"""Simulated CV workspace.

No images are touched. The workspace only tracks *which state each dataset is in*,
so that tool preconditions (and therefore step ordering) are meaningful:

    raw --segment_dataset (SAM3)--> annotated --export_yolo (detection)--> yolo --train_yolo--> (job)

Training never changes a dataset's state; it creates a job. Datasets have no default segmentation
prompt: the SAM3 text prompt must always come from the user's request.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from typing import Literal

DatasetState = Literal["raw", "annotated", "yolo"]


@dataclass
class Dataset:
    name: str
    state: DatasetState
    num_images: int
    description: str = ""
    classes: list[str] = field(default_factory=list)

    @property
    def path(self) -> str:
        return f"data/{self.name}"


@dataclass
class Job:
    job_id: str
    dataset: str
    model: str
    epochs: int
    status: Literal["queued", "running", "completed", "failed"]


@dataclass
class Workspace:
    datasets: dict[str, Dataset]
    jobs: dict[str, Job]
    projects: dict[str, dict] = field(default_factory=dict)
    _job_counter: int = 100
    _project_counter: int = 0

    def clone(self) -> Workspace:
        return copy.deepcopy(self)

    def resolve_dataset(self, ref: str | None) -> Dataset | None:
        key = normalize_dataset_ref(ref)
        return self.datasets.get(key) if key else None

    def next_job_id(self) -> str:
        self._job_counter += 1
        return f"train-{self._job_counter:04d}"

    def next_project_id(self) -> str:
        self._project_counter += 1
        return f"proj-{self._project_counter:03d}"

    def snapshot(self) -> dict:
        return {
            "datasets": {n: asdict(d) for n, d in self.datasets.items()},
            "jobs": {j: asdict(job) for j, job in self.jobs.items()},
            "projects": copy.deepcopy(self.projects),
        }

    def describe(self) -> str:
        lines = ["Datasets (name: state, images, description):"]
        for d in self.datasets.values():
            extra = f", annotated classes={d.classes}" if d.classes else ""
            lines.append(f"- {d.name}: {d.state}, {d.num_images} images, {d.description}{extra}")
        lines.append("Dataset states: raw = images only; annotated = has SAM3 segmentation annotations; "
                     "yolo = exported as a YOLO detection dataset and ready for training.")
        lines.append("Datasets have no default segmentation prompt.")
        lines.append(f"Existing training job ids: {', '.join(self.jobs)}")
        return "\n".join(lines)


def snapshot_diff(before: dict, after: dict) -> list[str]:
    """Human-readable changes between two `Workspace.snapshot()`s."""
    changes: list[str] = []
    for name, d in after["datasets"].items():
        old = before["datasets"].get(name)
        if old is None:
            changes.append(f"dataset {name} added ({d['state']})")
        elif old["state"] != d["state"]:
            changes.append(f"dataset {name}: {old['state']} -> {d['state']}")
        elif old["classes"] != d["classes"]:
            changes.append(f"dataset {name}: classes {old['classes']} -> {d['classes']}")
    for job_id, j in after["jobs"].items():
        if job_id not in before["jobs"]:
            changes.append(f"job {job_id} created ({j['model']} on {j['dataset']}, {j['epochs']} epochs)")
    for pid, p in after["projects"].items():
        if pid not in before["projects"]:
            changes.append(f"labeling project {pid} created ({p.get('name')} on {p.get('dataset')})")
    return changes


def normalize_dataset_ref(ref: str | None) -> str | None:
    """'./data/cats/' , 'data/cats', 'cats' -> 'cats'."""
    if ref is None:
        return None
    s = str(ref).strip().strip("'\"").replace("\\", "/")
    while s.startswith("./"):
        s = s[2:]
    s = s.rstrip("/")
    for prefix in ("data/", "datasets/", "dataset/"):
        if s.startswith(prefix):
            s = s[len(prefix):]
    for suffix in ("/annotations", "/yolo", "/images"):
        if s.endswith(suffix):
            s = s[: -len(suffix)]
    return s.lower() or None


def default_workspace() -> Workspace:
    """The fixed workspace that every benchmark case starts from."""
    datasets = [
        Dataset("cats", "raw", 240, "photos of domestic cats"),
        Dataset("dogs", "raw", 310, "photos of dogs in parks"),
        Dataset("traffic", "raw", 1200, "street camera frames with cars, buses and pedestrians"),
        Dataset("fruits", "raw", 450, "supermarket shelf photos of fruits"),
        Dataset("warehouse", "annotated", 800, "warehouse boxes and pallets", ["box", "pallet"]),
        Dataset("xray", "annotated", 150, "industrial X-ray scans of welds", ["crack", "pore"]),
        Dataset("parking", "yolo", 950, "parking lot top-down views", ["car", "motorcycle"]),
        Dataset("drones", "yolo", 600, "aerial drone footage frames", ["person", "vehicle"]),
    ]
    jobs = [
        Job("train-0041", "parking", "yolo11n", 50, "completed"),
        Job("train-0042", "drones", "yolo11s", 100, "running"),
        Job("train-0043", "parking", "yolo11m", 30, "failed"),
    ]
    return Workspace(datasets={d.name: d for d in datasets}, jobs={j.job_id: j for j in jobs})
