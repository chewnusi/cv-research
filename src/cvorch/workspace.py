"""Simulated CV workspace.

No images are touched. The workspace only tracks *which state each dataset is in*,
so that tool preconditions (and therefore step ordering) are meaningful:

    raw --segment_dataset (SAM3)--> annotated --export_yolo (detection)--> yolo --train_yolo--> (job)

Datasets live in one of two places, and those are the only two ways a request may refer to one:

* a **project** (labeling-platform project) whose cloud folder holds the uploaded dataset;
  referenced by the project name (or its cloud folder URL);
* a **local dataset** that is not in any project; referenced by its exact local path.

Descriptions ("the parking lot images") are not references. Training never changes a dataset's state;
it creates a job. At most `gpu_quota` training jobs may be queued or running at the same time; queued and
running jobs can be cancelled by id. Datasets have no default segmentation prompt.
"""
from __future__ import annotations

import copy
import re
from dataclasses import asdict, dataclass, field
from typing import Literal

DatasetState = Literal["raw", "annotated", "yolo"]
JobStatus = Literal["queued", "running", "completed", "failed", "cancelled"]
ACTIVE_JOB_STATUSES = ("queued", "running")
CLOUD_ROOT = "s3://cv-projects"


@dataclass
class Dataset:
    id: str
    state: DatasetState
    num_images: int
    description: str = ""
    classes: list[str] = field(default_factory=list)
    local_path: str | None = None  # set for local datasets only

    @property
    def path(self) -> str:
        return self.local_path or f"{CLOUD_ROOT}/{self.id}"


@dataclass
class Project:
    name: str
    project_id: str
    dataset: str  # dataset id

    @property
    def cloud_folder(self) -> str:
        return cloud_folder_for(self.name)


@dataclass
class Job:
    job_id: str
    dataset: str  # dataset id
    model: str
    epochs: int
    status: JobStatus


def cloud_folder_for(project_name: str) -> str:
    slug = re.sub(r"[^a-z0-9._-]+", "-", project_name.strip().lower()).strip("-")
    return f"{CLOUD_ROOT}/{slug}/"


def clean_ref(ref: object) -> str | None:
    """Formatting-only cleanup of a dataset reference: whitespace, quotes and a trailing slash."""
    if not isinstance(ref, str):
        return None
    s = ref.strip().strip("'\"`").strip()
    if len(s) > 1:
        s = s.rstrip("/")
    return s or None


@dataclass
class Workspace:
    datasets: dict[str, Dataset]
    jobs: dict[str, Job]
    projects: dict[str, Project] = field(default_factory=dict)  # keyed by project name
    gpu_quota: int = 3
    _job_counter: int = 100
    _project_counter: int = 0

    def clone(self) -> Workspace:
        return copy.deepcopy(self)

    # ---------------------------------------------------------------- references

    def project(self, name: str | None) -> Project | None:
        if not name:
            return None
        key = name.strip().lower()
        return next((p for n, p in self.projects.items() if n.lower() == key), None)

    def resolve_ref(self, ref: object) -> str | None:
        """Dataset id for a valid reference (project name, project cloud folder or exact local path)."""
        s = clean_ref(ref)
        if s is None:
            return None
        if (p := self.project(s)) is not None:
            return p.dataset
        for p in self.projects.values():
            if s.lower() == p.cloud_folder.rstrip("/"):
                return p.dataset
        for d in self.datasets.values():
            if d.local_path and s == d.local_path.rstrip("/"):
                return d.id
        return None

    def resolve_dataset(self, ref: object) -> Dataset | None:
        did = self.resolve_ref(ref)
        return self.datasets.get(did) if did else None

    # ---------------------------------------------------------------- jobs

    def active_jobs(self) -> list[str]:
        return [j for j, job in self.jobs.items() if job.status in ACTIVE_JOB_STATUSES]

    def next_job_id(self) -> str:
        self._job_counter += 1
        return f"train-{self._job_counter:04d}"

    def next_project_id(self) -> str:
        self._project_counter += 1
        return f"proj-{self._project_counter:03d}"

    def add_project(self, name: str, dataset_id: str) -> Project:
        p = Project(name.strip(), self.next_project_id(), dataset_id)
        self.projects[p.name] = p
        return p

    # ---------------------------------------------------------------- views

    def snapshot(self) -> dict:
        return {
            "datasets": {n: asdict(d) for n, d in self.datasets.items()},
            "jobs": {j: asdict(job) for j, job in self.jobs.items()},
            "projects": {n: {**asdict(p), "cloud_folder": p.cloud_folder} for n, p in self.projects.items()},
            "gpu_quota": self.gpu_quota,
        }

    def describe(self) -> str:
        lines = ["Projects (labeling-platform projects; each cloud folder holds the project's uploaded dataset; "
                 "refer to the dataset by the project name):"]
        for p in self.projects.values():
            d = self.datasets[p.dataset]
            extra = f", annotated classes={d.classes}" if d.classes else ""
            lines.append(f"- {p.name}: {p.cloud_folder} - dataset {d.state}, {d.num_images} images, "
                         f"{d.description}{extra}")
        local = [d for d in self.datasets.values() if d.local_path]
        if local:
            lines.append("Local datasets (not in any project; refer to them by their exact local path):")
            for d in local:
                extra = f", annotated classes={d.classes}" if d.classes else ""
                lines.append(f"- {d.local_path}: {d.state}, {d.num_images} images, {d.description}{extra}")
        lines.append("Dataset states: raw = images only; annotated = has SAM3 segmentation annotations; "
                     "yolo = exported as a YOLO detection dataset and ready for training.")
        lines.append("Datasets have no default segmentation prompt.")
        lines.append(f"Existing training job ids: {', '.join(self.jobs)}")
        lines.append(f"GPU quota: at most {self.gpu_quota} training jobs can be queued or running at the same time; "
                     f"{len(self.active_jobs())} job(s) are active now. Cancelling a queued or running job frees "
                     "its slot; completed, failed or cancelled jobs cannot be cancelled.")
        return "\n".join(lines)


def snapshot_diff(before: dict, after: dict) -> list[str]:
    """Human-readable changes between two `Workspace.snapshot()`s."""
    changes: list[str] = []
    for name, d in after["datasets"].items():
        old = before["datasets"].get(name)
        label = d.get("local_path") or name
        if old is None:
            changes.append(f"dataset {label} added ({d['state']})")
        elif old["state"] != d["state"]:
            changes.append(f"dataset {label}: {old['state']} -> {d['state']}")
        elif old["classes"] != d["classes"]:
            changes.append(f"dataset {label}: classes {old['classes']} -> {d['classes']}")
    for job_id, j in after["jobs"].items():
        old = before["jobs"].get(job_id)
        if old is None:
            changes.append(f"job {job_id} created ({j['model']} on {j['dataset']}, {j['epochs']} epochs)")
        elif old["status"] != j["status"]:
            changes.append(f"job {job_id}: {old['status']} -> {j['status']}")
    for name, p in after["projects"].items():
        if name not in before["projects"]:
            changes.append(f"project {name} created ({p['project_id']}, {p['cloud_folder']}, dataset {p['dataset']})")
    return changes


def default_workspace() -> Workspace:
    """The fixed workspace that every benchmark case starts from."""
    in_projects = [
        Dataset("cats", "raw", 240, "photos of domestic cats"),
        Dataset("dogs", "raw", 310, "photos of dogs in parks"),
        Dataset("traffic", "raw", 1200, "street camera frames with cars, buses and pedestrians"),
        Dataset("fruits", "raw", 450, "supermarket shelf photos of fruits"),
        Dataset("warehouse", "annotated", 800, "warehouse boxes and pallets", ["box", "pallet"]),
        Dataset("xray", "annotated", 150, "industrial X-ray scans of welds", ["crack", "pore"]),
        Dataset("parking", "yolo", 950, "parking lot top-down views", ["car", "motorcycle"]),
        Dataset("drones", "yolo", 600, "aerial drone footage frames", ["person", "vehicle"]),
    ]
    local = [
        Dataset("local:dock_cam", "raw", 320, "loading-dock camera frames with forklifts and pallets",
                local_path="/data/local/dock_cam"),
        Dataset("local:solar_panels", "annotated", 410, "drone photos of solar farms", ["panel", "hotspot"],
                local_path="/data/local/solar_panels"),
        Dataset("local:shelves", "yolo", 500, "retail shelf photos", ["product", "price_tag"],
                local_path="/data/local/shelves"),
    ]
    jobs = [
        Job("train-0041", "parking", "yolo11n", 50, "completed"),
        Job("train-0042", "drones", "yolo11s", 100, "running"),
        Job("train-0043", "parking", "yolo11m", 30, "failed"),
    ]
    ws = Workspace(datasets={d.id: d for d in in_projects + local}, jobs={j.job_id: j for j in jobs})
    for d in in_projects:
        ws.add_project(d.id, d.id)
    return ws
