"""Benchmark runner.

Fairness guarantees: every case (and every repeat) starts from a fresh `default_workspace()`, both
profiles use the same cases file, model settings, tool contracts and shared prompt. Fingerprints of the
cases file, the initial workspace and the shared prompt are stored in summary.json so runs can be checked
for comparability.
"""
from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from ..config import Settings
from ..llm import LLMClient
from ..orchestrators import make_orchestrator
from ..prompts import shared_prompt_prefix
from ..tools import openai_tools
from ..workspace import default_workspace
from .report import is_api_failure, summarize
from .scoring import score_case

BENCHMARK_DIR = Path(__file__).resolve().parents[3] / "benchmark"
DEFAULT_CASES = BENCHMARK_DIR / "cases.jsonl"


def load_cases(path: Path | str = DEFAULT_CASES, categories: list[str] | None = None,
               ids: list[str] | None = None, limit: int | None = None) -> list[dict]:
    cases = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if categories:
        cases = [c for c in cases if c["category"] in categories]
    if ids:
        cases = [c for c in cases if c["id"] in ids]
    return cases[:limit] if limit else cases


def _sha(obj) -> str:
    data = obj if isinstance(obj, str) else json.dumps(obj, sort_keys=True)
    return hashlib.sha256(data.encode()).hexdigest()[:16]


def fingerprints(cases: list[dict]) -> dict:
    return {
        "cases": _sha(cases),
        "initial_workspace": _sha(default_workspace().snapshot()),
        "tool_contracts": _sha(openai_tools()),
        "shared_prompt": _sha(shared_prompt_prefix(default_workspace())),
    }


def run_case(case: dict, llm: LLMClient, settings: Settings, repeat: int = 0) -> dict:
    orch = make_orchestrator(llm, settings)
    workspace = default_workspace()
    res = orch.run(case["prompt"], workspace)
    score = score_case(case, res)
    return {"case": case, "repeat": repeat, "result": res.to_dict(), "score": score.to_dict(),
            "final_workspace": workspace.snapshot()}


def run_benchmark(cases: list[dict], llm: LLMClient, settings: Settings, out_dir: Path | None = None,
                  repeats: int = 1, progress: bool = True, cases_path: Path | str | None = None,
                  ) -> tuple[list[dict], dict]:
    jobs = [(c, r) for r in range(repeats) for c in cases]
    t0 = time.perf_counter()

    def _do(job):
        case, r = job
        rec = run_case(case, llm, settings, r)
        if progress:
            sc = rec["score"]
            mark = "PASS" if sc["task_success"] else "FAIL"
            print(f"[{settings.profile}] {case['id']:<6} r{r} {mark} {','.join(sc['errors'])}", flush=True)
        return rec

    with ThreadPoolExecutor(max_workers=max(1, settings.bench_concurrency)) as ex:
        records = list(ex.map(_do, jobs))

    summary = summarize(records)
    summary["meta"] = {
        "profile": settings.profile,
        "settings": settings.public_dict(),
        "cases_path": str(cases_path) if cases_path else None,
        "num_cases": len(cases),
        "repeats": repeats,
        "fingerprints": fingerprints(cases),
        "wall_time_s": round(time.perf_counter() - t0, 2),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "cases.jsonl").open("w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return records, summary


def default_out_dir(settings: Settings, root: Path = Path("results")) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model = settings.llm_model.replace("/", "_").replace(":", "_")
    return root / f"{stamp}_{settings.profile}_{model}"


def retry_errors(out_dir: Path, llm: LLMClient, settings: Settings, progress: bool = True) -> tuple[list[dict], dict, int]:
    """Re-run only the (case, repeat) records of an existing run that ended in a system error (e.g. an API
    outage or exhausted credits), replace them in place and recompute the summary. Other records are kept."""
    out_dir = Path(out_dir)
    records = [json.loads(line) for line in (out_dir / "cases.jsonl").read_text().splitlines() if line.strip()]
    old = json.loads((out_dir / "summary.json").read_text())
    meta = old.get("meta", {})
    if meta.get("profile") != settings.profile:
        raise ValueError(f"{out_dir} was run with profile {meta.get('profile')!r}, not {settings.profile!r}")
    todo = [i for i, rec in enumerate(records) if is_api_failure(rec["result"].get("error"))]

    def _redo(i: int) -> dict:
        rec = run_case(records[i]["case"], llm, settings, records[i]["repeat"])
        if progress:
            sc = rec["score"]
            status = "ERROR" if rec["result"]["error"] else ("PASS" if sc["task_success"] else "FAIL")
            print(f"[{settings.profile}] retry {rec['case']['id']:<6} r{rec['repeat']} {status}", flush=True)
        return rec

    with ThreadPoolExecutor(max_workers=max(1, settings.bench_concurrency)) as ex:
        for i, rec in zip(todo, ex.map(_redo, todo)):
            records[i] = rec
    retried = len(todo)
    summary = summarize(records)
    summary["meta"] = {**meta, "retried_errors": meta.get("retried_errors", 0) + retried,
                       "retried_at": datetime.now().isoformat(timespec="seconds")}
    with (out_dir / "cases.jsonl").open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return records, summary, retried
