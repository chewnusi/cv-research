from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from ..config import Settings
from ..llm import LLMClient
from ..orchestrators import make_orchestrator
from ..workspace import default_workspace
from .report import summarize
from .scoring import score_case

DEFAULT_CASES = Path(__file__).resolve().parents[3] / "benchmark" / "cases.jsonl"


def load_cases(path: Path | str = DEFAULT_CASES, categories: list[str] | None = None,
               ids: list[str] | None = None, limit: int | None = None) -> list[dict]:
    cases = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if categories:
        cases = [c for c in cases if c["category"] in categories]
    if ids:
        cases = [c for c in cases if c["id"] in ids]
    return cases[:limit] if limit else cases


def run_case(case: dict, llm: LLMClient, settings: Settings, repeat: int = 0) -> dict:
    orch = make_orchestrator(llm, settings)
    res = orch.run(case["prompt"], default_workspace())
    score = score_case(case, res)
    return {"case": case, "repeat": repeat, "result": res.to_dict(), "score": score.to_dict()}


def run_benchmark(cases: list[dict], llm: LLMClient, settings: Settings, out_dir: Path | None = None,
                  repeats: int = 1, progress: bool = True) -> tuple[list[dict], dict]:
    jobs = [(c, r) for r in range(repeats) for c in cases]
    records: list[dict] = []
    t0 = time.perf_counter()

    def _do(job):
        case, r = job
        rec = run_case(case, llm, settings, r)
        if progress:
            sc = rec["score"]
            mark = "PASS" if sc["task_success"] else "FAIL"
            print(f"[{settings.mode}] {case['id']:<6} r{r} {mark} {','.join(sc['errors'])}", flush=True)
        return rec

    with ThreadPoolExecutor(max_workers=max(1, settings.bench_concurrency)) as ex:
        records = list(ex.map(_do, jobs))

    summary = summarize(records)
    summary["meta"] = {
        "settings": settings.public_dict(),
        "num_cases": len(cases),
        "repeats": repeats,
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
    return root / f"{stamp}_{settings.mode}_{model}"
