"""Streamlit demo of the two orchestration strategies (simulated tools only).

    uv run cvorch ui          # or: uv run streamlit run src/cvorch/ui/app.py

The UI reuses the same orchestrator classes as the CLI and the benchmark. In planned mode the plan is
only prepared (LLM + validation); nothing is executed until the user presses "Execute plan".
"""
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from cvorch.config import PROFILES, Settings
from cvorch.llm import make_client
from cvorch.orchestrators import PlannedOrchestrator, RunResult, make_orchestrator
from cvorch.tools import MockToolRuntime
from cvorch.workspace import ACTIVE_JOB_STATUSES, default_workspace, snapshot_diff

RESULTS_DIR = Path("results")

EXPLANATION = """
**Baseline - direct tool calling.** The LLM calls tools one by one; each call is executed immediately and
its result is fed back. There is no global check, so if a later step is invalid, the earlier steps have
already changed the workspace (partial side effects).

**Planned - plan, validate, execute.** The LLM returns one complete structured `ExecutionPlan`. A
deterministic validator checks every step (tool names, argument types and ranges, dataset/job/project
references, dataset and job states and the GPU quota along the whole plan, and that every value is stated
in the request) *before anything runs*. The LLM gets one chance to repair the plan from the validator's
feedback; any remaining issue blocks the entire plan.

**Explicit-request policy.** Datasets are referenced by project name or exact local path, models by their
exact name; nothing the user did not say is filled in beyond tool defaults.

**Stopping a job.** Queued or running training jobs can be cancelled by id, either by a request
("cancel train-0042") or directly with the *Cancel* button in the jobs table.

All tools are simulated: no SAM3 inference, no YOLO training and no images are involved. The demo shows
the orchestration decisions only.
"""


def _init_state() -> None:
    st.session_state.setdefault("ws", default_workspace())
    st.session_state.setdefault("pending", None)  # planned: prepared, not yet executed RunResult
    st.session_state.setdefault("last", None)  # (RunResult, before_snapshot, after_snapshot)
    st.session_state.setdefault("notice", None)


def _reset() -> None:
    st.session_state.ws = default_workspace()
    st.session_state.pending = None
    st.session_state.last = None
    st.session_state.notice = None


def cancel_job(ws, job_id: str) -> str:
    """Direct user action (no LLM): cancel a queued or running job through the same tool runtime."""
    rec = MockToolRuntime(ws).call("cancel_job", {"job_id": job_id})
    return f"Job {job_id} cancelled." if rec.ok else f"Could not cancel {job_id}: {rec.error}"


def _on_cancel(job_id: str) -> None:
    st.session_state.notice = cancel_job(st.session_state.ws, job_id)


def _workspace_view(ws) -> None:
    snap = ws.snapshot()
    st.markdown("**Projects** (referenced by project name)")
    st.table([{"project": n, "cloud folder": p["cloud_folder"], "state": snap["datasets"][p["dataset"]]["state"],
               "images": snap["datasets"][p["dataset"]]["num_images"],
               "classes": ", ".join(snap["datasets"][p["dataset"]]["classes"])} for n, p in snap["projects"].items()])
    local = [d for d in snap["datasets"].values() if d.get("local_path")]
    if local:
        st.markdown("**Local datasets** (referenced by exact path)")
        st.table([{"path": d["local_path"], "state": d["state"], "images": d["num_images"],
                   "classes": ", ".join(d["classes"])} for d in local])
    st.markdown(f"**Training jobs** · GPU quota {len(ws.active_jobs())}/{ws.gpu_quota} in use")
    if st.session_state.get("notice"):
        st.info(st.session_state.notice)
    for j, x in snap["jobs"].items():
        c1, c2 = st.columns([4, 1])
        c1.markdown(f"`{j}` · {x['model']} on {x['dataset']}, {x['epochs']} epochs · **{x['status']}**")
        if x["status"] in ACTIVE_JOB_STATUSES:
            c2.button("Cancel", key=f"cancel-{j}", on_click=_on_cancel, args=(j,), help=f"Cancel {j}")


def _calls_view(res: RunResult) -> None:
    if not res.calls:
        st.info("No tool was executed.")
    for i, c in enumerate(res.calls, 1):
        icon = "✅" if c.ok else "❌"
        with st.expander(f"{icon} {i}. {c.tool}", expanded=not c.ok):
            st.markdown("Arguments")
            st.json(c.args)
            if c.ok:
                st.markdown("Result")
                st.json(c.output)
            else:
                st.error(c.error)


def _changes_view(before: dict, after: dict) -> None:
    changes = snapshot_diff(before, after)
    st.markdown("**Workspace changes**")
    if changes:
        for ch in changes:
            st.markdown(f"- {ch}")
    else:
        st.markdown("_none_")


def _stats(res: RunResult) -> None:
    st.caption(f"LLM calls: {res.llm_calls} · input tokens: {res.input_tokens} · output tokens: "
               f"{res.output_tokens} · LLM latency: {res.llm_latency_s:.2f}s · total: {res.total_latency_s:.2f}s")


def _plan_view(res: RunResult, validated: bool) -> None:
    if res.error:
        st.error(f"System error: {res.error}")
        return
    if res.plan_parse_error:
        st.error("The model did not return a valid ExecutionPlan; nothing will be executed.")
        st.json(res.plan_attempts)
        return
    plan = res.plan or {}
    st.markdown(f"**Plan status:** `{res.plan_status}`")
    if plan.get("message"):
        st.markdown(f"> {plan['message']}")
    steps = plan.get("steps", [])
    if steps:
        st.table([{"#": i, "tool": s["tool"], "arguments": json.dumps(s.get("arguments", {}))}
                  for i, s in enumerate(steps, 1)])
    if res.inserted_prerequisites:
        st.info(f"{res.inserted_prerequisites} prerequisite step(s) inserted automatically (ablation profile).")
    if res.plan_status != "ok":
        st.info("The planner abstained; nothing will be executed.")
    elif res.blocked_by_validator:
        st.error("Validation failed - the whole plan is blocked and no step will be executed.")
        st.table([{"step": i["step"], "code": i["code"], "message": i["message"]} for i in res.validation_issues])
    elif not validated:
        st.warning("Validation is disabled in this ablation profile; the plan would be executed unchecked.")
    else:
        st.success("Validation passed for every step.")


def _results_tab() -> None:
    dirs = sorted([p for p in RESULTS_DIR.glob("**/summary.json") if "archive" not in p.relative_to(RESULTS_DIR).parts],
                  reverse=True) if RESULTS_DIR.exists() else []
    comps = sorted(RESULTS_DIR.glob("*/comparison.md"), reverse=True) if RESULTS_DIR.exists() else []
    if comps:
        choice = st.selectbox("Comparison report", comps, format_func=lambda p: str(p.parent))
        st.markdown(choice.read_text())
    if dirs:
        choice = st.selectbox("Single run summary", dirs, format_func=lambda p: str(p.parent))
        summary = json.loads(choice.read_text())
        st.json(summary.get("meta", {}))
        st.json(summary["overall"], expanded=False)
    if not comps and not dirs:
        st.info("No benchmark results yet. Run `scripts/compare.sh` first.")


def main() -> None:
    st.set_page_config(page_title="CV orchestration demo", layout="wide")
    _init_state()
    st.title("LLM orchestration of CV workflows")

    with st.sidebar:
        profile = st.selectbox("Strategy / profile", list(PROFILES), index=list(PROFILES).index("planned"),
                               format_func=lambda n: f"{n}{'' if PROFILES[n].primary else ' (ablation)'}")
        st.caption(PROFILES[profile].description)
        settings = Settings.from_env(profile=profile)
        st.markdown("**Model configuration**")
        st.json({k: v for k, v in settings.public_dict().items()
                 if k.startswith("llm_") or k.startswith("planner_") or k in ("mode", "tool_arg_validation")},
                expanded=False)
        if not settings.llm_api_key:
            st.warning("LLM_API_KEY is not set (.env); requests will fail.")
        st.button("Reset workspace", on_click=_reset)

    demo, results = st.tabs(["Demo", "Benchmark results"])
    with results:
        _results_tab()
    with demo:
        with st.expander("How the two approaches differ", expanded=False):
            st.markdown(EXPLANATION)
        left, right = st.columns([3, 2])
        with right:
            st.subheader("Workspace (simulated)")
            _workspace_view(st.session_state.ws)
        with left:
            prompt = st.text_area("Request", placeholder='e.g. Segment cats for "cat", export to YOLO and train '
                                                         "yolo11n for 50 epochs.")
            if st.button("Submit", type="primary", disabled=not prompt.strip()):
                orch = make_orchestrator(make_client(settings), settings)
                ws = st.session_state.ws
                st.session_state.pending = st.session_state.last = None
                with st.spinner("Calling the LLM..."):
                    if isinstance(orch, PlannedOrchestrator):
                        st.session_state.pending = (orch, orch.prepare(prompt, ws))
                    else:
                        before = ws.snapshot()
                        res = orch.run(prompt, ws)
                        st.session_state.last = (res, before, ws.snapshot())
                st.rerun()

            pending = st.session_state.pending
            if pending:
                orch, res = pending
                st.subheader("Proposed plan (not executed)")
                _plan_view(res, orch.settings.planner_validate)
                _stats(res)
                if res.executable:
                    c1, c2 = st.columns(2)
                    if c1.button("Execute plan", type="primary"):
                        ws = st.session_state.ws
                        before = ws.snapshot()
                        orch.execute(res, ws)
                        st.session_state.pending = None
                        st.session_state.last = (res, before, ws.snapshot())
                        st.rerun()
                    if c2.button("Discard plan"):
                        st.session_state.pending = None
                        st.rerun()

            last = st.session_state.last
            if last:
                res, before, after = last
                st.subheader("Execution trace")
                if res.mode == "planned":
                    if res.blocked_by_validator:
                        st.error("Re-validation against the current workspace failed; nothing was executed.")
                        st.table(res.validation_issues)
                elif res.final_message:
                    st.markdown(f"**Assistant:** {res.final_message}")
                if res.error:
                    st.error(f"System error: {res.error}")
                _calls_view(res)
                _changes_view(before, after)
                _stats(res)


if __name__ == "__main__":
    main()
