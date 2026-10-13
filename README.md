# cvorch: LLM orchestration of CV workflows

Research prototype for comparing two ways an LLM agent can turn an English natural-language request into
computer-vision workflow actions (segmentation → YOLO export → YOLO training, job status, labeling projects):

| Mode (`ORCH_MODE`) | Pipeline |
|---|---|
| `baseline` | LLM → direct tool calling (each call executes immediately, results are fed back) |
| `planned`  | LLM → structured JSON plan → deterministic validator → executor |

Same LLM, same tool contracts, same workspace context and policy, same benchmark. Only the orchestration differs.

**The tools are mocks.** No images or videos are processed. A simulated workspace tracks dataset states
(`raw → annotated → yolo`) so tool preconditions and step ordering are real; the subject of the study is
the agent's routing and decisions, not CV quality.

## Setup

```bash
uv venv && uv pip install -e '.[dev]'
cp .env.example .env   # set LLM_BASE_URL / LLM_API_KEY / LLM_MODEL
uv run pytest
```

Any OpenAI-compatible endpoint works (OpenAI, OpenRouter, Ollama `http://localhost:11434/v1`, vLLM).
The model must support tool calling (baseline) and JSON output (planned).

## Usage

```bash
ORCH_MODE=planned  uv run cvorch run "Segment cats, export to YOLO and train yolo11n for 50 epochs"
ORCH_MODE=baseline uv run cvorch run "Train Faster R-CNN on parking"

ORCH_MODE=baseline uv run cvorch bench --out results/baseline
ORCH_MODE=planned  uv run cvorch bench --out results/planned --repeats 3
uv run cvorch compare results/baseline results/planned --out results/comparison.md

scripts/compare.sh            # both modes + comparison in one go (args forwarded to `bench`)
uv run cvorch bench --category invalid_parameter unsupported --limit 10
```

## Feature flags (env vars)

| Variable | Default | Meaning |
|---|---|---|
| `ORCH_MODE` | `planned` | `baseline` or `planned` (main independent variable) |
| `PLANNER_VALIDATE` | `true` | ablation: run the planned mode without the validator |
| `PLANNER_REPAIR_ATTEMPTS` | `0` | let the planner revise a rejected plan using validator feedback |
| `PLANNER_RESPONSE_FORMAT` | `json_object` | `json_object` (schema in prompt) or `json_schema` (strict: decoder enforces tool and argument names; value ranges are left to the validator) |
| `PLANNER_AUTO_PREREQUISITES` | `false` | deterministically insert missing `segment_dataset`/`export_yolo` steps from dataset state |
| `TOOL_ARG_VALIDATION` | `true` | tools check argument types/ranges at call time, as typical tool frameworks do. Turn off to simulate tools with no defensive checks |
| `BASELINE_MAX_ROUNDS` | `8` | max LLM↔tool round trips in baseline |
| `LLM_TEMPERATURE`, `LLM_SEED` | `0`, `42` | reproducibility |
| `BENCH_CONCURRENCY` | `1` | parallel cases (keep 1 when measuring latency) |

## Tools

| Tool | Arguments (defaults) | Precondition |
|---|---|---|
| `segment_dataset` | `dataset`, `model` (sam2), `classes` (none), `confidence` (0.5) | dataset exists |
| `export_yolo` | `dataset`, `val_split` (0.2), `task` (segment) | dataset annotated |
| `train_yolo` | `dataset`, `model` (yolo11n), `epochs` (100), `batch_size` (16), `imgsz` (640) | dataset in YOLO format |
| `get_job_status` | `job_id` | job exists |
| `create_labeling_project` | `project_name`, `dataset` | dataset exists (MockLabelingService) |

Tools take a dataset name instead of chaining output paths (`annotation_path` → `dataset_path`).
Dependencies are still enforced through dataset state, and scoring stays unambiguous.

Planned-mode pipeline: LLM plan → format normalization (`src/cvorch/normalize.py`: dataset aliases,
`"10%"` → `0.1`, nulls dropped; values never clamped) → optional prerequisite completion → validation → execution.

Validator rules (`src/cvorch/validator.py`): unknown tool, schema violations (required/type/range/enum/extra args),
unknown dataset, unknown job, dependency/ordering violations simulated across the plan, empty plan.

## Benchmark

`benchmark/cases.jsonl` holds 132 English requests in 13 categories, generated from `benchmark/build_cases.py`:
single tool, tool + parameters, multi-step, complex multi-step, paraphrase, missing optional parameters,
implicit prerequisites, job status, labeling, invalid parameters, unsupported operations, ambiguous requests,
and references to datasets or jobs that don't exist. Each valid case has one or more acceptable expected plans;
invalid, unsupported, and ambiguous cases expect the system to abstain (execute nothing).

## Metrics

The decision level scores what the LLM chose: baseline = every tool call it issued; planned = its final plan.
The outcome level scores what actually executed successfully.

- **Tool selection accuracy**: the multiset of tools matches the expected plan.
- **Parameter accuracy**: expected args match, and any other arg equals the tool default (otherwise it counts as hallucinated).
- **Plan exact accuracy**: correct tools, order, and parameters.
- **Correct abstention / false abstention**: on should-abstain cases and on valid cases, respectively.
- **Task completion**: what executed equals an expected plan, or nothing executed when abstention was expected.
- **Unsafe execution**: something executed on a should-abstain request.
- **Failed with side effects**: the task failed but some tools had already run (partial execution).
- **Validator blocks**: correct catches vs. wrongly blocked correct plans.
- **Latency, LLM calls, and tokens** per request.
- **Error analysis tags**: `wrong_tool`, `missing_step`, `extra_step`, `wrong_order`, `wrong_parameter`, `missing_parameter`,
  `hallucinated_parameter`, `hallucinated_tool`, `false_abstention`, `acted_on_invalid_request`, `runtime_error`,
  `wrong_execution`, `false_block`, `system_error`.

Per-case traces (prompt, plan, validation issues, every tool call, scores) are saved in `results/<run>/cases.jsonl`.
The oracle tests check the benchmark and scorer together: a perfect LLM must score 100% in both modes.
