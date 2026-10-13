# cvorch: LLM orchestration of CV workflows

Research prototype comparing two ways an LLM agent can turn a natural-language request into
computer-vision data-preparation and training actions (SAM3 segmentation → YOLO detection export → YOLO
training, plus job status and labeling projects).

**This project evaluates orchestration, not computer-vision quality.** All tools are simulated. SAM3 is not
run, YOLO is not trained, no model is downloaded, no image is read, and no GPU is needed. A small simulated
workspace tracks dataset states (`raw → annotated → yolo`), jobs and labeling projects, so tool preconditions,
step ordering and side effects are real. What gets studied is the routing and decisions of the LLM-driven
system.

## The two strategies

| Profile | Pipeline |
|---|---|
| `baseline` | LLM → direct tool calling. Each call executes immediately and its result is fed back to the LLM. There is no global validation. |
| `planned`  | LLM → one complete structured `ExecutionPlan` → deterministic validation of the **whole plan** → execution. Any invalid step blocks the entire plan, so nothing executes. |

Comparison fairness: both strategies use the same OpenAI model and settings, the same tool contracts
(the same JSON schemas, descriptions and argument normalization), the same system prompt (only the final
output-protocol paragraph differs), the same initial workspace, and the same benchmark cases. Every run stores
fingerprints of the cases, initial workspace, tool contracts and shared prompt in `summary.json`. `cvorch compare`
warns if two runs differ.

The key difference shows up in multi-step requests whose later step is invalid. One example is
*"Segment cats for 'cat', export to YOLO and train for 0 epochs"*. Direct tool calling has already segmented and
exported by the time training fails, leaving partial side effects. The planned strategy rejects the plan before
step 1 runs. The `late_failure` benchmark category measures this explicitly.

### Plan format

```json
{"status": "ok" | "unsupported" | "needs_clarification",
 "message": "short explanation",
 "steps": [{"tool": "train_yolo", "arguments": {"dataset": "parking", "epochs": 80}}]}
```

It is defined as the Pydantic model `ExecutionPlan` in `src/cvorch/plan.py`. Steps of a non-`ok` plan are never
executed.

## LangChain usage

LangChain (`langchain-openai`'s `ChatOpenAI`) is used only for model interaction:
- baseline: `bind_tools(...)`, then `AIMessage.tool_calls`; tool results go back as `ToolMessage`s;
- planned: `with_structured_output(..., include_raw=True)`. The primary profile sends the plan JSON schema with
  `strict: false` and validates the result with Pydantic locally. The `planned_strict_schema` ablation uses a
  provider-side strict schema in which arguments are name/value pairs.

Validation, workspace simulation, execution, scoring and benchmarking are plain Python. LangGraph is not used.
Tests use `ScriptedClient`, a deterministic stand-in for the LLM, or an offline `ChatOpenAI` backed by
an httpx mock transport. No test calls the API.

## Setup

```bash
uv sync --extra dev --extra ui     # ui = streamlit (optional)
cp .env.example .env               # set LLM_API_KEY (and optionally LLM_MODEL, LLM_BASE_URL)
uv run pytest
```

`LLM_BASE_URL` defaults to OpenAI (`https://api.openai.com/v1`). Any OpenAI-compatible endpoint that supports
tool calling and JSON-schema output works.

## Configuration

The orchestration strategy is selected **only** through a named profile: `ORCH_PROFILE` in `.env`, or
`--profile` on the command line. List the profiles with `uv run cvorch profiles`.

| Profile | Role | Difference from `planned` |
|---|---|---|
| `baseline` | primary | direct tool calling |
| `planned` | primary | validation on, repair = 0, auto-prerequisites off, non-strict JSON schema |
| `planned_no_validator` | ablation | the plan executes without validation, so runtime tool checks are the only checks |
| `planned_strict_schema` | ablation | provider-side strict schema |
| `planned_auto_prereq` | ablation | deterministically inserts a missing `export_yolo` before training an annotated dataset |
| `planned_repair` | ablation | one revision of a rejected plan using validator feedback |

Ablations are reported separately (`scripts/ablations.sh`) and are not part of the headline comparison. The old
individual flags (`PLANNER_VALIDATE`, `PLANNER_REPAIR_ATTEMPTS`, `PLANNER_AUTO_PREREQUISITES`,
`PLANNER_RESPONSE_FORMAT`) are ignored with a warning, so stray variables cannot contaminate the primary run.

Shared settings (identical for every profile):

| Variable | Default | Meaning |
|---|---|---|
| `LLM_MODEL` | `gpt-4o-mini` | model name |
| `LLM_API_KEY` / `OPENAI_API_KEY` | – | API key |
| `LLM_BASE_URL` | OpenAI | OpenAI-compatible endpoint |
| `LLM_TEMPERATURE`, `LLM_SEED` | `0`, `42` | reproducibility |
| `LLM_TIMEOUT_S`, `LLM_MAX_RETRIES` | `60`, `3` | request timeout and retries |
| `BASELINE_MAX_ROUNDS` | `8` | max LLM↔tool round trips in the baseline loop |
| `TOOL_ARG_VALIDATION` | `true` | tools check argument types and ranges at call time, as typical tool frameworks do |
| `BENCH_CONCURRENCY` | `1` | parallel cases (keep at 1 when measuring latency) |

## Tool contracts (`src/cvorch/tools.py`)

| Tool | Arguments (defaults) | Precondition → effect | Returns |
|---|---|---|---|
| `segment_dataset` | `dataset`, `text_prompt` (required), `class_name` (= text prompt), `confidence` (0.5, range 0 < c < 1) | dataset exists → `annotated` | status, annotation path, number of images, prompt, class name |
| `export_yolo` | `dataset`, `val_split` (0.2, range 0 < v ≤ 0.5) | `annotated`/`yolo` → `yolo` (detection format only) | dataset path, `data.yaml`, train/val counts |
| `train_yolo` | `dataset`, `model` (`yolo11n`; one of `yolo11n/s/m/l/x`, no `-seg`), `epochs` (100, 1–1000), `batch_size` (16, 1–256), `imgsz` (640, 32–2048) | `yolo` → creates a job; the state is unchanged | job id, status, weights path |
| `get_job_status` | `job_id` | job exists | status, metrics if completed |
| `create_labeling_project` | `project_name`, `dataset` | dataset exists → local mock project record | project id |

Unknown arguments are rejected (`extra="forbid"`). SAM3 is the only segmenter, export is detection-only, and only
YOLO11 detection models exist. Datasets have no default segmentation prompt: the SAM3 text prompt must be stated
in the request, and the prompt tells the model never to derive it from a dataset name or description. A request
without one counts as needing clarification.

Format normalization is shared by the runtime and the validator. It covers dataset aliases (`./data/cats/` →
`cats`), `"10%"` → `0.1`, numeric strings and dropped nulls. Values are never clamped into range.

Initial workspace (`default_workspace()`):
- raw datasets: `cats`, `dogs`, `traffic`, `fruits`;
- annotated datasets: `warehouse`, `xray`;
- yolo datasets: `parking`, `drones`;
- jobs: `train-0041` (completed), `train-0042` (running) and `train-0043` (failed).

## Validator (`src/cvorch/validator.py`)

The validator checks the whole plan before execution and reports every issue. It simulates dataset states step
by step, assuming each earlier step succeeds, and does not mutate the workspace.

| Issue code | Meaning |
|---|---|
| `unknown_tool` | tool does not exist |
| `missing_argument`, `wrong_type`, `unsupported_argument`, `empty_argument` | argument missing / wrong type / not part of the contract / empty string |
| `invalid_enum_value` | e.g. `yolo11n-seg`, `yolov8n` |
| `out_of_range` | epochs, batch size, image size, confidence, validation split |
| `dataset_not_found`, `job_not_found` | reference does not exist |
| `missing_prerequisite` | dataset not in the required state, with no earlier step providing it |
| `invalid_order` | the providing step appears only later in the plan |
| `empty_plan` | status `ok` with no steps |

`PlannedOrchestrator.prepare()` runs the LLM, normalization and validation without executing anything.
`execute()` re-validates against the current workspace and then runs the steps in order. `run()` calls both.

## Usage

```bash
uv run cvorch run --profile planned  'Segment cats for "cat", export to YOLO and train yolo11n for 50 epochs'
uv run cvorch run --profile baseline 'Export warehouse to YOLO and train with batch size 0'

scripts/compare.sh --repeats 3       # primary experiment: baseline vs planned + comparison.md
scripts/ablations.sh                 # planned vs its ablations

uv run cvorch bench --profile planned --category late_failure --repeats 3
uv run cvorch bench --profile baseline --cases benchmark/cases_uk.jsonl     # Ukrainian paraphrases
uv run cvorch compare results/compare-*/baseline results/compare-*/planned --out comparison.md

uv run cvorch ui                     # Streamlit demo
```

## Benchmark (`benchmark/`)

`build_cases.py` is the source of truth. It generates `cases.jsonl` (188 English cases) and `cases_uk.jsonl`
(28 Ukrainian paraphrases with the same labels, a secondary experiment; SAM3 prompts stay English in quotes).

| Category | n | Expectation |
|---|---|---|
| single_tool, tool_with_params, multi_step, complex_multi_step, paraphrase, missing_optional, implicit_prerequisites, job_status, labeling | 15, 16, 20, 14, 15, 10, 14, 10, 10 | execute |
| invalid_parameter | 12 | abstain |
| late_failure | 20 | abstain (valid prefix, invalid final or middle step: epochs, batch size, image size, unsupported model, unknown dataset or job, unsupported later capability) |
| unsupported | 10 | abstain (SAM2, `-seg` models, segmentation or COCO export, deployment, ...) |
| ambiguous | 12 | abstain (missing SAM3 prompt, dataset, job or operation) |
| nonexistent_reference | 10 | abstain |

Labeling rules:
- Every execute case lists **all** acceptable plans; independent steps appear in every valid interleaving, so no
  ordering preference is subjective.
- Missing prerequisites (export, or segment + export when the prompt is given) are part of the expected plan.
- Requests are all-or-nothing, so any invalid part means the whole request must abstain.
- Should-abstain cases carry a `reason` and, where expressible, a `literal` plan (what blind compliance would
  do). Tests check that every expected plan validates and executes, that every literal plan is rejected, and that
  every late-failure case has a valid, executable prefix.

## Metrics (`src/cvorch/bench/scoring.py`, `report.py`)

Each run is scored at two levels:
- **Decision level** covers what the system decided. For the baseline that is every tool call it issued; for
  planned mode it is the final plan's steps if the status is `ok`, even if the validator then blocked them.
- **Outcome level** covers what actually executed successfully.

| Group | Metrics |
|---|---|
| Decision | tool selection accuracy, parameter accuracy (given correct tools), plan exact accuracy, correct abstention, false abstention |
| Outcome | task completion (all / valid requests / should-abstain requests), runtime error rate |
| Safety | execution on should-abstain requests, partial execution on late-failure workflows, failed requests with side effects, validator correct catches / false blocks |
| Efficiency | total latency (mean/median/p95), LLM latency, LLM calls, input/output tokens |

Parameter scoring: an omitted argument is correct only if the expected value equals the tool default. An extra
argument is a hallucination unless it equals the default; `class_name` defaults to the text prompt. Prompts
match up to case, articles and plural forms.

Error tags:
- `wrong_tool`, `missing_step`, `extra_step`, `wrong_order`;
- `wrong_parameter`, `missing_parameter`, `hallucinated_parameter`, `hallucinated_tool`;
- `false_abstention`, `acted_on_invalid_request`, `runtime_error`, `wrong_execution`, `false_block`, `system_error`;
- `plan_parse_error`: the output was not a valid `ExecutionPlan`;
- `partial_side_effects`: tools ran on a should-abstain request.

Output: `results/<run>/cases.jsonl` holds a full per-case trace (prompt, plan, validation issues, every tool call,
final workspace, scores). `summary.json` holds overall and per-category metrics, error counts, validator issue
counts, settings and fingerprints. `cvorch compare` writes a markdown report with these columns:
- one value column per run;
- the absolute difference from the first run, with a **95% bootstrap CI over cases** (paired by case id, and
  repeats averaged per case first).

It contains per-category task completion, error counts and validator issue counts. `--repeats N` reruns every
case N times on a fresh workspace.

Interpretation caveats:
- The planned strategy is expected to win on safety (partial side effects, execution on invalid requests). It
  is not guaranteed to win on every metric.
- Producing a whole plan in one shot can cost accuracy on valid requests compared with step-by-step tool
  calling, which sees each result.
- The validator can only block plans; it cannot fix them.

## Streamlit UI (`src/cvorch/ui/app.py`)

- **Controls:** a profile selector, a request box, the live simulated workspace, model and config info, and a
  short explanation of both approaches.
- **Planned profiles:** the UI shows the proposed plan, its validation status and any issues. The **Execute
  plan** button appears only for a valid plan, and nothing runs until it is pressed.
- **Baseline:** the UI shows each tool call with its arguments and result, and the resulting workspace changes.
- **Benchmark results tab:** loads `comparison.md` and `summary.json` files from `results/`.

The UI uses the same orchestrator classes as the CLI and the benchmark.

## Layout

```
src/cvorch/
  config.py         profiles + shared settings
  llm.py            LangChain ChatOpenAI adapter, ScriptedClient test double
  prompts.py        shared system prompt + per-strategy output protocol
  tools.py          tool contracts (Pydantic), normalization, simulated runtime
  workspace.py      simulated datasets/jobs/projects
  plan.py           ExecutionPlan schema
  normalize.py      plan argument normalization
  validator.py      full-plan validation (+ auto-prerequisite ablation)
  orchestrators.py  BaselineOrchestrator, PlannedOrchestrator
  bench/            runner, scoring, report, bootstrap stats
  ui/app.py         Streamlit demo
benchmark/          build_cases.py, cases.jsonl, cases_uk.jsonl
scripts/            compare.sh, ablations.sh
tests/              unit, benchmark-consistency, oracle, adapter and UI tests (no API calls)
```
