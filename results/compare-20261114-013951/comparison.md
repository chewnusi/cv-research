Model: gpt-4o-mini; temperature: 0.0; repeats: 3, 3

Rates are over all case×repeat observations; 95% CIs are case-level bootstrap intervals (differences are paired by case).

### Decision-level metrics

| Metric | baseline | planned | Δ planned − baseline [95% CI] |
|---|---|---|---|
| Tool selection accuracy | 94.1% (n=423) [90.3%, 97.2%] | 92.0% (n=423) [87.7%, 95.7%] | -2.1 pp [-6.6, +1.9] |
| Parameter accuracy (given correct tools) | 100.0% (n=398) | 100.0% (n=389) | +0.0 pp [+0.0, +0.0] |
| Workflow/plan exact accuracy | 94.1% (n=423) [90.3%, 97.2%] | 92.0% (n=423) [87.7%, 95.7%] | -2.1 pp [-6.6, +1.9] |
| Correct abstention by the LLM (should-abstain requests) | 54.7% (n=285) [44.9%, 64.9%] | 99.0% (n=285) [96.8%, 100.0%] | +44.2 pp [+34.7, +54.0] |
| False abstention rate (valid requests) | 0.7% (n=423) [0.0%, 2.1%] | 7.8% (n=423) [4.0%, 12.1%] | +7.1 pp [+3.5, +11.1] |

### Outcome-level metrics

| Metric | baseline | planned | Δ planned − baseline [95% CI] |
|---|---|---|---|
| Task completion rate (all requests) | 81.5% (n=708) [76.4%, 86.2%] | 95.2% (n=708) [92.5%, 97.5%] | +13.7 pp [+7.9, +19.5] |
| Task completion (valid requests) | 97.4% (n=423) [94.8%, 99.3%] | 92.0% (n=423) [87.7%, 95.7%] | -5.4 pp [-9.9, -1.2] |
| Task completion (should-abstain requests: nothing executed) | 57.9% (n=285) [48.1%, 68.1%] | 100.0% (n=285) [100.0%, 100.0%] | +42.1 pp [+32.3, +51.6] |
| Cases with runtime tool errors | 8.2% (n=708) | 0.0% (n=708) | -8.2 pp [-11.7, -5.1] |

### Safety / control metrics

| Metric | baseline | planned | Δ planned − baseline [95% CI] |
|---|---|---|---|
| Execution on should-abstain requests | 42.1% (n=285) [31.9%, 51.9%] | 0.0% (n=285) [0.0%, 0.0%] | -42.1 pp [-51.6, -32.3] |
| Partial execution on late-failure workflows | 62.6% (n=99) [46.5%, 77.8%] | 0.0% (n=99) [0.0%, 0.0%] | -62.6 pp [-77.8, -46.5] |
| Failed requests that left side effects | 18.1% (n=708) [13.6%, 23.0%] | 0.1% (n=708) [0.0%, 0.4%] | -17.9 pp [-22.7, -13.0] |
| Plans blocked by validator | 0 | 3 | +3 |
|   ...correct catches | 0 | 3 | +3 |
|   ...false blocks (correct plan blocked) | 0 | 0 | +0 |

### Efficiency

| Metric | baseline | planned | Δ planned − baseline |
|---|---|---|---|
| Total latency mean, s | 3.646 | 1.915 | -1.731 |
| Total latency median, s | 2.936 | 1.577 | -1.358 |
| Total latency p95, s | 8.175 | 3.837 | -4.338 |
| LLM latency mean, s | 3.646 | 1.914 | -1.732 |
| LLM calls per request | 2.143 | 1.145 | -0.997 |
| Input tokens per request | 7429.008 | 3465.701 | -3963.308 |
| Output tokens per request | 101.703 | 58.263 | -43.441 |
| Total tokens per request | 7530.712 | 3523.963 | -4006.749 |
| Plan parse errors | 0 | 0 | +0.000 |
| System errors | 4 | 0 | -4.000 |

### Task completion rate (all requests) by suite

| Suite | baseline | planned | Δ planned − baseline |
|---|---|---|---|
| core | 87.3% (n=576) | 94.6% (n=576) | +7.3 pp |
| extended | 56.1% (n=132) | 97.7% (n=132) | +41.7 pp |

### Task completion (valid requests) by suite

| Suite | baseline | planned | Δ planned − baseline |
|---|---|---|---|
| core | 98.7% (n=372) | 91.7% (n=372) | -7.0 pp |
| extended | 88.2% (n=51) | 94.1% (n=51) | +5.9 pp |

### Task completion (should-abstain requests: nothing executed) by suite

| Suite | baseline | planned | Δ planned − baseline |
|---|---|---|---|
| core | 66.7% (n=204) | 100.0% (n=204) | +33.3 pp |
| extended | 35.8% (n=81) | 100.0% (n=81) | +64.2 pp |

### Failed requests that left side effects by suite

| Suite | baseline | planned | Δ planned − baseline |
|---|---|---|---|
| core | 12.2% (n=576) | 0.2% (n=576) | -12.0 pp |
| extended | 43.9% (n=132) | 0.0% (n=132) | -43.9 pp |

### Task completion rate (all requests) by language

| Language | baseline | planned | Δ planned − baseline |
|---|---|---|---|
| en | 81.5% (n=708) | 95.2% (n=708) | +13.7 pp |

### Task completion rate (all requests) by category

| Category | baseline | planned | Δ planned − baseline |
|---|---|---|---|
| ambiguous | 47.2% (n=36) | 100.0% (n=36) | +52.8 pp |
| cancel_job | 88.9% (n=27) | 100.0% (n=27) | +11.1 pp |
| complex_multi_step | 100.0% (n=42) | 95.2% (n=42) | -4.8 pp |
| gpu_quota | 52.4% (n=21) | 100.0% (n=21) | +47.6 pp |
| implicit_prerequisites | 97.6% (n=42) | 71.4% (n=42) | -26.2 pp |
| inexact_reference | 25.0% (n=48) | 100.0% (n=48) | +75.0 pp |
| invalid_parameter | 100.0% (n=36) | 100.0% (n=36) | +0.0 pp |
| job_status | 100.0% (n=30) | 96.7% (n=30) | -3.3 pp |
| labeling | 100.0% (n=30) | 93.3% (n=30) | -6.7 pp |
| late_failure | 44.0% (n=84) | 100.0% (n=84) | +56.0 pp |
| long_workflow | 79.2% (n=24) | 87.5% (n=24) | +8.3 pp |
| missing_optional | 100.0% (n=30) | 96.7% (n=30) | -3.3 pp |
| multi_step | 100.0% (n=60) | 91.7% (n=60) | -8.3 pp |
| nonexistent_reference | 100.0% (n=30) | 100.0% (n=30) | +0.0 pp |
| paraphrase | 97.8% (n=45) | 93.3% (n=45) | -4.4 pp |
| single_tool | 100.0% (n=45) | 95.6% (n=45) | -4.4 pp |
| tool_with_params | 93.8% (n=48) | 93.8% (n=48) | +0.0 pp |
| unsupported | 80.0% (n=30) | 100.0% (n=30) | +20.0 pp |

### Error analysis (number of case×repeat observations with each error)

| Error | baseline | planned | Δ planned − baseline |
|---|---|---|---|
| acted_on_invalid_request | 129 | 3 | -126 |
| extra_step | 21 | 1 | -20 |
| false_abstention | 3 | 33 | +30 |
| missing_step | 1 | 0 | -1 |
| partial_side_effects | 120 | 0 | -120 |
| runtime_error | 58 | 0 | -58 |
| system_error | 4 | 0 | -4 |
| wrong_execution | 8 | 1 | -7 |
| wrong_parameter | 5 | 0 | -5 |

### Validator issue codes in blocked plans

| Issue code | baseline | planned | Δ planned − baseline |
|---|---|---|---|
| request_value_not_used | 0 | 3 | +3 |
