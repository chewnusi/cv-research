Model: gpt-4o-mini; temperature: 0.0; repeats: 3, 3, 3, 3, 3, 3, 3, 3, 3

Rates are over all case×repeat observations; 95% CIs are case-level bootstrap intervals (differences are paired by case).

### Decision-level metrics

| Metric | planned | planned_no_repair | planned_no_grounding | planned_minimal | planned_no_validator | planned_strict_schema | planned_auto_prereq | baseline | baseline_step_guard | Δ planned_no_repair − planned [95% CI] | Δ planned_no_grounding − planned [95% CI] | Δ planned_minimal − planned [95% CI] | Δ planned_no_validator − planned [95% CI] | Δ planned_strict_schema − planned [95% CI] | Δ planned_auto_prereq − planned [95% CI] | Δ baseline − planned [95% CI] | Δ baseline_step_guard − planned [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Tool selection accuracy | 92.0% (n=423) [87.7%, 95.7%] | 91.2% (n=423) [86.5%, 95.3%] | 92.4% (n=423) [87.9%, 96.5%] | 90.5% (n=423) [85.6%, 95.0%] | 91.7% (n=423) [87.2%, 95.7%] | 95.0% (n=423) [92.0%, 97.6%] | 94.1% (n=423) [90.1%, 97.4%] | 94.1% (n=423) [90.3%, 97.2%] | 94.6% (n=423) [91.2%, 97.4%] | -0.7 pp [-2.8, +1.4] | +0.5 pp [-0.9, +1.9] | -1.4 pp [-3.5, +0.5] | -0.2 pp [-2.1, +1.7] | +3.1 pp [-0.0, +6.4] | +2.1 pp [+0.2, +4.5] | +2.1 pp [-1.9, +6.6] | +2.6 pp [-1.2, +6.6] |
| Parameter accuracy (given correct tools) | 100.0% (n=389) | 100.0% (n=386) | 100.0% (n=391) | 100.0% (n=383) | 100.0% (n=388) | 100.0% (n=402) | 100.0% (n=398) | 100.0% (n=398) | 99.2% (n=400) | +0.0 pp [+0.0, +0.0] | +0.0 pp [+0.0, +0.0] | +0.0 pp [+0.0, +0.0] | +0.0 pp [+0.0, +0.0] | +0.0 pp [+0.0, +0.0] | +0.0 pp [+0.0, +0.0] | +0.0 pp [+0.0, +0.0] | -0.8 pp [-2.3, +0.0] |
| Workflow/plan exact accuracy | 92.0% (n=423) [87.7%, 95.7%] | 91.2% (n=423) [86.5%, 95.3%] | 92.4% (n=423) [87.9%, 96.5%] | 90.5% (n=423) [85.6%, 95.0%] | 91.7% (n=423) [87.2%, 95.7%] | 95.0% (n=423) [92.0%, 97.6%] | 94.1% (n=423) [90.1%, 97.4%] | 94.1% (n=423) [90.3%, 97.2%] | 93.8% (n=423) [90.1%, 96.9%] | -0.7 pp [-2.8, +1.4] | +0.5 pp [-0.9, +1.9] | -1.4 pp [-3.5, +0.5] | -0.2 pp [-2.1, +1.7] | +3.1 pp [-0.0, +6.4] | +2.1 pp [+0.2, +4.5] | +2.1 pp [-1.9, +6.6] | +1.9 pp [-2.1, +6.1] |
| Correct abstention by the LLM (should-abstain requests) | 99.0% (n=285) [96.8%, 100.0%] | 67.7% (n=285) [58.6%, 76.8%] | 81.0% (n=285) [73.0%, 88.4%] | 68.4% (n=285) [59.3%, 77.5%] | 68.1% (n=285) [59.0%, 77.5%] | 99.0% (n=285) [96.8%, 100.0%] | 99.0% (n=285) [96.8%, 100.0%] | 54.7% (n=285) [44.9%, 64.9%] | 54.0% (n=285) [44.2%, 64.6%] | -31.2 pp [-40.4, -22.5] | -17.9 pp [-25.3, -10.9] | -30.5 pp [-39.3, -21.4] | -30.9 pp [-40.0, -22.1] | +0.0 pp [+0.0, +0.0] | +0.0 pp [+0.0, +0.0] | -44.2 pp [-54.0, -34.4] | -44.9 pp [-54.7, -35.1] |
| False abstention rate (valid requests) | 7.8% (n=423) [4.0%, 12.1%] | 6.6% (n=423) [3.3%, 10.4%] | 6.9% (n=423) [3.1%, 11.1%] | 8.3% (n=423) [4.3%, 12.8%] | 7.1% (n=423) [3.5%, 11.1%] | 5.0% (n=423) [2.4%, 8.0%] | 5.4% (n=423) [2.4%, 9.0%] | 0.7% (n=423) [0.0%, 2.1%] | 0.5% (n=423) [0.0%, 1.4%] | -1.2 pp [-3.1, +0.7] | -0.9 pp [-2.4, +0.5] | +0.5 pp [-1.4, +2.4] | -0.7 pp [-2.1, +0.9] | -2.8 pp [-6.1, +0.0] | -2.4 pp [-4.7, -0.5] | -7.1 pp [-11.1, -3.5] | -7.3 pp [-11.3, -3.8] |

### Outcome-level metrics

| Metric | planned | planned_no_repair | planned_no_grounding | planned_minimal | planned_no_validator | planned_strict_schema | planned_auto_prereq | baseline | baseline_step_guard | Δ planned_no_repair − planned [95% CI] | Δ planned_no_grounding − planned [95% CI] | Δ planned_minimal − planned [95% CI] | Δ planned_no_validator − planned [95% CI] | Δ planned_strict_schema − planned [95% CI] | Δ planned_auto_prereq − planned [95% CI] | Δ baseline − planned [95% CI] | Δ baseline_step_guard − planned [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Task completion rate (all requests) | 95.2% (n=708) [92.5%, 97.5%] | 94.8% (n=708) [92.0%, 97.3%] | 87.8% (n=708) [83.6%, 92.0%] | 87.3% (n=708) [82.9%, 91.2%] | 84.0% (n=708) [79.5%, 88.4%] | 97.0% (n=708) [95.1%, 98.6%] | 96.5% (n=708) [93.9%, 98.5%] | 81.5% (n=708) [76.4%, 86.2%] | 88.6% (n=708) [84.6%, 92.4%] | -0.4 pp [-1.8, +0.8] | -7.3 pp [-10.9, -4.1] | -7.9 pp [-11.6, -4.7] | -11.2 pp [-15.5, -7.2] | +1.8 pp [-0.1, +3.8] | +1.3 pp [+0.0, +2.7] | -13.7 pp [-19.4, -7.9] | -6.6 pp [-11.3, -1.8] |
| Task completion (valid requests) | 92.0% (n=423) [87.7%, 95.7%] | 91.2% (n=423) [86.5%, 95.3%] | 92.4% (n=423) [87.9%, 96.5%] | 90.5% (n=423) [85.6%, 95.0%] | 91.7% (n=423) [87.2%, 95.7%] | 95.0% (n=423) [92.0%, 97.6%] | 94.1% (n=423) [90.1%, 97.4%] | 97.4% (n=423) [94.8%, 99.3%] | 97.9% (n=423) [95.7%, 99.5%] | -0.7 pp [-2.8, +1.4] | +0.5 pp [-0.9, +1.9] | -1.4 pp [-3.5, +0.5] | -0.2 pp [-2.1, +1.7] | +3.1 pp [-0.0, +6.4] | +2.1 pp [+0.2, +4.5] | +5.4 pp [+1.2, +9.9] | +5.9 pp [+1.9, +10.2] |
| Task completion (should-abstain requests: nothing executed) | 100.0% (n=285) [100.0%, 100.0%] | 100.0% (n=285) [100.0%, 100.0%] | 81.0% (n=285) [73.0%, 88.4%] | 82.5% (n=285) [74.7%, 89.8%] | 72.6% (n=285) [63.5%, 81.0%] | 100.0% (n=285) [100.0%, 100.0%] | 100.0% (n=285) [100.0%, 100.0%] | 57.9% (n=285) [48.1%, 68.1%] | 74.7% (n=285) [66.3%, 83.5%] | +0.0 pp [+0.0, +0.0] | -18.9 pp [-26.3, -11.6] | -17.5 pp [-25.3, -10.5] | -27.4 pp [-36.1, -19.3] | +0.0 pp [+0.0, +0.0] | +0.0 pp [+0.0, +0.0] | -42.1 pp [-51.6, -32.3] | -25.3 pp [-34.0, -16.8] |
| Cases with runtime tool errors | 0.0% (n=708) | 0.0% (n=708) | 0.0% (n=708) | 0.0% (n=708) | 5.9% (n=708) | 0.0% (n=708) | 0.0% (n=708) | 8.2% (n=708) | 17.5% (n=708) | +0.0 pp [+0.0, +0.0] | +0.0 pp [+0.0, +0.0] | +0.0 pp [+0.0, +0.0] | +5.9 pp [+3.2, +9.0] | +0.0 pp [+0.0, +0.0] | +0.0 pp [+0.0, +0.0] | +8.2 pp [+5.1, +11.7] | +17.5 pp [+12.9, +22.5] |

### Safety / control metrics

| Metric | planned | planned_no_repair | planned_no_grounding | planned_minimal | planned_no_validator | planned_strict_schema | planned_auto_prereq | baseline | baseline_step_guard | Δ planned_no_repair − planned [95% CI] | Δ planned_no_grounding − planned [95% CI] | Δ planned_minimal − planned [95% CI] | Δ planned_no_validator − planned [95% CI] | Δ planned_strict_schema − planned [95% CI] | Δ planned_auto_prereq − planned [95% CI] | Δ baseline − planned [95% CI] | Δ baseline_step_guard − planned [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Execution on should-abstain requests | 0.0% (n=285) [0.0%, 0.0%] | 0.0% (n=285) [0.0%, 0.0%] | 18.9% (n=285) [11.6%, 27.0%] | 17.5% (n=285) [10.2%, 25.3%] | 27.4% (n=285) [18.9%, 36.5%] | 0.0% (n=285) [0.0%, 0.0%] | 0.0% (n=285) [0.0%, 0.0%] | 42.1% (n=285) [31.9%, 51.9%] | 25.3% (n=285) [16.5%, 33.7%] | +0.0 pp [+0.0, +0.0] | +18.9 pp [+11.9, +26.3] | +17.5 pp [+10.5, +25.3] | +27.4 pp [+19.3, +36.5] | +0.0 pp [+0.0, +0.0] | +0.0 pp [+0.0, +0.0] | +42.1 pp [+32.3, +51.6] | +25.3 pp [+16.8, +34.0] |
| Partial execution on late-failure workflows | 0.0% (n=99) [0.0%, 0.0%] | 0.0% (n=99) [0.0%, 0.0%] | 19.2% (n=99) [7.1%, 33.3%] | 16.2% (n=99) [5.1%, 29.3%] | 41.4% (n=99) [24.2%, 57.6%] | 0.0% (n=99) [0.0%, 0.0%] | 0.0% (n=99) [0.0%, 0.0%] | 62.6% (n=99) [46.5%, 77.8%] | 62.6% (n=99) [45.5%, 78.8%] | +0.0 pp [+0.0, +0.0] | +19.2 pp [+7.1, +33.3] | +16.2 pp [+5.1, +29.3] | +41.4 pp [+25.3, +58.6] | +0.0 pp [+0.0, +0.0] | +0.0 pp [+0.0, +0.0] | +62.6 pp [+46.5, +77.8] | +62.6 pp [+45.5, +78.8] |
| Failed requests that left side effects | 0.1% (n=708) [0.0%, 0.4%] | 0.4% (n=708) [0.0%, 1.3%] | 8.1% (n=708) [4.8%, 11.6%] | 7.2% (n=708) [4.2%, 10.7%] | 11.0% (n=708) [7.3%, 15.0%] | 0.0% (n=708) [0.0%, 0.0%] | 0.3% (n=708) [0.0%, 0.9%] | 18.1% (n=708) [13.6%, 23.0%] | 11.2% (n=708) [7.5%, 15.1%] | +0.3 pp [+0.0, +0.8] | +7.9 pp [+4.8, +11.6] | +7.1 pp [+4.0, +10.5] | +10.9 pp [+7.1, +15.1] | -0.1 pp [-0.4, +0.0] | +0.1 pp [+0.0, +0.4] | +17.9 pp [+13.0, +22.7] | +11.0 pp [+7.2, +15.1] |
| Plans blocked by validator | 3 | 101 | 0 | 44 | 0 | 3 | 3 | 0 | 0 | +98 | -3 | +41 | -3 | +0 | +0 | -3 | -3 |
|   ...correct catches | 3 | 101 | 0 | 44 | 0 | 3 | 3 | 0 | 0 | +98 | -3 | +41 | -3 | +0 | +0 | -3 | -3 |
|   ...false blocks (correct plan blocked) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | +0 | +0 | +0 | +0 | +0 | +0 | +0 | +0 |

### Efficiency

| Metric | planned | planned_no_repair | planned_no_grounding | planned_minimal | planned_no_validator | planned_strict_schema | planned_auto_prereq | baseline | baseline_step_guard | Δ planned_no_repair − planned | Δ planned_no_grounding − planned | Δ planned_minimal − planned | Δ planned_no_validator − planned | Δ planned_strict_schema − planned | Δ planned_auto_prereq − planned | Δ baseline − planned | Δ baseline_step_guard − planned |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Total latency mean, s | 1.915 | 1.725 | 1.771 | 1.861 | 1.646 | 3.120 | 2.111 | 3.646 | 3.618 | -0.190 | -0.144 | -0.054 | -0.269 | +1.205 | +0.196 | +1.731 | +1.703 |
| Total latency median, s | 1.577 | 1.513 | 1.496 | 1.458 | 1.435 | 2.111 | 1.743 | 2.936 | 2.745 | -0.064 | -0.082 | -0.120 | -0.143 | +0.534 | +0.166 | +1.358 | +1.168 |
| Total latency p95, s | 3.837 | 3.021 | 3.630 | 4.052 | 3.180 | 7.945 | 4.215 | 8.175 | 8.382 | -0.816 | -0.207 | +0.215 | -0.657 | +4.107 | +0.378 | +4.338 | +4.545 |
| LLM latency mean, s | 1.914 | 1.724 | 1.770 | 1.860 | 1.645 | 3.119 | 2.110 | 3.646 | 3.618 | -0.190 | -0.144 | -0.054 | -0.269 | +1.205 | +0.196 | +1.732 | +1.704 |
| LLM calls per request | 1.145 | 1 | 1.061 | 1 | 1 | 1.165 | 1.136 | 2.143 | 2.082 | -0.145 | -0.085 | -0.145 | -0.145 | +0.020 | -0.010 | +0.997 | +0.936 |
| Input tokens per request | 3465.701 | 2999.161 | 3193.896 | 2999.161 | 2999.161 | 3930.667 | 3454.404 | 7429.008 | 7193.359 | -466.540 | -271.805 | -466.540 | -466.540 | +464.966 | -11.297 | +3963.308 | +3727.658 |
| Output tokens per request | 58.263 | 53.006 | 54.664 | 52.452 | 52.233 | 80.254 | 57.465 | 101.703 | 101.465 | -5.257 | -3.599 | -5.811 | -6.030 | +21.991 | -0.798 | +43.441 | +43.202 |
| Total tokens per request | 3523.963 | 3052.167 | 3248.559 | 3051.613 | 3051.394 | 4010.921 | 3511.869 | 7530.712 | 7294.823 | -471.797 | -275.404 | -472.350 | -472.569 | +486.958 | -12.095 | +4006.749 | +3770.860 |
| Plan parse errors | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |
| System errors | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 0 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +4.000 | +0.000 |

### Task completion rate (all requests) by suite

| Suite | planned | planned_no_repair | planned_no_grounding | planned_minimal | planned_no_validator | planned_strict_schema | planned_auto_prereq | baseline | baseline_step_guard | Δ planned_no_repair − planned | Δ planned_no_grounding − planned | Δ planned_minimal − planned | Δ planned_no_validator − planned | Δ planned_strict_schema − planned | Δ planned_auto_prereq − planned | Δ baseline − planned | Δ baseline_step_guard − planned |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| core | 94.6% (n=576) | 94.1% (n=576) | 89.4% (n=576) | 88.0% (n=576) | 87.7% (n=576) | 96.9% (n=576) | 96.2% (n=576) | 87.3% (n=576) | 91.7% (n=576) | -0.5 pp | -5.2 pp | -6.6 pp | -7.0 pp | +2.3 pp | +1.6 pp | -7.3 pp | -3.0 pp |
| extended | 97.7% (n=132) | 97.7% (n=132) | 81.1% (n=132) | 84.1% (n=132) | 68.2% (n=132) | 97.7% (n=132) | 97.7% (n=132) | 56.1% (n=132) | 75.0% (n=132) | +0.0 pp | -16.7 pp | -13.6 pp | -29.5 pp | +0.0 pp | +0.0 pp | -41.7 pp | -22.7 pp |

### Task completion (valid requests) by suite

| Suite | planned | planned_no_repair | planned_no_grounding | planned_minimal | planned_no_validator | planned_strict_schema | planned_auto_prereq | baseline | baseline_step_guard | Δ planned_no_repair − planned | Δ planned_no_grounding − planned | Δ planned_minimal − planned | Δ planned_no_validator − planned | Δ planned_strict_schema − planned | Δ planned_auto_prereq − planned | Δ baseline − planned | Δ baseline_step_guard − planned |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| core | 91.7% (n=372) | 90.9% (n=372) | 92.2% (n=372) | 90.0% (n=372) | 91.4% (n=372) | 95.2% (n=372) | 94.1% (n=372) | 98.7% (n=372) | 98.7% (n=372) | -0.8 pp | +0.5 pp | -1.6 pp | -0.3 pp | +3.5 pp | +2.4 pp | +7.0 pp | +7.0 pp |
| extended | 94.1% (n=51) | 94.1% (n=51) | 94.1% (n=51) | 94.1% (n=51) | 94.1% (n=51) | 94.1% (n=51) | 94.1% (n=51) | 88.2% (n=51) | 92.2% (n=51) | +0.0 pp | +0.0 pp | +0.0 pp | +0.0 pp | +0.0 pp | +0.0 pp | -5.9 pp | -2.0 pp |

### Task completion (should-abstain requests: nothing executed) by suite

| Suite | planned | planned_no_repair | planned_no_grounding | planned_minimal | planned_no_validator | planned_strict_schema | planned_auto_prereq | baseline | baseline_step_guard | Δ planned_no_repair − planned | Δ planned_no_grounding − planned | Δ planned_minimal − planned | Δ planned_no_validator − planned | Δ planned_strict_schema − planned | Δ planned_auto_prereq − planned | Δ baseline − planned | Δ baseline_step_guard − planned |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| core | 100.0% (n=204) | 100.0% (n=204) | 84.3% (n=204) | 84.3% (n=204) | 80.9% (n=204) | 100.0% (n=204) | 100.0% (n=204) | 66.7% (n=204) | 78.9% (n=204) | +0.0 pp | -15.7 pp | -15.7 pp | -19.1 pp | +0.0 pp | +0.0 pp | -33.3 pp | -21.1 pp |
| extended | 100.0% (n=81) | 100.0% (n=81) | 72.8% (n=81) | 77.8% (n=81) | 51.8% (n=81) | 100.0% (n=81) | 100.0% (n=81) | 35.8% (n=81) | 64.2% (n=81) | +0.0 pp | -27.2 pp | -22.2 pp | -48.2 pp | +0.0 pp | +0.0 pp | -64.2 pp | -35.8 pp |

### Failed requests that left side effects by suite

| Suite | planned | planned_no_repair | planned_no_grounding | planned_minimal | planned_no_validator | planned_strict_schema | planned_auto_prereq | baseline | baseline_step_guard | Δ planned_no_repair − planned | Δ planned_no_grounding − planned | Δ planned_minimal − planned | Δ planned_no_validator − planned | Δ planned_strict_schema − planned | Δ planned_auto_prereq − planned | Δ baseline − planned | Δ baseline_step_guard − planned |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| core | 0.2% (n=576) | 0.5% (n=576) | 6.1% (n=576) | 5.7% (n=576) | 6.8% (n=576) | 0.0% (n=576) | 0.4% (n=576) | 12.2% (n=576) | 8.0% (n=576) | +0.3 pp | +5.9 pp | +5.6 pp | +6.6 pp | -0.2 pp | +0.2 pp | +12.0 pp | +7.8 pp |
| extended | 0.0% (n=132) | 0.0% (n=132) | 16.7% (n=132) | 13.6% (n=132) | 29.5% (n=132) | 0.0% (n=132) | 0.0% (n=132) | 43.9% (n=132) | 25.0% (n=132) | +0.0 pp | +16.7 pp | +13.6 pp | +29.5 pp | +0.0 pp | +0.0 pp | +43.9 pp | +25.0 pp |

### Task completion rate (all requests) by language

| Language | planned | planned_no_repair | planned_no_grounding | planned_minimal | planned_no_validator | planned_strict_schema | planned_auto_prereq | baseline | baseline_step_guard | Δ planned_no_repair − planned | Δ planned_no_grounding − planned | Δ planned_minimal − planned | Δ planned_no_validator − planned | Δ planned_strict_schema − planned | Δ planned_auto_prereq − planned | Δ baseline − planned | Δ baseline_step_guard − planned |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| en | 95.2% (n=708) | 94.8% (n=708) | 87.8% (n=708) | 87.3% (n=708) | 84.0% (n=708) | 97.0% (n=708) | 96.5% (n=708) | 81.5% (n=708) | 88.6% (n=708) | -0.4 pp | -7.4 pp | -7.9 pp | -11.2 pp | +1.8 pp | +1.3 pp | -13.7 pp | -6.6 pp |

### Task completion rate (all requests) by category

| Category | planned | planned_no_repair | planned_no_grounding | planned_minimal | planned_no_validator | planned_strict_schema | planned_auto_prereq | baseline | baseline_step_guard | Δ planned_no_repair − planned | Δ planned_no_grounding − planned | Δ planned_minimal − planned | Δ planned_no_validator − planned | Δ planned_strict_schema − planned | Δ planned_auto_prereq − planned | Δ baseline − planned | Δ baseline_step_guard − planned |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ambiguous | 100.0% (n=36) | 100.0% (n=36) | 63.9% (n=36) | 66.7% (n=36) | 66.7% (n=36) | 100.0% (n=36) | 100.0% (n=36) | 47.2% (n=36) | 100.0% (n=36) | +0.0 pp | -36.1 pp | -33.3 pp | -33.3 pp | +0.0 pp | +0.0 pp | -52.8 pp | +0.0 pp |
| cancel_job | 100.0% (n=27) | 100.0% (n=27) | 88.9% (n=27) | 100.0% (n=27) | 88.9% (n=27) | 100.0% (n=27) | 100.0% (n=27) | 88.9% (n=27) | 88.9% (n=27) | +0.0 pp | -11.1 pp | +0.0 pp | -11.1 pp | +0.0 pp | +0.0 pp | -11.1 pp | -11.1 pp |
| complex_multi_step | 95.2% (n=42) | 95.2% (n=42) | 92.9% (n=42) | 92.9% (n=42) | 92.9% (n=42) | 97.6% (n=42) | 92.9% (n=42) | 100.0% (n=42) | 97.6% (n=42) | +0.0 pp | -2.4 pp | -2.4 pp | -2.4 pp | +2.4 pp | -2.4 pp | +4.8 pp | +2.4 pp |
| gpu_quota | 100.0% (n=21) | 100.0% (n=21) | 100.0% (n=21) | 100.0% (n=21) | 76.2% (n=21) | 100.0% (n=21) | 100.0% (n=21) | 52.4% (n=21) | 52.4% (n=21) | +0.0 pp | +0.0 pp | +0.0 pp | -23.8 pp | +0.0 pp | +0.0 pp | -47.6 pp | -47.6 pp |
| implicit_prerequisites | 71.4% (n=42) | 69.0% (n=42) | 78.6% (n=42) | 69.0% (n=42) | 69.0% (n=42) | 92.9% (n=42) | 90.5% (n=42) | 97.6% (n=42) | 100.0% (n=42) | -2.4 pp | +7.1 pp | -2.4 pp | -2.4 pp | +21.4 pp | +19.1 pp | +26.2 pp | +28.6 pp |
| inexact_reference | 100.0% (n=48) | 100.0% (n=48) | 52.1% (n=48) | 50.0% (n=48) | 41.7% (n=48) | 100.0% (n=48) | 100.0% (n=48) | 25.0% (n=48) | 85.4% (n=48) | +0.0 pp | -47.9 pp | -50.0 pp | -58.3 pp | +0.0 pp | +0.0 pp | -75.0 pp | -14.6 pp |
| invalid_parameter | 100.0% (n=36) | 100.0% (n=36) | 100.0% (n=36) | 100.0% (n=36) | 100.0% (n=36) | 100.0% (n=36) | 100.0% (n=36) | 100.0% (n=36) | 100.0% (n=36) | +0.0 pp | +0.0 pp | +0.0 pp | +0.0 pp | +0.0 pp | +0.0 pp | +0.0 pp | +0.0 pp |
| job_status | 96.7% (n=30) | 93.3% (n=30) | 96.7% (n=30) | 93.3% (n=30) | 96.7% (n=30) | 100.0% (n=30) | 96.7% (n=30) | 100.0% (n=30) | 100.0% (n=30) | -3.3 pp | +0.0 pp | -3.3 pp | +0.0 pp | +3.3 pp | +0.0 pp | +3.3 pp | +3.3 pp |
| labeling | 93.3% (n=30) | 96.7% (n=30) | 93.3% (n=30) | 100.0% (n=30) | 93.3% (n=30) | 86.7% (n=30) | 100.0% (n=30) | 100.0% (n=30) | 100.0% (n=30) | +3.3 pp | +0.0 pp | +6.7 pp | +0.0 pp | -6.7 pp | +6.7 pp | +6.7 pp | +6.7 pp |
| late_failure | 100.0% (n=84) | 100.0% (n=84) | 82.1% (n=84) | 83.3% (n=84) | 64.3% (n=84) | 100.0% (n=84) | 100.0% (n=84) | 44.0% (n=84) | 44.0% (n=84) | +0.0 pp | -17.9 pp | -16.7 pp | -35.7 pp | +0.0 pp | +0.0 pp | -56.0 pp | -56.0 pp |
| long_workflow | 87.5% (n=24) | 87.5% (n=24) | 87.5% (n=24) | 87.5% (n=24) | 87.5% (n=24) | 87.5% (n=24) | 87.5% (n=24) | 79.2% (n=24) | 87.5% (n=24) | +0.0 pp | +0.0 pp | +0.0 pp | +0.0 pp | +0.0 pp | +0.0 pp | -8.3 pp | +0.0 pp |
| missing_optional | 96.7% (n=30) | 86.7% (n=30) | 93.3% (n=30) | 83.3% (n=30) | 90.0% (n=30) | 96.7% (n=30) | 100.0% (n=30) | 100.0% (n=30) | 100.0% (n=30) | -10.0 pp | -3.3 pp | -13.3 pp | -6.7 pp | +0.0 pp | +3.3 pp | +3.3 pp | +3.3 pp |
| multi_step | 91.7% (n=60) | 90.0% (n=60) | 90.0% (n=60) | 91.7% (n=60) | 95.0% (n=60) | 96.7% (n=60) | 90.0% (n=60) | 100.0% (n=60) | 100.0% (n=60) | -1.7 pp | -1.7 pp | +0.0 pp | +3.3 pp | +5.0 pp | -1.7 pp | +8.3 pp | +8.3 pp |
| nonexistent_reference | 100.0% (n=30) | 100.0% (n=30) | 100.0% (n=30) | 100.0% (n=30) | 100.0% (n=30) | 100.0% (n=30) | 100.0% (n=30) | 100.0% (n=30) | 100.0% (n=30) | +0.0 pp | +0.0 pp | +0.0 pp | +0.0 pp | +0.0 pp | +0.0 pp | +0.0 pp | +0.0 pp |
| paraphrase | 93.3% (n=45) | 93.3% (n=45) | 93.3% (n=45) | 93.3% (n=45) | 93.3% (n=45) | 100.0% (n=45) | 93.3% (n=45) | 97.8% (n=45) | 95.6% (n=45) | +0.0 pp | +0.0 pp | +0.0 pp | +0.0 pp | +6.7 pp | +0.0 pp | +4.4 pp | +2.2 pp |
| single_tool | 95.6% (n=45) | 100.0% (n=45) | 100.0% (n=45) | 95.6% (n=45) | 97.8% (n=45) | 97.8% (n=45) | 95.6% (n=45) | 100.0% (n=45) | 100.0% (n=45) | +4.4 pp | +4.4 pp | +0.0 pp | +2.2 pp | +2.2 pp | +0.0 pp | +4.4 pp | +4.4 pp |
| tool_with_params | 93.8% (n=48) | 93.8% (n=48) | 93.8% (n=48) | 91.7% (n=48) | 93.8% (n=48) | 87.5% (n=48) | 93.8% (n=48) | 93.8% (n=48) | 95.8% (n=48) | +0.0 pp | +0.0 pp | -2.1 pp | +0.0 pp | -6.2 pp | +0.0 pp | +0.0 pp | +2.1 pp |
| unsupported | 100.0% (n=30) | 100.0% (n=30) | 100.0% (n=30) | 100.0% (n=30) | 100.0% (n=30) | 100.0% (n=30) | 100.0% (n=30) | 80.0% (n=30) | 80.0% (n=30) | +0.0 pp | +0.0 pp | +0.0 pp | +0.0 pp | +0.0 pp | +0.0 pp | -20.0 pp | -20.0 pp |

### Error analysis (number of case×repeat observations with each error)

| Error | planned | planned_no_repair | planned_no_grounding | planned_minimal | planned_no_validator | planned_strict_schema | planned_auto_prereq | baseline | baseline_step_guard | Δ planned_no_repair − planned | Δ planned_no_grounding − planned | Δ planned_minimal − planned | Δ planned_no_validator − planned | Δ planned_strict_schema − planned | Δ planned_auto_prereq − planned | Δ baseline − planned | Δ baseline_step_guard − planned |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| acted_on_invalid_request | 3 | 92 | 54 | 90 | 91 | 3 | 3 | 129 | 131 | +89 | +51 | +87 | +88 | +0 | +0 | +126 | +128 |
| extra_step | 1 | 4 | 2 | 0 | 0 | 0 | 2 | 21 | 20 | +3 | +1 | -1 | -1 | -1 | +1 | +20 | +19 |
| false_abstention | 33 | 28 | 29 | 35 | 30 | 21 | 23 | 3 | 2 | -5 | -4 | +2 | -3 | -12 | -10 | -30 | -31 |
| missing_step | 0 | 5 | 0 | 4 | 5 | 0 | 0 | 1 | 1 | +5 | +0 | +4 | +5 | +0 | +0 | +1 | +1 |
| partial_side_effects | 0 | 0 | 54 | 50 | 78 | 0 | 0 | 120 | 72 | +0 | +54 | +50 | +78 | +0 | +0 | +120 | +72 |
| runtime_error | 0 | 0 | 0 | 0 | 42 | 0 | 0 | 58 | 124 | +0 | +0 | +0 | +42 | +0 | +0 | +58 | +124 |
| system_error | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 0 | +0 | +0 | +0 | +0 | +0 | +0 | +4 | +0 |
| wrong_execution | 1 | 3 | 3 | 1 | 0 | 0 | 2 | 8 | 7 | +2 | +2 | +0 | -1 | -1 | +1 | +7 | +6 |
| wrong_parameter | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5 | 3 | +0 | +0 | +0 | +0 | +0 | +0 | +5 | +3 |
| wrong_tool | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | +0 | +1 | +1 | +0 | +0 | +0 | +0 | +0 |

### Validator issue codes in blocked plans

| Issue code | planned | planned_no_repair | planned_no_grounding | planned_minimal | planned_no_validator | planned_strict_schema | planned_auto_prereq | baseline | baseline_step_guard | Δ planned_no_repair − planned | Δ planned_no_grounding − planned | Δ planned_minimal − planned | Δ planned_no_validator − planned | Δ planned_strict_schema − planned | Δ planned_auto_prereq − planned | Δ baseline − planned | Δ baseline_step_guard − planned |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dataset_not_found | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | +1 | +0 | +0 | +0 | +0 | +0 | +0 | +0 |
| invalid_enum_value | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | +0 | +0 | +1 | +0 | +0 | +0 | +0 | +0 |
| job_not_cancellable | 0 | 18 | 0 | 18 | 0 | 0 | 0 | 0 | 0 | +18 | +0 | +18 | +0 | +0 | +0 | +0 | +0 |
| missing_prerequisite | 0 | 5 | 0 | 4 | 0 | 0 | 0 | 0 | 0 | +5 | +0 | +4 | +0 | +0 | +0 | +0 | +0 |
| out_of_range | 0 | 7 | 0 | 6 | 0 | 0 | 0 | 0 | 0 | +7 | +0 | +6 | +0 | +0 | +0 | +0 | +0 |
| project_exists | 0 | 6 | 0 | 6 | 0 | 0 | 0 | 0 | 0 | +6 | +0 | +6 | +0 | +0 | +0 | +0 | +0 |
| quota_exceeded | 0 | 8 | 0 | 9 | 0 | 0 | 0 | 0 | 0 | +8 | +0 | +9 | +0 | +0 | +0 | +0 | +0 |
| request_value_not_used | 3 | 12 | 0 | 0 | 0 | 3 | 3 | 0 | 0 | +9 | -3 | -3 | -3 | +0 | +0 | -3 | -3 |
| ungrounded_value | 0 | 50 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | +50 | +0 | +0 | +0 | +0 | +0 | +0 | +0 |
