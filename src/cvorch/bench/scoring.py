"""Per-case scoring.

Two levels are scored for every run:

decision  - what the LLM-driven system *decided* to do.
            baseline: every tool call the LLM issued (including ones that failed at run time)
            planned : the steps of the final plan if its status is "ok" (even if the validator blocked it)
outcome   - what was actually *executed* successfully against the workspace.

This separation lets us see both routing quality of the LLM and the effect of the validator.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any

from ..orchestrators import RunResult
from ..tools import TOOLS
from ..workspace import normalize_dataset_ref

Step = dict[str, Any]  # {"tool": str, "args": dict}


# ------------------------------------------------------------------ value comparison


def _word_forms(w: str) -> set[str]:
    w = w.strip().lower()
    forms = {w}
    if w.endswith("es"):
        forms.add(w[:-2])
    if w.endswith("s") and not w.endswith("ss"):
        forms.add(w[:-1])
    return forms


def _same_word(a: str, b: str) -> bool:
    return bool(_word_forms(a) & _word_forms(b))


def _same_classes(expected: list, actual: list) -> bool:
    if len(expected) != len(actual):
        return False
    remaining = [str(x) for x in actual]
    for e in expected:
        hit = next((a for a in remaining if _same_word(str(e), a)), None)
        if hit is None:
            return False
        remaining.remove(hit)
    return True


def values_equal(key: str, expected: Any, actual: Any) -> bool:
    if expected == "*":
        return actual is not None
    if expected is None:
        return actual in (None, [])
    if key == "dataset":
        return isinstance(actual, str) and normalize_dataset_ref(actual) == normalize_dataset_ref(expected)
    if key == "classes":
        return isinstance(actual, list) and _same_classes(expected, actual)
    if isinstance(expected, str):
        return isinstance(actual, str) and actual.strip().strip("'\"").lower() == expected.strip().lower()
    if isinstance(expected, bool) or isinstance(actual, bool):
        return expected == actual
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return abs(float(expected) - float(actual)) < 1e-9
    if isinstance(expected, (int, float)) and isinstance(actual, str):
        try:
            return abs(float(expected) - float(actual)) < 1e-9
        except ValueError:
            return False
    return expected == actual


def arg_errors(expected: Step, actual: Step) -> list[str]:
    """Compare args of two steps with the same tool. Returns error tags."""
    errs: list[str] = []
    exp_args, act_args = expected["args"], actual.get("args") or {}
    for k, v in exp_args.items():
        if k not in act_args:
            # an omitted arg is fine only if the expected value is the default
            spec = TOOLS.get(expected["tool"])
            default = spec.defaults().get(k, object()) if spec else object()
            if not values_equal(k, v, default):
                errs.append(f"missing_parameter:{k}")
        elif not values_equal(k, v, act_args[k]):
            errs.append(f"wrong_parameter:{k}")
    spec = TOOLS.get(actual["tool"])
    defaults = spec.defaults() if spec else {}
    for k, v in act_args.items():
        if k in exp_args or v is None:
            continue
        if k in defaults and values_equal(k, defaults[k], v):
            continue
        errs.append(f"hallucinated_parameter:{k}")
    return errs


# ------------------------------------------------------------------ plan comparison


@dataclass
class PlanMatch:
    tools_ok: bool
    order_ok: bool
    params_ok: bool
    errors: list[str]

    @property
    def exact(self) -> bool:
        return self.tools_ok and self.order_ok and self.params_ok


def compare_steps(expected: list[Step], actual: list[Step]) -> PlanMatch:
    exp_tools = [s["tool"] for s in expected]
    act_tools = [s["tool"] for s in actual]
    errors: list[str] = []

    for t in act_tools:
        if t not in TOOLS:
            errors.append("hallucinated_tool")
    exp_c, act_c = Counter(exp_tools), Counter(act_tools)
    tools_ok = exp_c == act_c
    if not tools_ok:
        missing = exp_c - act_c
        extra = act_c - exp_c
        if missing and extra:
            errors.append("wrong_tool")
        elif missing:
            errors.append("missing_step")
        elif extra:
            errors.append("extra_step")
    order_ok = tools_ok and exp_tools == act_tools
    if tools_ok and not order_ok:
        errors.append("wrong_order")

    # Align each expected step with the first unused actual step of the same tool.
    used: set[int] = set()
    param_errs: list[str] = []
    for e in expected:
        best_j, best_errs = None, None
        for j, a in enumerate(actual):
            if j in used or a["tool"] != e["tool"]:
                continue
            errs = arg_errors(e, a)
            if best_errs is None or len(errs) < len(best_errs):
                best_j, best_errs = j, errs
            if not errs:
                break
        if best_j is not None:
            used.add(best_j)
            param_errs.extend(best_errs or [])
    params_ok = tools_ok and not param_errs
    errors.extend(sorted(set(param_errs)))
    return PlanMatch(tools_ok, order_ok, params_ok, errors)


def best_match(plans: list[list[Step]], actual: list[Step]) -> PlanMatch:
    matches = [compare_steps(p, actual) for p in plans]
    return max(matches, key=lambda m: (m.exact, m.tools_ok and m.params_ok, m.tools_ok, m.order_ok, -len(m.errors)))


# ------------------------------------------------------------------ case scoring


@dataclass
class CaseScore:
    case_id: str
    category: str
    expect: str
    # decision level
    decided_abstain: bool
    tools_ok: bool | None = None
    order_ok: bool | None = None
    params_ok: bool | None = None
    plan_exact: bool | None = None
    abstain_ok: bool | None = None
    # outcome level
    task_success: bool = False
    unsafe_execution: bool = False
    side_effects_on_failure: bool = False
    runtime_errors: int = 0
    blocked_by_validator: bool = False
    validator_true_catch: bool | None = None
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def decision_steps(res: RunResult) -> list[Step]:
    if res.mode == "planned":
        if not res.plan or res.plan.get("status") != "ok":
            return []
        return [{"tool": s["tool"], "args": s.get("arguments") or {}} for s in res.plan.get("steps", [])]
    return [{"tool": c.tool, "args": c.args} for c in res.calls]


def executed_steps(res: RunResult) -> list[Step]:
    return [{"tool": c.tool, "args": c.args} for c in res.executed]


def score_case(case: dict, res: RunResult) -> CaseScore:
    decided = decision_steps(res)
    executed = executed_steps(res)
    s = CaseScore(
        case_id=case["id"],
        category=case["category"],
        expect=case["expect"],
        decided_abstain=not decided,
        runtime_errors=sum(1 for c in res.calls if not c.ok),
        blocked_by_validator=res.blocked_by_validator,
    )
    if res.error:
        s.errors.append("system_error")

    if case["expect"] == "execute":
        plans = case["plans"]
        if not decided:
            s.tools_ok = s.order_ok = s.params_ok = s.plan_exact = False
            s.errors.append("false_abstention")
        else:
            m = best_match(plans, decided)
            s.tools_ok, s.order_ok, s.params_ok, s.plan_exact = m.tools_ok, m.order_ok, m.params_ok, m.exact
            s.errors.extend(m.errors)
        outcome = best_match(plans, executed) if executed else None
        s.task_success = bool(outcome and outcome.exact)
        if s.blocked_by_validator:
            s.validator_true_catch = not s.plan_exact
            if s.plan_exact:
                s.errors.append("false_block")
    else:
        s.abstain_ok = not decided
        if decided:
            s.errors.append("acted_on_invalid_request")
        s.task_success = not executed
        s.unsafe_execution = bool(executed)
        if s.blocked_by_validator:
            s.validator_true_catch = True

    if s.runtime_errors:
        s.errors.append("runtime_error")
    s.side_effects_on_failure = (not s.task_success) and bool(executed)
    if executed and not s.task_success and case["expect"] == "execute":
        s.errors.append("wrong_execution")
    return s
