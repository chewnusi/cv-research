from cvorch.bench.scoring import arg_errors, compare_steps, same_phrase, score_case, values_equal
from cvorch.orchestrators import RunResult
from cvorch.tools import ToolCallRecord


def S(tool, **args):
    return {"tool": tool, "args": args}


def test_values_equal_normalization():
    assert values_equal("dataset", "cats", "./data/cats/")
    assert values_equal("project_name", "Traffic QA", "'traffic qa'")
    assert values_equal("epochs", 50, 50.0)
    assert not values_equal("epochs", 50, 100)
    assert values_equal("text_prompt", "car", "Cars")
    assert values_equal("text_prompt", "bus", "the buses")
    assert not values_equal("text_prompt", "car", "truck")


def test_same_phrase():
    assert same_phrase("red apple", "the red apples")
    assert not same_phrase("apple", "red apple")


def test_default_valued_extra_args_are_fine_others_are_hallucinated():
    assert arg_errors(S("train_yolo", dataset="parking"), S("train_yolo", dataset="parking", epochs=100)) == []
    assert arg_errors(S("train_yolo", dataset="parking"), S("train_yolo", dataset="parking", epochs=50)) == [
        "hallucinated_parameter:epochs"]
    assert arg_errors(S("segment_dataset", dataset="cats", text_prompt="cat"),
                      S("segment_dataset", dataset="cats", text_prompt="cat", confidence=0.9)) == [
        "hallucinated_parameter:confidence"]


def test_class_name_defaults_to_text_prompt():
    exp = S("segment_dataset", dataset="cats", text_prompt="cat")
    assert arg_errors(exp, S("segment_dataset", dataset="cats", text_prompt="cat", class_name="cat")) == []
    assert arg_errors(exp, S("segment_dataset", dataset="cats", text_prompt="cat", class_name="feline")) == [
        "hallucinated_parameter:class_name"]
    exp_named = S("segment_dataset", dataset="traffic", text_prompt="bus", class_name="city_bus")
    assert arg_errors(exp_named, S("segment_dataset", dataset="traffic", text_prompt="bus")) == [
        "missing_parameter:class_name"]


def test_omitted_arg_equal_to_default_is_fine():
    assert arg_errors(S("train_yolo", dataset="x", epochs=100), S("train_yolo", dataset="x")) == []
    assert arg_errors(S("train_yolo", dataset="x", epochs=50), S("train_yolo", dataset="x")) == [
        "missing_parameter:epochs"]
    assert arg_errors(S("segment_dataset", dataset="x", text_prompt="cat"), S("segment_dataset", dataset="x")) == [
        "missing_parameter:text_prompt"]


def test_compare_steps_error_types():
    exp = [S("segment_dataset", dataset="cats", text_prompt="cat"), S("export_yolo", dataset="cats"),
           S("train_yolo", dataset="cats")]
    assert compare_steps(exp, exp).exact
    assert "wrong_order" in compare_steps(exp, [exp[0], exp[2], exp[1]]).errors
    assert "missing_step" in compare_steps(exp, [exp[0], exp[2]]).errors
    assert "extra_step" in compare_steps(exp, exp + [S("get_job_status", job_id="x")]).errors
    assert "wrong_tool" in compare_steps(exp[:1], [S("train_yolo", dataset="cats")]).errors
    assert "hallucinated_tool" in compare_steps(exp[:1], [S("magic_tool")]).errors
    m = compare_steps(exp, [exp[0], exp[1], S("train_yolo", dataset="dogs")])
    assert m.tools_ok and m.order_ok and not m.params_ok and "wrong_parameter:dataset" in m.errors


# ------------------------------------------------------------------ case-level scoring

EXEC_CASE = {"id": "x", "category": "c", "expect": "execute",
             "plans": [[S("segment_dataset", dataset="cats", text_prompt="cat"), S("export_yolo", dataset="cats")]]}
ABSTAIN_CASE = {"id": "y", "category": "late_failure", "expect": "abstain", "reason": "invalid_value"}


def _rec(tool, ok=True, **args):
    return ToolCallRecord(tool, args, ok, output={} if ok else None, error=None if ok else "boom")


def test_baseline_partial_side_effects_on_should_abstain_case():
    res = RunResult("baseline", "baseline", "p", calls=[
        _rec("segment_dataset", dataset="cats", text_prompt="cat"), _rec("export_yolo", dataset="cats"),
        _rec("train_yolo", ok=False, dataset="cats", epochs=0)])
    s = score_case(ABSTAIN_CASE, res)
    assert not s.task_success and s.unsafe_execution and s.side_effects_on_failure
    assert {"acted_on_invalid_request", "partial_side_effects", "runtime_error"} <= set(s.errors)


def test_planned_blocked_should_abstain_case_is_a_true_catch():
    res = RunResult("planned", "planned", "p", plan={"status": "ok", "steps": [
        {"tool": "train_yolo", "arguments": {"dataset": "cats", "epochs": 0}}]}, plan_status="ok",
        blocked_by_validator=True)
    s = score_case(ABSTAIN_CASE, res)
    # the LLM decided to act (decision level), but nothing was executed (outcome level)
    assert s.abstain_ok is False and s.task_success and not s.unsafe_execution
    assert s.validator_true_catch and "acted_on_invalid_request" in s.errors
    assert "partial_side_effects" not in s.errors


def test_false_block_and_false_abstention():
    steps = [{"tool": s["tool"], "arguments": s["args"]} for s in EXEC_CASE["plans"][0]]
    res = RunResult("planned", "planned", "p", plan={"status": "ok", "steps": steps}, plan_status="ok",
                    blocked_by_validator=True)
    s = score_case(EXEC_CASE, res)
    assert s.plan_exact and not s.task_success and s.validator_true_catch is False and "false_block" in s.errors
    res = RunResult("planned", "planned", "p", plan={"status": "unsupported", "steps": []}, plan_status="unsupported")
    s = score_case(EXEC_CASE, res)
    assert s.decided_abstain and "false_abstention" in s.errors


def test_task_success_uses_executed_steps():
    res = RunResult("baseline", "baseline", "p", calls=[
        _rec("segment_dataset", dataset="cats", text_prompt="cat"), _rec("export_yolo", ok=False, dataset="cats")])
    s = score_case(EXEC_CASE, res)
    assert s.plan_exact and not s.task_success and "wrong_execution" in s.errors


def test_plan_parse_error_tag():
    res = RunResult("planned", "planned", "p", plan_status="parse_error", plan_parse_error=True)
    assert "plan_parse_error" in score_case(EXEC_CASE, res).errors
