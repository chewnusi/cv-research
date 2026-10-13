from cvorch.bench.scoring import arg_errors, compare_steps, values_equal


def S(tool, **args):
    return {"tool": tool, "args": args}


def test_values_equal_normalization():
    assert values_equal("dataset", "cats", "./data/cats/")
    assert values_equal("classes", ["car", "bus"], ["Buses", "cars"])
    assert values_equal("project_name", "Traffic QA", "'traffic qa'")
    assert values_equal("epochs", 50, 50.0)
    assert not values_equal("epochs", 50, 100)


def test_default_valued_extra_args_are_fine_others_are_hallucinated():
    assert arg_errors(S("train_yolo", dataset="parking"), S("train_yolo", dataset="parking", epochs=100)) == []
    assert arg_errors(S("train_yolo", dataset="parking"), S("train_yolo", dataset="parking", epochs=50)) == [
        "hallucinated_parameter:epochs"]
    assert arg_errors(S("segment_dataset", dataset="cats"), S("segment_dataset", dataset="cats", classes=["cat"])) == [
        "hallucinated_parameter:classes"]


def test_omitted_arg_equal_to_default_is_fine():
    assert arg_errors(S("train_yolo", dataset="x", epochs=100), S("train_yolo", dataset="x")) == []
    assert arg_errors(S("train_yolo", dataset="x", epochs=50), S("train_yolo", dataset="x")) == [
        "missing_parameter:epochs"]


def test_compare_steps_error_types():
    exp = [S("segment_dataset", dataset="cats"), S("export_yolo", dataset="cats"), S("train_yolo", dataset="cats")]
    assert compare_steps(exp, exp).exact
    assert "wrong_order" in compare_steps(exp, [exp[0], exp[2], exp[1]]).errors
    assert "missing_step" in compare_steps(exp, [exp[0], exp[2]]).errors
    assert "extra_step" in compare_steps(exp, exp + [S("get_job_status", job_id="x")]).errors
    assert "wrong_tool" in compare_steps(exp[:1], [S("train_yolo", dataset="cats")]).errors
    assert "hallucinated_tool" in compare_steps(exp[:1], [S("magic_tool")]).errors
    m = compare_steps(exp, [exp[0], exp[1], S("train_yolo", dataset="dogs")])
    assert m.tools_ok and m.order_ok and not m.params_ok and "wrong_parameter:dataset" in m.errors
