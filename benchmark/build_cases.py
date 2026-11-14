"""Source of truth for the benchmark. Regenerate the JSONL files with:

    uv run python benchmark/build_cases.py

Each case is either
  expect="execute": `plans` lists every acceptable plan. Step order matters within a plan; independent
                    steps (different datasets, jobs, labeling projects) are listed in all valid
                    interleavings, so no ordering preference is subjective.
  expect="abstain": the system must not execute anything. `reason` is one of
                    invalid_value | unsupported | ambiguous | not_found.
                    `literal` (optional) is the plan a system would produce by literally complying with
                    the request; tests check that the deterministic validator rejects it, and for
                    late_failure cases that its first invalid step is not the first step (so step-by-step
                    execution would leave partial side effects).

Expected `args` list only what the request determines. Any other argument the system passes must equal the
tool default (class_name defaults to the text prompt), otherwise it counts as a hallucinated parameter.

Every case has `language` (en | uk | mixed) and `suite`:
  core      the original benchmark (relabelled for the explicit-request policy);
  extended  cases added with the richer workspace: job cancellation, the GPU quota, long multi-dataset
            workflows, local datasets and inexact references. Reported separately from `core`.
`late` (abstain cases) marks requests whose literal plan is valid up to a later step, so step-by-step execution
leaves partial side effects.

Labeling rules (no subjective labels):
  * explicit-request policy: a dataset is referenced by a project name (token in the request) or the exact
    local path of a local dataset; models are named exactly; SAM3 prompts, job ids and project names are
    written in the request. Descriptions ("the weld scans", "the small model", "the running job") need
    clarification;
  * a SAM3 text prompt must be stated in the request; without it a segmentation request is ambiguous;
  * missing prerequisite steps (export before training an annotated dataset; segment+export before training
    a raw dataset when the prompt is given) are part of the expected plan;
  * a request is all-or-nothing: if any part is invalid/unsupported/unknown, the whole request must abstain.

Workspace (see cvorch.workspace.default_workspace):
  projects  raw: cats, dogs, traffic, fruits | annotated: warehouse, xray | yolo: parking, drones
  local     raw: /data/local/dock_cam | annotated: /data/local/solar_panels | yolo: /data/local/shelves
  jobs      train-0041 completed, train-0042 running, train-0043 failed; GPU quota 3 active jobs
"""
from __future__ import annotations

import json
from itertools import permutations
from pathlib import Path


def _step(tool, **args):
    return {"tool": tool, "args": args}


def S(ds, prompt, **kw):
    return _step("segment_dataset", dataset=ds, text_prompt=prompt, **kw)


def E(ds, **kw):
    return _step("export_yolo", dataset=ds, **kw)


def T(ds, **kw):
    return _step("train_yolo", dataset=ds, **kw)


def J(job):
    return _step("get_job_status", job_id=job)


def L(name, ds):
    return _step("create_labeling_project", project_name=name, dataset=ds)


def C(job):
    return _step("cancel_job", job_id=job)


DOCK, SOLAR, SHELVES = "/data/local/dock_cam", "/data/local/solar_panels", "/data/local/shelves"
QUOTA, ACTIVE_JOBS = 3, ("train-0042",)


def within_quota(plan):
    active = set(ACTIVE_JOBS)
    for i, st in enumerate(plan):
        if st["tool"] == "train_yolo":
            active.add(i)
            if len(active) > QUOTA:
                return False
        elif st["tool"] == "cancel_job":
            active.discard(st["args"]["job_id"])
    return True


def quota_filter(plans):
    return [p for p in plans if within_quota(p)]


def X(tool, **args):
    """A step with a tool that does not exist (only used in `literal` plans)."""
    return _step(tool, **args)


def interleave(*chains):
    """All merges of the chains that keep the order inside each chain."""
    chains = [list(c) for c in chains if c]
    if not chains:
        return [[]]
    out = []
    for i, ch in enumerate(chains):
        rest = chains[:i] + [ch[1:]] + chains[i + 1:]
        out += [[ch[0], *tail] for tail in interleave(*rest)]
    return out


def anyorder(*steps):
    return [list(p) for p in permutations(steps)]


CASES: list[dict] = []
UK_CASES: list[dict] = []
SUITE = "core"


def ok(cid, cat, prompt, *plans):
    CASES.append({"id": cid, "category": cat, "prompt": prompt, "expect": "execute", "language": "en",
                  "suite": SUITE, "plans": [list(p) for p in plans]})


def oks(cid, cat, prompt, plans):
    ok(cid, cat, prompt, *plans)


def no(cid, cat, prompt, reason, note, literal=None, late=False):
    case = {"id": cid, "category": cat, "prompt": prompt, "expect": "abstain", "language": "en", "suite": SUITE,
            "reason": reason, "note": note}
    if literal is not None:
        case["literal"] = list(literal)
    if late or cat == "late_failure":
        case["late"] = True
    CASES.append(case)


# ---------------------------------------------------------------- single tool
c = "single_tool"
ok("st-01", c, 'Segment the cats dataset with the text prompt "cat".', [S("cats", "cat")])
ok("st-02", c, "Run SAM3 on dogs to segment every dog.", [S("dogs", "dog")])
ok("st-03", c, "Segment the traffic dataset for cars.", [S("traffic", "car")])
ok("st-04", c, 'Annotate fruits automatically by segmenting "apple".', [S("fruits", "apple")])
ok("st-05", c, "Export the warehouse dataset to YOLO format.", [E("warehouse")])
ok("st-06", c, "Convert the xray annotations into a YOLO detection dataset.", [E("xray")])
ok("st-07", c, "Train a YOLO model on parking.", [T("parking")])
ok("st-08", c, "Start a training run on the drones dataset.", [T("drones")])
ok("st-09", c, "Segment pedestrians in traffic.", [S("traffic", "pedestrian")])
ok("st-10", c, "Fit YOLO to the parking data.", [T("parking")])
ok("st-11", c, "Export xray to YOLO.", [E("xray")])
ok("st-12", c, "Kick off YOLO training using drones.", [T("drones")])
ok("st-13", c, "Re-export the parking dataset to YOLO.", [E("parking")])
ok("st-14", c, 'Segment the warehouse images again, this time with the prompt "forklift".',
   [S("warehouse", "forklift")])
ok("st-15", c, "Use SAM3 to segment bananas in the fruits dataset.", [S("fruits", "banana")])

# ---------------------------------------------------------------- tool + parameters
c = "tool_with_params"
ok("tp-01", c, "Train YOLO on parking for 80 epochs.", [T("parking", epochs=80)])
ok("tp-02", c, "Train yolo11s on drones.", [T("drones", model="yolo11s")])
ok("tp-03", c, "Train yolo11m on parking for 150 epochs with batch size 32.",
   [T("parking", model="yolo11m", epochs=150, batch_size=32)])
ok("tp-04", c, "Train on drones with image size 1280 for 60 epochs.", [T("drones", imgsz=1280, epochs=60)])
ok("tp-05", c, 'Segment cats with the prompt "cat" and a confidence threshold of 0.7.',
   [S("cats", "cat", confidence=0.7)])
ok("tp-06", c, 'Segment the traffic dataset for "bus" and store the class as "city_bus".',
   [S("traffic", "bus", class_name="city_bus")])
ok("tp-07", c, "Export warehouse to YOLO with a validation split of 0.1.", [E("warehouse", val_split=0.1)])
ok("tp-08", c, "Train yolo11x on drones, 25 epochs, batch 8.",
   [T("drones", model="yolo11x", epochs=25, batch_size=8)])
ok("tp-09", c, "Train yolo11l on parking for 200 epochs at imgsz 960.",
   [T("parking", model="yolo11l", epochs=200, imgsz=960)])
ok("tp-10", c, 'Segment dogs for "dog" with confidence 0.35.', [S("dogs", "dog", confidence=0.35)])
ok("tp-11", c, "Export xray to YOLO with val_split 0.25.", [E("xray", val_split=0.25)])
ok("tp-12", c, "Train yolo11n on parking with batch size 64 and image size 512.",
   [T("parking", model="yolo11n", batch_size=64, imgsz=512)])
ok("tp-13", c, 'Segment fruits looking for "orange", name the class "orange_fruit", confidence 0.6.',
   [S("fruits", "orange", class_name="orange_fruit", confidence=0.6)])
ok("tp-14", c, "Train on drones for 1 epoch as a smoke test.", [T("drones", epochs=1)])
ok("tp-15", c, "Export the parking dataset again with a 0.5 validation split.", [E("parking", val_split=0.5)])
ok("tp-16", c, "Train yolo11s on parking for 1000 epochs with batch size 256 and imgsz 2048.",
   [T("parking", model="yolo11s", epochs=1000, batch_size=256, imgsz=2048)])

# ---------------------------------------------------------------- multi-step
c = "multi_step"
ok("ms-01", c, 'Segment cats for "cat" and export them to YOLO.', [S("cats", "cat"), E("cats")])
ok("ms-02", c, "Export warehouse to YOLO and then train on it.", [E("warehouse"), T("warehouse")])
ok("ms-03", c, 'Annotate fruits by segmenting "apple", then convert to YOLO format.',
   [S("fruits", "apple"), E("fruits")])
ok("ms-04", c, "Export xray to YOLO and train yolo11s for 40 epochs.",
   [E("xray"), T("xray", model="yolo11s", epochs=40)])
oks("ms-05", c, 'Segment traffic for "car" and create a labeling project "Traffic QA" so annotators can fix the masks.',
    anyorder(S("traffic", "car"), L("Traffic QA", "traffic")))
oks("ms-06", c, "Train on parking and create a labeling project called parking-audit for it.",
    anyorder(T("parking"), L("parking-audit", "parking")))
oks("ms-07", c, 'Segment cats for "cat" and dogs for "dog".', anyorder(S("cats", "cat"), S("dogs", "dog")))
oks("ms-08", c, "Export both warehouse and xray to YOLO.", anyorder(E("warehouse"), E("xray")))
ok("ms-09", c, 'Segment dogs with the prompt "dog", then export it with a 0.15 validation split.',
   [S("dogs", "dog"), E("dogs", val_split=0.15)])
oks("ms-10", c, "Check job train-0041 and start a new training on parking for 60 epochs.",
    anyorder(J("train-0041"), T("parking", epochs=60)))
oks("ms-11", c, "Train yolo11n and yolo11s on drones, 50 epochs each.",
    anyorder(T("drones", model="yolo11n", epochs=50), T("drones", model="yolo11s", epochs=50)))
oks("ms-12", c, "Train yolo11m on parking and yolo11s on drones.",
    anyorder(T("parking", model="yolo11m"), T("drones", model="yolo11s")))
oks("ms-13", c, "Get the status of train-0042 and train-0043.", anyorder(J("train-0042"), J("train-0043")))
ok("ms-14", c, "Re-export parking with a 0.1 validation split and then train yolo11s on it.",
   [E("parking", val_split=0.1), T("parking", model="yolo11s")])
ok("ms-15", c, 'Segment the xray scans again with the prompt "crack" and export them to YOLO.',
   [S("xray", "crack"), E("xray")])
oks("ms-16", c, 'Export warehouse to YOLO and create a labeling project named "warehouse-check" for warehouse.',
    anyorder(E("warehouse"), L("warehouse-check", "warehouse")))
oks("ms-17", c, "Train YOLO on parking for 30 epochs, then check job train-0042.",
    anyorder(T("parking", epochs=30), J("train-0042")))
ok("ms-18", c, "Export xray to YOLO and then train on it with batch size 8.",
   [E("xray"), T("xray", batch_size=8)])
ok("ms-19", c, 'Segment traffic for "pedestrian" with confidence 0.4 and export it with val_split 0.3.',
   [S("traffic", "pedestrian", confidence=0.4), E("traffic", val_split=0.3)])
oks("ms-20", c, "Train yolo11l on drones and also on parking, 120 epochs each.",
    anyorder(T("drones", model="yolo11l", epochs=120), T("parking", model="yolo11l", epochs=120)))

# ---------------------------------------------------------------- complex multi-step
c = "complex_multi_step"
ok("cm-01", c, 'Segment cats for "cat", export them to YOLO and train yolo11n for 50 epochs.',
   [S("cats", "cat"), E("cats"), T("cats", model="yolo11n", epochs=50)])
ok("cm-02", c, 'Segment the traffic dataset for "car", export to YOLO and train yolo11s for 100 epochs.',
   [S("traffic", "car"), E("traffic"), T("traffic", model="yolo11s", epochs=100)])
ok("cm-03", c, 'Take dogs, segment "dog" with confidence 0.6, export with a 0.1 validation split, and train '
               "yolo11m with batch size 32.",
   [S("dogs", "dog", confidence=0.6), E("dogs", val_split=0.1), T("dogs", model="yolo11m", batch_size=32)])
oks("cm-04", c, 'Segment fruits for "apple", export to YOLO, train yolo11n, and create a labeling project '
                '"fruit-review" for fruits.',
    interleave([S("fruits", "apple"), E("fruits"), T("fruits", model="yolo11n")], [L("fruit-review", "fruits")]))
oks("cm-05", c, "Export warehouse and xray to YOLO and train yolo11s on both.",
    interleave([E("warehouse"), T("warehouse", model="yolo11s")], [E("xray"), T("xray", model="yolo11s")]))
oks("cm-06", c, 'Segment cats ("cat") and dogs ("dog"), then export both to YOLO.',
    interleave([S("cats", "cat"), E("cats")], [S("dogs", "dog"), E("dogs")]))
oks("cm-07", c, 'Segment cats for "cat", export, train yolo11s for 30 epochs, and check job train-0041.',
    interleave([S("cats", "cat"), E("cats"), T("cats", model="yolo11s", epochs=30)], [J("train-0041")]))
ok("cm-08", c, "Export the xray scans with a 0.2 validation split, then train yolo11l at imgsz 1024 for 300 epochs.",
   [E("xray", val_split=0.2), T("xray", model="yolo11l", imgsz=1024, epochs=300)])
ok("cm-09", c, 'Segment traffic for "bus", export with 0.25 validation, and train yolo11x for 10 epochs with '
               "batch size 4.",
   [S("traffic", "bus"), E("traffic", val_split=0.25), T("traffic", model="yolo11x", epochs=10, batch_size=4)])
ok("cm-10", c, 'Full pipeline on fruits with the prompt "orange": segmentation, YOLO export and yolo11n training '
               "for 75 epochs.",
   [S("fruits", "orange"), E("fruits"), T("fruits", model="yolo11n", epochs=75)])
oks("cm-11", c, 'Segment dogs for "dog", export them, and train yolo11n and yolo11s on them for 40 epochs each.',
    [[S("dogs", "dog"), E("dogs"), *ts] for ts in anyorder(T("dogs", model="yolo11n", epochs=40),
                                                          T("dogs", model="yolo11s", epochs=40))])
oks("cm-12", c, "Export warehouse to YOLO, train yolo11m on it, and check the status of train-0042.",
    interleave([E("warehouse"), T("warehouse", model="yolo11m")], [J("train-0042")]))
oks("cm-13", c, 'Segment cats for "cat", export them to YOLO, and create a labeling project "cats-qa" for cats.',
    interleave([S("cats", "cat"), E("cats")], [L("cats-qa", "cats")]))
oks("cm-14", c, "Train yolo11s on parking, train yolo11s on drones, and check job train-0043.",
    anyorder(T("parking", model="yolo11s"), T("drones", model="yolo11s"), J("train-0043")))

# ---------------------------------------------------------------- different wording
c = "paraphrase"
ok("pp-01", c, 'I need masks for every dog in the dogs set, prompt "dog".', [S("dogs", "dog")])
ok("pp-02", c, "Could you get the warehouse annotations into YOLO format please?", [E("warehouse")])
ok("pp-03", c, "Fire up a YOLO training on the parking lot data.", [T("parking")])
ok("pp-04", c, "Have SAM3 outline the cars in traffic.", [S("traffic", "car")])
ok("pp-05", c, "Turn xray into a YOLO dataset.", [E("xray")])
ok("pp-06", c, "Let's train a detector on drones, yolo11s please.", [T("drones", model="yolo11s")])
ok("pp-07", c, "Please kick off a YOLO11m run on parking for 90 epochs.",
   [T("parking", model="yolo11m", epochs=90)])
ok("pp-08", c, 'Grab the cats folder, mask out "cat", and make it YOLO-ready.', [S("cats", "cat"), E("cats")])
ok("pp-09", c, "How is train-0042 doing?", [J("train-0042")])
ok("pp-10", c, 'Spin up a labeling project called "drone-labels" on drones.', [L("drone-labels", "drones")])
ok("pp-11", c, f'Using {DOCK}, segment "pallet" and produce a YOLO export.', [S(DOCK, "pallet"), E(DOCK)])
ok("pp-12", c, f"Go ahead and train on {SHELVES}/ with yolo11n.", [T(SHELVES, model="yolo11n")])
ok("pp-13", c, "Convert warehouse to YOLO, then fit yolo11x on it.",
   [E("warehouse"), T("warehouse", model="yolo11x")])
ok("pp-14", c, 'Please have SAM3 label the "crack" regions in xray again.', [S("xray", "crack")])
ok("pp-15", c, "Use a hundred and twenty epochs and train on drones.", [T("drones", epochs=120)])

# ---------------------------------------------------------------- missing optional parameters
c = "missing_optional"
ok("mo-01", c, "Train on parking.", [T("parking")])
ok("mo-02", c, "Export xray.", [E("xray")])
ok("mo-03", c, 'Segment dogs for "dog". Use whatever settings are standard.', [S("dogs", "dog")])
ok("mo-04", c, "Train a YOLO model on drones with the default settings.", [T("drones")])
ok("mo-05", c, "Export warehouse to YOLO; I do not care about the validation split.", [E("warehouse")])
ok("mo-06", c, "Train yolo11m on parking; keep everything else default.", [T("parking", model="yolo11m")])
ok("mo-07", c, 'Segment fruits for "lemon" with the default confidence.', [S("fruits", "lemon")])
ok("mo-08", c, "Train on drones for 10 epochs, nothing else changed.", [T("drones", epochs=10)])
ok("mo-09", c, 'Segment cats for "cat" and export them; defaults are fine.', [S("cats", "cat"), E("cats")])
ok("mo-10", c, "Train YOLO on parking at whatever image size it normally uses.", [T("parking")])

# ---------------------------------------------------------------- implicit prerequisites
c = "implicit_prerequisites"
ok("ip-01", c, "Train YOLO on warehouse.", [E("warehouse"), T("warehouse")])
ok("ip-02", c, "Train yolo11s on xray for 50 epochs.", [E("xray"), T("xray", model="yolo11s", epochs=50)])
ok("ip-03", c, 'Segment cats for "cat" and train a YOLO model on them.', [S("cats", "cat"), E("cats"), T("cats")])
ok("ip-04", c, 'Train on the dogs dataset after segmenting "dog".', [S("dogs", "dog"), E("dogs"), T("dogs")])
ok("ip-05", c, 'Train yolo11m on traffic, segmenting "car" first.',
   [S("traffic", "car"), E("traffic"), T("traffic", model="yolo11m")])
ok("ip-06", c, "Start training on warehouse with batch size 8.", [E("warehouse"), T("warehouse", batch_size=8)])
oks("ip-07", c, "Train on xray and on parking.", interleave([E("xray"), T("xray")], [T("parking")]))
ok("ip-08", c, 'Segment fruits for "apple", then train yolo11n for 20 epochs.',
   [S("fruits", "apple"), E("fruits"), T("fruits", model="yolo11n", epochs=20)])
oks("ip-09", c, "Train yolo11n on both warehouse and xray.",
    interleave([E("warehouse"), T("warehouse", model="yolo11n")], [E("xray"), T("xray", model="yolo11n")]))
oks("ip-10", c, 'Train on warehouse for 70 epochs and create a labeling project "wh-review" for warehouse.',
    interleave([E("warehouse"), T("warehouse", epochs=70)], [L("wh-review", "warehouse")]))
ok("ip-11", c, 'Make the dogs dataset YOLO-ready; segment "dog".', [S("dogs", "dog"), E("dogs")])
ok("ip-12", c, "Train yolo11x on xray with image size 1024.", [E("xray"), T("xray", model="yolo11x", imgsz=1024)])
ok("ip-13", c, 'Train on cats for 30 epochs, using the segmentation prompt "cat".',
   [S("cats", "cat"), E("cats"), T("cats", epochs=30)])
ok("ip-14", c, "Train on warehouse using a 0.1 validation split.",
   [E("warehouse", val_split=0.1), T("warehouse")])

# ---------------------------------------------------------------- job status
c = "job_status"
ok("js-01", c, "What is the status of job train-0041?", [J("train-0041")])
ok("js-02", c, "Check train-0042.", [J("train-0042")])
ok("js-03", c, "Did training job train-0043 finish?", [J("train-0043")])
ok("js-04", c, "Show me the metrics of train-0041.", [J("train-0041")])
ok("js-05", c, "Is train-0042 still running?", [J("train-0042")])
oks("js-06", c, "Report the state of jobs train-0041 and train-0042.", anyorder(J("train-0041"), J("train-0042")))
oks("js-07", c, "Check all three existing jobs: train-0041, train-0042 and train-0043.",
    anyorder(J("train-0041"), J("train-0042"), J("train-0043")))
ok("js-08", c, "Look up job train-0043 and tell me why it is not done.", [J("train-0043")])
ok("js-09", c, "Give me an update on the train-0042 job.", [J("train-0042")])
oks("js-10", c, 'Check train-0041 and create a labeling project "parking-errors" for parking.',
    anyorder(J("train-0041"), L("parking-errors", "parking")))

# ---------------------------------------------------------------- labeling (mock backend)
c = "labeling"
ok("lb-01", c, 'Create a labeling project called "cats-review" for the cats dataset.', [L("cats-review", "cats")])
ok("lb-02", c, 'Open a labeling project named "Warehouse QA" on warehouse.', [L("Warehouse QA", "warehouse")])
ok("lb-03", c, 'Set up a labeling project "xray-fix" for xray.', [L("xray-fix", "xray")])
ok("lb-04", c, "Make a labeling project for drones called drone-audit.", [L("drone-audit", "drones")])
oks("lb-05", c, 'Create labeling projects "p1" for parking and "d1" for drones.',
    anyorder(L("p1", "parking"), L("d1", "drones")))
oks("lb-06", c, 'Segment fruits for "apple" and set up a labeling project "fruit-fix" to correct the masks.',
    anyorder(S("fruits", "apple"), L("fruit-fix", "fruits")))
ok("lb-07", c, 'Create a labeling project named "traffic-raw" for the raw traffic images.',
   [L("traffic-raw", "traffic")])
ok("lb-08", c, f'Start a labeling project "dock-v1" on {DOCK}.', [L("dock-v1", DOCK)])
oks("lb-09", c, 'Create a labeling project called "parking-review" for parking, then train yolo11s on parking.',
    anyorder(L("parking-review", "parking"), T("parking", model="yolo11s")))
oks("lb-10", c, 'Export xray to YOLO and open a labeling project "xray-audit" for it.',
    anyorder(E("xray"), L("xray-audit", "xray")))

# ---------------------------------------------------------------- invalid parameters (must not execute)
c = "invalid_parameter"
no("iv-01", c, "Train YOLO on parking for 0 epochs.", "invalid_value", "epochs must be >= 1",
   [T("parking", epochs=0)])
no("iv-02", c, "Train on drones for -5 epochs.", "invalid_value", "negative epochs", [T("drones", epochs=-5)])
no("iv-03", c, "Train on parking with batch size 0.", "invalid_value", "batch_size must be >= 1",
   [T("parking", batch_size=0)])
no("iv-04", c, "Train on drones with image size 4096.", "invalid_value", "imgsz must be <= 2048",
   [T("drones", imgsz=4096)])
no("iv-05", c, 'Segment cats for "cat" with confidence 1.5.', "invalid_value", "confidence must be in (0, 1)",
   [S("cats", "cat", confidence=1.5)])
no("iv-06", c, 'Segment dogs for "dog" with confidence 0.', "invalid_value", "confidence must be > 0",
   [S("dogs", "dog", confidence=0)])
no("iv-07", c, "Export warehouse to YOLO with a validation split of 0.8.", "invalid_value",
   "val_split must be <= 0.5", [E("warehouse", val_split=0.8)])
no("iv-08", c, "Export xray with val_split 0.", "invalid_value", "val_split must be > 0", [E("xray", val_split=0)])
no("iv-09", c, "Train on parking for 5000 epochs.", "invalid_value", "epochs must be <= 1000",
   [T("parking", epochs=5000)])
no("iv-10", c, "Train on drones with batch size 1024.", "invalid_value", "batch_size must be <= 256",
   [T("drones", batch_size=1024)])
no("iv-11", c, 'Segment fruits for "apple" with a confidence of 120%.', "invalid_value", "confidence 1.2 > 1",
   [S("fruits", "apple", confidence=1.2)])
no("iv-12", c, "Train on parking with an image size of 16.", "invalid_value", "imgsz must be >= 32",
   [T("parking", imgsz=16)])

# ---------------------------------------------------------------- late failures (valid prefix, invalid later step)
c = "late_failure"
no("lf-01", c, 'Segment cats for "cat", export to YOLO and train yolo11n for 0 epochs.', "invalid_value",
   "final step: epochs 0", [S("cats", "cat"), E("cats"), T("cats", model="yolo11n", epochs=0)])
no("lf-02", c, 'Segment dogs for "dog", export them, and train for 2000 epochs.', "invalid_value",
   "final step: epochs > 1000", [S("dogs", "dog"), E("dogs"), T("dogs", epochs=2000)])
no("lf-03", c, 'Segment traffic for "car", export to YOLO and train yolo11n-seg.', "unsupported",
   "final step: segmentation model", [S("traffic", "car"), E("traffic"), T("traffic", model="yolo11n-seg")])
no("lf-04", c, 'Segment fruits for "apple", export, and train yolov8n.', "unsupported",
   "final step: unsupported model family", [S("fruits", "apple"), E("fruits"), T("fruits", model="yolov8n")])
no("lf-05", c, "Export warehouse to YOLO and train with batch size 0.", "invalid_value",
   "final step: batch_size 0", [E("warehouse"), T("warehouse", batch_size=0)])
no("lf-06", c, "Export xray to YOLO and train with image size 8192.", "invalid_value",
   "final step: imgsz > 2048", [E("xray"), T("xray", imgsz=8192)])
no("lf-07", c, 'Segment cats for "cat", export, and train with batch size 512.', "invalid_value",
   "final step: batch_size > 256", [S("cats", "cat"), E("cats"), T("cats", batch_size=512)])
no("lf-08", c, "Export warehouse to YOLO and then train on the forklifts dataset.", "not_found",
   "final step: dataset 'forklifts' does not exist", [E("warehouse"), T("forklifts")])
no("lf-09", c, 'Segment dogs for "dog", export them, then train on the "pets" dataset.', "not_found",
   "final step: dataset 'pets' does not exist", [S("dogs", "dog"), E("dogs"), T("pets")])
no("lf-10", c, 'Segment cats for "cat", export with val_split 0.9, and train yolo11n.', "invalid_value",
   "middle step: val_split > 0.5", [S("cats", "cat"), E("cats", val_split=0.9), T("cats", model="yolo11n")])
no("lf-11", c, 'Segment traffic for "bus", export it with a 70% validation split, and train yolo11s.',
   "invalid_value", "middle step: val_split 0.7",
   [S("traffic", "bus"), E("traffic", val_split=0.7), T("traffic", model="yolo11s")])
no("lf-12", c, "Export warehouse to YOLO, train yolo11s on it, and then deploy the model to production.",
   "unsupported", "final step: no deployment tool",
   [E("warehouse"), T("warehouse", model="yolo11s"), X("deploy_model", dataset="warehouse")])
no("lf-13", c, 'Segment fruits for "orange", export to YOLO, and evaluate mAP on a test set.', "unsupported",
   "final step: no evaluation tool", [S("fruits", "orange"), E("fruits"), X("evaluate_model", dataset="fruits")])
no("lf-14", c, "Export xray to YOLO and train the segmentation model yolo11s-seg on it.", "unsupported",
   "final step: segmentation model", [E("xray"), T("xray", model="yolo11s-seg")])
no("lf-15", c, "Train yolo11s on parking for 50 epochs and then train yolo11z on drones.", "unsupported",
   "final step: model yolo11z does not exist",
   [T("parking", model="yolo11s", epochs=50), T("drones", model="yolo11z")])
no("lf-16", c, "Check job train-0041, then train on parking for -1 epochs.", "invalid_value",
   "final step: negative epochs", [J("train-0041"), T("parking", epochs=-1)])
no("lf-17", c, 'Segment cats for "cat", export them, train yolo11n, and check the status of job train-9999.',
   "not_found", "final step: job train-9999 does not exist",
   [S("cats", "cat"), E("cats"), T("cats", model="yolo11n"), J("train-9999")])
no("lf-18", c, 'Create a labeling project "dogs-qa" for dogs, segment dogs for "dog", and export them as a '
               "segmentation (polygon) dataset.", "unsupported", "final step: export is detection-only",
   [L("dogs-qa", "dogs"), S("dogs", "dog"), E("dogs", task="segment")])
no("lf-19", c, "Export warehouse with a 0.1 split, train yolo11m for 100 epochs, then train yolo11m again with "
               "imgsz 10.", "invalid_value", "final step: imgsz < 32",
   [E("warehouse", val_split=0.1), T("warehouse", model="yolo11m", epochs=100),
    T("warehouse", model="yolo11m", imgsz=10)])
no("lf-20", c, 'Segment dogs for "dog", export to YOLO, train yolo11n, and convert the trained model to ONNX.',
   "unsupported", "final step: no model conversion tool",
   [S("dogs", "dog"), E("dogs"), T("dogs", model="yolo11n"), X("export_onnx", dataset="dogs")])

# ---------------------------------------------------------------- unsupported operations
c = "unsupported"
no("us-01", c, 'Segment cats with SAM2 using the prompt "cat".', "unsupported", "only SAM3 is available",
   [S("cats", "cat", model="sam2")])
no("us-02", c, "Train yolo11n-seg on parking.", "unsupported", "detection models only",
   [T("parking", model="yolo11n-seg")])
no("us-03", c, "Export warehouse as a YOLO segmentation dataset with polygons.", "unsupported",
   "export is detection-only", [E("warehouse", task="segment")])
no("us-04", c, "Train yolov8s on drones.", "unsupported", "only YOLO11 detection models",
   [T("drones", model="yolov8s")])
no("us-05", c, "Export xray to COCO format.", "unsupported", "only YOLO export", [E("xray", format="coco")])
no("us-06", c, "Deploy the parking model to an edge device.", "unsupported", "no deployment tool",
   [X("deploy_model", dataset="parking")])
no("us-07", c, 'Segment dogs using point prompts instead of a text prompt.', "unsupported",
   "SAM3 tool only takes a text prompt")
no("us-08", c, "Run inference with a trained model on new images from parking.", "unsupported", "no inference tool",
   [X("run_inference", dataset="parking")])
no("us-09", c, "Delete the cats dataset.", "unsupported", "no delete tool", [X("delete_dataset", dataset="cats")])
no("us-10", c, "Train a Faster R-CNN on parking.", "unsupported", "only YOLO11 detection models",
   [T("parking", model="faster_rcnn")])

# ---------------------------------------------------------------- ambiguous requests
c = "ambiguous"
no("am-01", c, "Segment the cats dataset.", "ambiguous", "no SAM3 text prompt given")
no("am-02", c, "Segment traffic.", "ambiguous", "no SAM3 text prompt given (the description must not be used)")
no("am-03", c, "Train a model.", "ambiguous", "no dataset")
no("am-04", c, "Export it to YOLO.", "ambiguous", "no dataset")
no("am-05", c, "Check the status of my job.", "ambiguous", "three jobs exist")
no("am-06", c, "Train YOLO on cats.", "ambiguous", "raw dataset needs segmentation, no text prompt given")
no("am-07", c, "Create a labeling project.", "ambiguous", "no project name or dataset")
no("am-08", c, "Run the full pipeline on dogs.", "ambiguous", "no SAM3 text prompt given")
no("am-09", c, "Annotate the fruits dataset with SAM3.", "ambiguous", "no SAM3 text prompt given")
no("am-10", c, "Do the usual with the new data.", "ambiguous", "no operation or dataset")
no("am-11", c, "Train on one of the yolo datasets.", "ambiguous", "two yolo datasets exist")
no("am-12", c, "Prepare traffic for training.", "ambiguous", "raw dataset, no SAM3 text prompt given")

# ---------------------------------------------------------------- references to things that do not exist
c = "nonexistent_reference"
no("ne-01", c, "Train YOLO on the birds dataset.", "not_found", "dataset 'birds'", [T("birds")])
no("ne-02", c, 'Segment the horses dataset for "horse".', "not_found", "dataset 'horses'", [S("horses", "horse")])
no("ne-03", c, "Export data/vehicles to YOLO.", "not_found", "dataset 'vehicles'", [E("vehicles")])
no("ne-04", c, "What is the status of job train-9999?", "not_found", "job train-9999", [J("train-9999")])
no("ne-05", c, "Check job train-0100.", "not_found", "job train-0100", [J("train-0100")])
no("ne-06", c, 'Create a labeling project "sat-review" for the satellite dataset.', "not_found",
   "dataset 'satellite'", [L("sat-review", "satellite")])
no("ne-07", c, "Train yolo11s on parking2.", "not_found", "dataset 'parking2'", [T("parking2", model="yolo11s")])
no("ne-08", c, 'Segment the medical dataset for "tumor".', "not_found", "dataset 'medical'",
   [S("medical", "tumor")])
no("ne-09", c, 'Export the "pallets" dataset to YOLO.', "not_found", "'pallet' is a class in warehouse, not a dataset",
   [E("pallets")])
no("ne-10", c, "Show job train-0044.", "not_found", "job train-0044", [J("train-0044")])


# ---------------------------------------------------------------- inexact references (explicit-request policy)
# A description of a dataset or model is not a reference; the system must ask instead of guessing.
# `literal` is the silent substitution a system might make; the validator's grounding check must reject it.
c = "inexact_reference"
no("ir-01", c, "Let's train a detector on drones, small model please.", "ambiguous", "model given only by description",
   [T("drones", model="yolo11s")])
no("ir-02", c, "Please kick off a medium YOLO11 run on parking for 90 epochs.", "ambiguous",
   "model given only by description", [T("parking", model="yolo11m", epochs=90)])
no("ir-03", c, "Go ahead and train on parking with the large model.", "ambiguous", "model given only by description",
   [T("parking", model="yolo11l")])
no("ir-04", c, "Convert warehouse to YOLO, then fit the extra-large model on it.", "ambiguous",
   "model given only by description", [E("warehouse"), T("warehouse", model="yolo11x")], late=True)

SUITE = "extended"
no("ir-05", c, "Train YOLO on the top-down car park images.", "ambiguous", "dataset given only by description",
   [T("parking")])
no("ir-06", c, "Export the weld scans to YOLO.", "ambiguous", "dataset given only by description", [E("xray")])
no("ir-07", c, "Segment cars in the street camera frames.", "ambiguous", "dataset given only by description",
   [S("traffic", "car")])
no("ir-08", c, "Train yolov8n on drones for 30 epochs.", "unsupported", "unsupported model must not be replaced",
   [T("drones", epochs=30)])
no("ir-09", c, "Train YOLO on the aerial footage.", "ambiguous", "dataset given only by description", [T("drones")])
no("ir-10", c, 'Segment the dock camera frames for "forklift".', "ambiguous", "local dataset without its path",
   [S(DOCK, "forklift")])
no("ir-11", c, "Train yolo11s on shelves.", "ambiguous", "local dataset must be referenced by its exact path",
   [T(SHELVES, model="yolo11s")])
no("ir-12", c, "Export solar_panels to YOLO.", "ambiguous", "local dataset must be referenced by its exact path",
   [E(SOLAR)])
no("ir-13", c, "Re-segment parking for the usual objects and export it again.", "ambiguous", "no SAM3 text prompt",
   [S("parking", "car"), E("parking")])
no("ir-14", c, "Check the status of the last job.", "ambiguous", "job id not written", [J("train-0043")])
no("ir-15", c, "Create a labeling project for warehouse.", "ambiguous", "no project name",
   [L("warehouse-review", "warehouse")])
no("ir-16", c, "Cancel the running job.", "ambiguous", "job id not written (destructive)", [C("train-0042")])

# ---------------------------------------------------------------- job cancellation
c = "cancel_job"
ok("cj-01", c, "Cancel job train-0042.", [C("train-0042")])
ok("cj-02", c, "Stop train-0042, then check its status.", [C("train-0042"), J("train-0042")])
oks("cj-03", c, "Cancel train-0042 and start yolo11s on drones for 50 epochs instead.",
    anyorder(C("train-0042"), T("drones", model="yolo11s", epochs=50)))
oks("cj-04", c, "Kill the running job train-0042, then export warehouse and train yolo11m on it.",
    interleave([C("train-0042")], [E("warehouse"), T("warehouse", model="yolo11m")]))
oks("cj-05", c, 'Abort train-0042 and create a labeling project "drones-relabel" for drones.',
    anyorder(C("train-0042"), L("drones-relabel", "drones")))
no("cj-06", c, "Cancel train-0041.", "invalid_value", "train-0041 is completed", [C("train-0041")])
no("cj-07", c, "Cancel job train-0099.", "not_found", "job train-0099", [C("train-0099")])
no("cj-08", c, "Cancel my training job.", "ambiguous", "job id not written")
no("cj-09", c, "Stop train-0042 and train-0043.", "invalid_value", "train-0043 already failed",
   [C("train-0042"), C("train-0043")], late=True)

# ---------------------------------------------------------------- GPU quota (3 queued/running jobs, 1 active)
c = "gpu_quota"
oks("gq-01", c, "Train yolo11n and yolo11s on parking.",
    anyorder(T("parking", model="yolo11n"), T("parking", model="yolo11s")))
oks("gq-02", c, "Cancel train-0042, then train yolo11n, yolo11s and yolo11m on parking.",
    [[C("train-0042"), *p] for p in anyorder(T("parking", model="yolo11n"), T("parking", model="yolo11s"),
                                             T("parking", model="yolo11m"))])
oks("gq-03", c, "Train yolo11m on drones and on parking, and cancel train-0042.",
    quota_filter(anyorder(T("drones", model="yolo11m"), T("parking", model="yolo11m"), C("train-0042"))))
oks("gq-04", c, 'Segment cats for "cat", export it, cancel train-0042, and train yolo11n on cats, parking and drones.',
    quota_filter(interleave([S("cats", "cat"), E("cats"), T("cats", model="yolo11n")], [C("train-0042")],
                            [T("parking", model="yolo11n")], [T("drones", model="yolo11n")])))
no("gq-05", c, "Train yolo11n, yolo11s and yolo11m on parking.", "invalid_value", "4 active jobs > quota 3",
   [T("parking", model="yolo11n"), T("parking", model="yolo11s"), T("parking", model="yolo11m")], late=True)
no("gq-06", c, "Export warehouse and xray to YOLO and train yolo11s on warehouse, xray and drones.", "invalid_value",
   "4 active jobs > quota 3", [E("warehouse"), T("warehouse", model="yolo11s"), E("xray"), T("xray", model="yolo11s"),
                               T("drones", model="yolo11s")], late=True)
no("gq-07", c, f"Train yolo11s on parking, drones and {SHELVES} for 20 epochs each.", "invalid_value",
   "4 active jobs > quota 3", [T("parking", model="yolo11s", epochs=20), T("drones", model="yolo11s", epochs=20),
                               T(SHELVES, model="yolo11s", epochs=20)], late=True)

# ---------------------------------------------------------------- long multi-dataset workflows
c = "long_workflow"
oks("lw-01", c, 'Segment cats for "cat" and dogs for "dog", export both to YOLO, and train yolo11n on both.',
    interleave([S("cats", "cat"), E("cats"), T("cats", model="yolo11n")],
               [S("dogs", "dog"), E("dogs"), T("dogs", model="yolo11n")]))
oks("lw-02", c, "Cancel train-0042, export warehouse and xray to YOLO, and train yolo11s on warehouse, xray and "
                "drones.",
    quota_filter(interleave([C("train-0042")], [E("warehouse"), T("warehouse", model="yolo11s")],
                            [E("xray"), T("xray", model="yolo11s")], [T("drones", model="yolo11s")])))
oks("lw-03", c, f'Segment {DOCK} for "pallet", export it with val_split 0.1, train yolo11s for 60 epochs, and create '
                f'a labeling project "dock-review" from {DOCK}.',
    interleave([S(DOCK, "pallet"), E(DOCK, val_split=0.1), T(DOCK, model="yolo11s", epochs=60)],
               [L("dock-review", DOCK)]))
oks("lw-04", c, f"Export {SOLAR} to YOLO, train yolo11m on it for 80 epochs, and check jobs train-0041 and "
                "train-0042.",
    interleave([E(SOLAR), T(SOLAR, model="yolo11m", epochs=80)], [J("train-0041")], [J("train-0042")]))
oks("lw-05", c, 'Segment traffic for "bus" with confidence 0.6, export with a 0.25 validation split, train yolo11l at '
                'imgsz 960 for 120 epochs, and create a labeling project "bus-qa" for traffic.',
    interleave([S("traffic", "bus", confidence=0.6), E("traffic", val_split=0.25),
                T("traffic", model="yolo11l", imgsz=960, epochs=120)], [L("bus-qa", "traffic")]))
ok("lw-06", c, f'Create a labeling project "dock-qa" from {DOCK}, then segment dock-qa for "forklift", export it and '
               "train yolo11n on dock-qa.",
   [L("dock-qa", DOCK), S("dock-qa", "forklift"), E("dock-qa"), T("dock-qa", model="yolo11n")])
oks("lw-07", c, f"Train yolo11n on {SHELVES} and yolo11s on parking, 40 epochs each, then check train-0041.",
    [[*p, J("train-0041")] for p in anyorder(T(SHELVES, model="yolo11n", epochs=40),
                                             T("parking", model="yolo11s", epochs=40))])
oks("lw-08", c, 'Segment fruits for "apple", export to YOLO, train yolo11n for 30 epochs, and create a labeling '
                'project "fruit-qa" for fruits.',
    interleave([S("fruits", "apple"), E("fruits"), T("fruits", model="yolo11n", epochs=30)], [L("fruit-qa", "fruits")]))

# ---------------------------------------------------------------- late failures in longer / destructive workflows
c = "late_failure"
no("lf-21", c, 'Segment cats for "cat" and dogs for "dog", export both, then train yolo11n on cats and yolov8n on dogs.',
   "unsupported", "last step: unsupported model",
   [S("cats", "cat"), S("dogs", "dog"), E("cats"), E("dogs"), T("cats", model="yolo11n"), T("dogs", model="yolov8n")])
no("lf-22", c, "Cancel train-0042 and retrain drones with yolov8n.", "unsupported",
   "destructive first step, unsupported model", [C("train-0042"), T("drones", model="yolov8n")])
no("lf-23", c, 'Export warehouse to YOLO, train yolo11s on it, create a labeling project "wh-qa" for warehouse, then '
               "train yolo11m on the pallets dataset.", "not_found", "last step: dataset 'pallets'",
   [E("warehouse"), T("warehouse", model="yolo11s"), L("wh-qa", "warehouse"), T("pallets", model="yolo11m")])
no("lf-24", c, "Retrain parking with yolo11m, then cancel train-0043.", "invalid_value", "train-0043 already failed",
   [T("parking", model="yolo11m"), C("train-0043")])
no("lf-25", c, 'Segment traffic for "car", export it, train yolo11s on it, and cancel train-0041.', "invalid_value",
   "train-0041 is completed", [S("traffic", "car"), E("traffic"), T("traffic", model="yolo11s"), C("train-0041")])
no("lf-26", c, f'Segment {DOCK} for "pallet", export it, and create a labeling project "parking" from it.',
   "invalid_value", "project 'parking' already exists", [S(DOCK, "pallet"), E(DOCK), L("parking", DOCK)])
no("lf-27", c, "Cancel train-0042, export xray, and train yolo11s on xray for -10 epochs.", "invalid_value",
   "destructive first step, negative epochs", [C("train-0042"), E("xray"), T("xray", model="yolo11s", epochs=-10)])
no("lf-28", c, f"Export {SOLAR} to YOLO, train yolo11m on it, and then train yolo11s on /data/local/solar_panel.",
   "not_found", "last step: path typo", [E(SOLAR), T(SOLAR, model="yolo11m"), T("/data/local/solar_panel",
                                                                              model="yolo11s")])
SUITE = "core"

# ---------------------------------------------------------------- Ukrainian paraphrases (secondary experiment)
# The SAM3 text prompt stays an explicit English phrase in quotes; everything else is Ukrainian.
_UK = {
    "st-01": 'Сегментуй датасет cats з текстовим промптом "cat".',
    "st-05": "Експортуй датасет warehouse у формат YOLO.",
    "st-07": "Натренуй модель YOLO на parking.",
    "tp-01": "Натренуй YOLO на parking протягом 80 епох.",
    "tp-03": "Натренуй yolo11m на parking: 150 епох, розмір батчу 32.",
    "tp-07": "Експортуй warehouse у YOLO з часткою валідації 0.1.",
    "tp-10": 'Сегментуй dogs за промптом "dog" з порогом впевненості 0.35.',
    "ms-01": 'Сегментуй cats за промптом "cat" і експортуй у YOLO.',
    "ms-02": "Експортуй warehouse у YOLO, а потім натренуй на ньому модель.",
    "ms-04": "Експортуй xray у YOLO і натренуй yolo11s на 40 епох.",
    "cm-01": 'Сегментуй cats за промптом "cat", експортуй у YOLO і натренуй yolo11n на 50 епох.',
    "cm-03": 'Візьми dogs, сегментуй "dog" з впевненістю 0.6, експортуй з валідацією 10% і натренуй yolo11m '
             "з батчем 32.",
    "ip-01": "Натренуй YOLO на warehouse.",
    "ip-03": 'Сегментуй cats за промптом "cat" і натренуй на них модель YOLO.',
    "js-01": "Який статус задачі train-0041?",
    "js-05": "Чи train-0042 досі виконується?",
    "lb-01": 'Створи проєкт розмітки "cats-review" для датасету cats.',
    "iv-01": "Натренуй YOLO на parking протягом 0 епох.",
    "iv-07": "Експортуй warehouse у YOLO з часткою валідації 0.8.",
    "lf-01": 'Сегментуй cats за промптом "cat", експортуй у YOLO і натренуй yolo11n на 0 епох.',
    "lf-03": 'Сегментуй traffic за промптом "car", експортуй у YOLO і натренуй yolo11n-seg.',
    "lf-08": "Експортуй warehouse у YOLO, а потім натренуй на датасеті forklifts.",
    "lf-12": "Експортуй warehouse у YOLO, натренуй yolo11s і розгорни модель у продакшн.",
    "us-01": 'Сегментуй cats за допомогою SAM2 з промптом "cat".',
    "us-02": "Натренуй yolo11n-seg на parking.",
    "am-01": "Сегментуй датасет cats.",
    "am-03": "Натренуй модель.",
    "ne-01": "Натренуй YOLO на датасеті birds.",
}


# Extended-suite cases in Ukrainian and mixed Ukrainian-English technical language.
_UK_EXT = {
    "cj-01": ("uk", "Скасуй задачу train-0042."),
    "cj-03": ("mixed", "Зупини train-0042 і запусти yolo11s на drones на 50 epochs."),
    "cj-06": ("uk", "Скасуй train-0041."),
    "gq-02": ("uk", "Скасуй train-0042, потім натренуй yolo11n, yolo11s і yolo11m на parking."),
    "gq-05": ("uk", "Натренуй yolo11n, yolo11s і yolo11m на parking."),
    "lw-03": ("mixed", f'Просегментуй {DOCK} по "pallet", експортуй з val_split 0.1, натренуй yolo11s на 60 epochs '
                       f'і створи labeling project "dock-review" з {DOCK}.'),
    "lw-08": ("mixed", 'Просегментуй fruits по промпту "apple", експортуй у YOLO, натренуй yolo11n на 30 epochs і '
                       'створи labeling project "fruit-qa" для fruits.'),
    "lf-22": ("uk", "Скасуй train-0042 і перенавчи drones на yolov8n."),
    "ir-01": ("uk", "Натренуй детектор на drones, маленьку модель, будь ласка."),
    "ir-06": ("uk", "Експортуй знімки зварних швів у YOLO."),
    "ir-11": ("mixed", "Натренуй yolo11s на shelves."),
    "ir-16": ("uk", "Скасуй задачу, яка зараз виконується."),
}


def _build_uk() -> None:
    by_id = {case["id"]: case for case in CASES}
    for src, prompt in _UK.items():
        UK_CASES.append({**by_id[src], "id": f"uk-{src}", "prompt": prompt, "source_id": src, "language": "uk"})
    for src, (lang, prompt) in _UK_EXT.items():
        UK_CASES.append({**by_id[src], "id": f"uk-{src}", "prompt": prompt, "source_id": src, "language": lang})


_build_uk()


def _check() -> None:
    ids = [case["id"] for case in CASES]
    assert len(ids) == len(set(ids)), "duplicate case ids"
    for case in CASES:
        if case["expect"] == "execute":
            keys = [json.dumps(p, sort_keys=True) for p in case["plans"]]
            assert len(keys) == len(set(keys)), f"{case['id']}: duplicate acceptable plans"
            assert all(within_quota(p) for p in case["plans"]), f"{case['id']}: plan exceeds the GPU quota"


def _write(path: Path, cases: list[dict]) -> None:
    with path.open("w") as f:
        for case in cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    _check()
    here = Path(__file__).parent
    _write(here / "cases.jsonl", CASES)
    _write(here / "cases_uk.jsonl", UK_CASES)
    from collections import Counter
    print(f"wrote {len(CASES)} cases ({Counter(c['expect'] for c in CASES)}) and {len(UK_CASES)} Ukrainian cases")
    for cat, n in Counter((c["suite"], c["category"]) for c in CASES).items():
        print(f"  {cat[0]:<9} {cat[1]:<24} {n}")
