"""Source of truth for the benchmark. Regenerate the JSONL with:

    uv run python benchmark/build_cases.py

Each case is either
  expect="execute": `plans` lists one or more acceptable expected plans (step order matters
                    within a plan; alternatives cover equally valid orderings), or
  expect="abstain": the system must not execute anything (unsupported / invalid / ambiguous / unknown refs).

Expected `args` list only what the request determines. Any other argument the system passes must be
equal to the tool default, otherwise it counts as a hallucinated parameter. The value "*" accepts anything.

Workspace (see cvorch.workspace.default_workspace):
  raw: cats, dogs, traffic, fruits | annotated: warehouse, xray | yolo: parking, drones
  jobs: train-0041 completed, train-0042 running, train-0043 failed
"""
from __future__ import annotations

import json
from pathlib import Path


def _step(tool, **args):
    return {"tool": tool, "args": args}


def S(ds, **kw):
    return _step("segment_dataset", dataset=ds, **kw)


def E(ds, **kw):
    return _step("export_yolo", dataset=ds, **kw)


def T(ds, **kw):
    return _step("train_yolo", dataset=ds, **kw)


def J(job):
    return _step("get_job_status", job_id=job)


def L(name, ds):
    return _step("create_labeling_project", project_name=name, dataset=ds)


CASES: list[dict] = []


def ok(cid, cat, prompt, *plans):
    CASES.append({"id": cid, "category": cat, "prompt": prompt, "expect": "execute", "plans": [list(p) for p in plans]})


def no(cid, cat, prompt, reason):
    CASES.append({"id": cid, "category": cat, "prompt": prompt, "expect": "abstain", "reason": reason})


# ---------------------------------------------------------------- single tool
c = "single_tool"
ok("st-01", c, "Segment the cats dataset.", [S("cats")])
ok("st-02", c, "Run segmentation on the dogs images.", [S("dogs")])
ok("st-03", c, "Please segment traffic.", [S("traffic")])
ok("st-04", c, "Annotate the fruits dataset automatically.", [S("fruits")])
ok("st-05", c, "Export the warehouse dataset to YOLO format.", [E("warehouse")])
ok("st-06", c, "Convert the xray annotations into a YOLO dataset.", [E("xray")])
ok("st-07", c, "Export warehouse for YOLO.", [E("warehouse")])
ok("st-08", c, "Train a YOLO model on parking.", [T("parking")])
ok("st-09", c, "Start a training run on the drones dataset.", [T("drones")])
ok("st-10", c, "Fit YOLO to the parking data.", [T("parking")])
ok("st-11", c, "Segment all images in data/dogs.", [S("dogs")])
ok("st-12", c, "Kick off YOLO training using drones.", [T("drones")])

# ---------------------------------------------------------------- tool + parameters
c = "tool_with_params"
ok("tp-01", c, "Train YOLO on parking for 80 epochs.", [T("parking", epochs=80)])
ok("tp-02", c, "Train yolo11s on drones.", [T("drones", model="yolo11s")])
ok("tp-03", c, "Train yolo11m on parking for 150 epochs with batch size 32.",
   [T("parking", model="yolo11m", epochs=150, batch_size=32)])
ok("tp-04", c, "Train on drones with image size 1280 for 60 epochs.", [T("drones", imgsz=1280, epochs=60)])
ok("tp-05", c, "Segment cats using yolo11n-seg.", [S("cats", model="yolo11n-seg")])
ok("tp-06", c, "Segment the traffic dataset, only cars and buses.", [S("traffic", classes=["car", "bus"])])
ok("tp-07", c, "Segment dogs with a confidence threshold of 0.7.", [S("dogs", confidence=0.7)])
ok("tp-08", c, "Export warehouse to YOLO with a 10% validation split.", [E("warehouse", val_split=0.1)])
ok("tp-09", c, "Export xray as a detection dataset (bounding boxes) for YOLO.", [E("xray", task="detect")])
ok("tp-10", c, "Train the large YOLO11 model on parking for 200 epochs.", [T("parking", model="yolo11l", epochs=200)])
ok("tp-11", c, "Train yolo11x on drones, 25 epochs, batch 8.", [T("drones", model="yolo11x", epochs=25, batch_size=8)])
ok("tp-12", c, "Segment fruits looking for apples, bananas and oranges using SAM2.",
   [S("fruits", classes=["apple", "banana", "orange"], model="sam2")])
ok("tp-13", c, "Run a 300-epoch training of yolo11n-seg on parking.", [T("parking", model="yolo11n-seg", epochs=300)])
ok("tp-14", c, "Export warehouse to YOLO, use 30 percent of the images for validation.", [E("warehouse", val_split=0.3)])
ok("tp-15", c, "Train YOLO on drones for a hundred and twenty epochs.", [T("drones", epochs=120)])
ok("tp-16", c, "Segment the traffic frames, pedestrians only, confidence 0.35.",
   [S("traffic", classes=["pedestrian"], confidence=0.35)])

# ---------------------------------------------------------------- multi-step
c = "multi_step"
ok("ms-01", c, "Segment cats and export them to YOLO.", [S("cats"), E("cats")])
ok("ms-02", c, "Export warehouse to YOLO and then train on it.", [E("warehouse"), T("warehouse")])
ok("ms-03", c, "Segment dogs and train YOLO on them.", [S("dogs"), E("dogs"), T("dogs")])
ok("ms-04", c, "Annotate fruits, then convert to YOLO format.", [S("fruits"), E("fruits")])
ok("ms-05", c, "Export xray to YOLO and train yolo11s for 40 epochs.", [E("xray"), T("xray", model="yolo11s", epochs=40)])
ok("ms-06", c, "Segment traffic and create a labeling project 'Traffic QA' so annotators can fix the masks.",
   [S("traffic"), L("Traffic QA", "traffic")])
ok("ms-07", c, "Train on parking and create a labeling project called parking-audit for it.",
   [T("parking"), L("parking-audit", "parking")], [L("parking-audit", "parking"), T("parking")])
ok("ms-08", c, "Segment cats and dogs.", [S("cats"), S("dogs")], [S("dogs"), S("cats")])
ok("ms-09", c, "Export both warehouse and xray to YOLO.", [E("warehouse"), E("xray")], [E("xray"), E("warehouse")])
ok("ms-10", c, "Segment the dogs dataset with sam2, then export it with a 0.15 validation split.",
   [S("dogs", model="sam2"), E("dogs", val_split=0.15)])
ok("ms-11", c, "Check job train-0041 and start a new training on parking for 60 epochs.",
   [J("train-0041"), T("parking", epochs=60)], [T("parking", epochs=60), J("train-0041")])
ok("ms-12", c, "Train yolo11n and yolo11s on drones, 50 epochs each.",
   [T("drones", model="yolo11n", epochs=50), T("drones", model="yolo11s", epochs=50)],
   [T("drones", model="yolo11s", epochs=50), T("drones", model="yolo11n", epochs=50)])

# ---------------------------------------------------------------- complex multi-step
c = "complex_multi_step"
ok("cm-01", c, "Segment cats, export them to YOLO and train YOLO11n for 50 epochs.",
   [S("cats"), E("cats"), T("cats", model="yolo11n", epochs=50)])
ok("cm-02", c, "Segment the traffic dataset (cars, buses, pedestrians), export to YOLO and train yolo11s for 100 epochs.",
   [S("traffic", classes=["car", "bus", "pedestrian"]), E("traffic"), T("traffic", model="yolo11s", epochs=100)])
ok("cm-03", c, "Take the fruits images, segment them with yolo11s-seg, export with 20% validation, and train yolo11m with batch size 8.",
   [S("fruits", model="yolo11s-seg"), E("fruits", val_split=0.2), T("fruits", model="yolo11m", batch_size=8)])
ok("cm-04", c, "Segment dogs with confidence 0.6, export as detection labels, then train yolo11n for 75 epochs at image size 512.",
   [S("dogs", confidence=0.6), E("dogs", task="detect"), T("dogs", model="yolo11n", epochs=75, imgsz=512)])
ok("cm-05", c, "Segment cats, export to YOLO, train for 30 epochs, and create a labeling project 'cats-review' for manual checking.",
   [S("cats"), E("cats"), T("cats", epochs=30), L("cats-review", "cats")],
   [S("cats"), L("cats-review", "cats"), E("cats"), T("cats", epochs=30)],
   [S("cats"), E("cats"), L("cats-review", "cats"), T("cats", epochs=30)])
ok("cm-06", c, "Full pipeline on traffic: annotate, convert to YOLO and train a yolo11n-seg model for 120 epochs.",
   [S("traffic"), E("traffic"), T("traffic", model="yolo11n-seg", epochs=120)])
ok("cm-07", c, "Prepare the dogs dataset for YOLO and train the medium model for 90 epochs with batch 24.",
   [S("dogs"), E("dogs"), T("dogs", model="yolo11m", epochs=90, batch_size=24)])
ok("cm-08", c, "Segment fruits, export it, and train two models on it: yolo11n and yolo11s, 50 epochs each.",
   [S("fruits"), E("fruits"), T("fruits", model="yolo11n", epochs=50), T("fruits", model="yolo11s", epochs=50)],
   [S("fruits"), E("fruits"), T("fruits", model="yolo11s", epochs=50), T("fruits", model="yolo11n", epochs=50)])
ok("cm-09", c, "Export xray to YOLO with polygon masks, train yolo11s-seg for 200 epochs at image size 1024, and open a labeling project 'xray-cracks'.",
   [E("xray", task="segment"), T("xray", model="yolo11s-seg", epochs=200, imgsz=1024), L("xray-cracks", "xray")],
   [L("xray-cracks", "xray"), E("xray", task="segment"), T("xray", model="yolo11s-seg", epochs=200, imgsz=1024)],
   [E("xray", task="segment"), L("xray-cracks", "xray"), T("xray", model="yolo11s-seg", epochs=200, imgsz=1024)])
ok("cm-10", c, "Run everything for cats: segmentation with SAM2, YOLO export, training yolo11l for 10 epochs.",
   [S("cats", model="sam2"), E("cats"), T("cats", model="yolo11l", epochs=10)])

# ---------------------------------------------------------------- different wording
c = "paraphrase"
ok("pp-01", c, "Prepare the cats images for YOLO training.", [S("cats"), E("cats")])
ok("pp-02", c, "I need a YOLO-ready version of the dogs dataset.", [S("dogs"), E("dogs")])
ok("pp-03", c, "Get the warehouse data ready for training a detector.", [E("warehouse")],
   [E("warehouse", task="detect")])
ok("pp-04", c, "Teach a YOLO model to recognise stuff in the parking images.", [T("parking")])
ok("pp-05", c, "Can you make masks for every photo in fruits?", [S("fruits")])
ok("pp-06", c, "Turn traffic into a trained YOLO model.", [S("traffic"), E("traffic"), T("traffic")])
ok("pp-07", c, "I'd like the xray scans in YOLO format so I can train later.", [E("xray")])
ok("pp-08", c, "Produce pixel-level annotations for the dogs pictures.", [S("dogs")])
ok("pp-09", c, "Fine-tune YOLO on drones, let it run 40 passes over the data.", [T("drones", epochs=40)])
ok("pp-10", c, "hey can u train yolo on parking pls, like 20 epochs", [T("parking", epochs=20)])
ok("pp-11", c, "From raw cats photos to a trained yolo11n: do the whole thing.",
   [S("cats"), E("cats"), T("cats", model="yolo11n")])
ok("pp-12", c, "Give the annotators a workspace to review warehouse; name it Pallet check.",
   [L("Pallet check", "warehouse")])
ok("pp-13", c, "How is train-0043 doing?", [J("train-0043")])
ok("pp-14", c, "Build me a YOLO dataset out of the fruits pictures and then fit the small model to it.",
   [S("fruits"), E("fruits"), T("fruits", model="yolo11s")])

# ---------------------------------------------------------------- missing optional parameters
c = "missing_optional"
ok("mo-01", c, "Train YOLO on this dataset: parking.", [T("parking")])
ok("mo-02", c, "Train a model on drones.", [T("drones")])
ok("mo-03", c, "Segment cats.", [S("cats")])
ok("mo-04", c, "Export warehouse.", [E("warehouse")])
ok("mo-05", c, "Segment traffic and export it.", [S("traffic"), E("traffic")])
ok("mo-06", c, "Train on parking with the default settings.", [T("parking")])
ok("mo-07", c, "Export xray, then train.", [E("xray"), T("xray")])
ok("mo-08", c, "Segment fruits, export, train.", [S("fruits"), E("fruits"), T("fruits")])

# ---------------------------------------------------------------- implicit prerequisites
c = "implicit_prerequisites"
ok("ip-01", c, "Train YOLO on cats.", [S("cats"), E("cats"), T("cats")])
ok("ip-02", c, "Train yolo11s on dogs for 30 epochs.", [S("dogs"), E("dogs"), T("dogs", model="yolo11s", epochs=30)])
ok("ip-03", c, "Train a YOLO model on warehouse.", [E("warehouse"), T("warehouse")])
ok("ip-04", c, "Train on xray for 70 epochs.", [E("xray"), T("xray", epochs=70)])
ok("ip-05", c, "Export cats to YOLO.", [S("cats"), E("cats")])
ok("ip-06", c, "Convert the fruits dataset to YOLO format.", [S("fruits"), E("fruits")])
ok("ip-07", c, "Train yolo11n-seg on traffic with image size 960.",
   [S("traffic"), E("traffic"), T("traffic", model="yolo11n-seg", imgsz=960)])
ok("ip-08", c, "I want a YOLO model trained on the dogs photos, 15 epochs.", [S("dogs"), E("dogs"), T("dogs", epochs=15)])

# ---------------------------------------------------------------- job status
c = "job_status"
ok("js-01", c, "What's the status of job train-0042?", [J("train-0042")])
ok("js-02", c, "Is training train-0041 finished? Show me its metrics.", [J("train-0041")])
ok("js-03", c, "Check on train-0043.", [J("train-0043")])
ok("js-04", c, "Did the job train-0041 complete?", [J("train-0041")])
ok("js-05", c, "Show the progress of the drones training job (train-0042).", [J("train-0042")])
ok("js-06", c, "Status of train-0041 and train-0042, please.",
   [J("train-0041"), J("train-0042")], [J("train-0042"), J("train-0041")])

# ---------------------------------------------------------------- labeling (mock backend)
c = "labeling"
ok("lb-01", c, "Create a labeling project for warehouse called 'Warehouse review'.", [L("Warehouse review", "warehouse")])
ok("lb-02", c, "Set up a labeling project named xray-qa for the xray dataset.", [L("xray-qa", "xray")])
ok("lb-03", c, "Open a project in the labeling tool for parking; call it Parking v2.", [L("Parking v2", "parking")])
ok("lb-04", c, "Make a labeling project 'dogs-manual' with the dogs dataset.", [L("dogs-manual", "dogs")])
ok("lb-05", c, "Segment fruits and then create a labeling project 'fruit-fix' so we can correct the masks.",
   [S("fruits"), L("fruit-fix", "fruits")])
ok("lb-06", c, "Create labeling projects for warehouse and xray, named 'wh' and 'xr' respectively.",
   [L("wh", "warehouse"), L("xr", "xray")], [L("xr", "xray"), L("wh", "warehouse")])

# ---------------------------------------------------------------- invalid parameters (must not execute)
c = "invalid_parameter"
no("iv-01", c, "Train YOLO on parking for -10 epochs.", "epochs must be >= 1")
no("iv-02", c, "Train on drones for 0 epochs.", "epochs must be >= 1")
no("iv-03", c, "Train yolo11n on parking with batch size -4.", "batch_size must be >= 1")
no("iv-04", c, "Segment cats with confidence 1.5.", "confidence must be in (0, 1)")
no("iv-05", c, "Export warehouse with a validation split of 0.9.", "val_split must be <= 0.5")
no("iv-06", c, "Train YOLO on parking for 5000 epochs.", "epochs must be <= 1000")
no("iv-07", c, "Segment dogs, export them and train for -30 epochs.", "epochs must be >= 1 (multi-step)")
no("iv-08", c, "Export xray to YOLO and train with image size 10.", "imgsz must be >= 32 (multi-step)")
no("iv-09", c, "Segment traffic with confidence -0.2 and export it.", "confidence must be in (0, 1) (multi-step)")
no("iv-10", c, "Train on drones for 2.5 epochs.", "epochs must be an integer")
no("iv-11", c, "Export warehouse with 0% validation.", "val_split must be > 0")
no("iv-12", c, "Train yolo11s on parking with batch size 0 for 20 epochs.", "batch_size must be >= 1")

# ---------------------------------------------------------------- unsupported operations
c = "unsupported"
no("us-01", c, "Train Faster R-CNN on parking.", "unsupported model family")
no("us-02", c, "Train a YOLOv5 model on drones.", "unsupported model version")
no("us-03", c, "Export warehouse to COCO format.", "unsupported export format")
no("us-04", c, "Deploy the model from train-0041 to production.", "no deployment tool")
no("us-05", c, "Delete the cats dataset.", "no delete tool")
no("us-06", c, "Augment the dogs images with random rotations.", "no augmentation tool")
no("us-07", c, "Download the COCO dataset and train on it.", "no download tool")
no("us-08", c, "Run inference with the weights from train-0041 on the traffic images.", "no inference tool")
no("us-09", c, "Track objects across the video frames in drones.", "no tracking tool")
no("us-10", c, "Train a DETR model on parking for 50 epochs.", "unsupported model family")
no("us-11", c, "Export the trained parking model to ONNX.", "no model export tool")
no("us-12", c, "Segment the traffic dataset with Mask R-CNN.", "unsupported segmentation model")

# ---------------------------------------------------------------- ambiguous requests
c = "ambiguous"
no("am-01", c, "Process this dataset.", "no dataset, no operation")
no("am-02", c, "Train it.", "no dataset")
no("am-03", c, "Segment the animal dataset.", "cats or dogs")
no("am-04", c, "Do the usual.", "no operation")
no("am-05", c, "Train YOLO.", "no dataset")
no("am-06", c, "Export it to YOLO.", "no dataset")
no("am-07", c, "Check my job.", "which job")
no("am-08", c, "Make it better.", "no operation")
no("am-09", c, "Segment the images.", "no dataset")
no("am-10", c, "Train on the annotated dataset.", "warehouse or xray")

# ---------------------------------------------------------------- references to things that do not exist
c = "nonexistent_reference"
no("ne-01", c, "Segment the birds dataset.", "dataset does not exist")
no("ne-02", c, "Train YOLO on data/cars for 50 epochs.", "dataset does not exist")
no("ne-03", c, "What's the status of job train-9999?", "job does not exist")
no("ne-04", c, "Export the satellite dataset to YOLO.", "dataset does not exist")
no("ne-05", c, "Create a labeling project 'x' for dataset horses.", "dataset does not exist")
no("ne-06", c, "Train yolo11n on the parkings2 dataset.", "dataset does not exist")


if __name__ == "__main__":
    ids = [c["id"] for c in CASES]
    assert len(ids) == len(set(ids)), "duplicate case ids"
    out = Path(__file__).with_name("cases.jsonl")
    out.write_text("".join(json.dumps(c) + "\n" for c in CASES))
    print(f"wrote {len(CASES)} cases to {out}")
