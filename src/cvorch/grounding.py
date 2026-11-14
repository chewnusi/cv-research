"""Deterministic grounding of plan values in the user's request (the explicit-request policy).

The policy: never fill in what the user did not say beyond tool defaults. A plan may only use dataset
references, model names, SAM3 prompts / class names, job ids, project names and non-default numbers that
are written in the request. The validator reports two kinds of violations:

  ungrounded_value         a step uses a value the request does not state (e.g. `yolov8n` silently
                           replaced by `yolo11n`, "the weld scans" resolved to `xray`, an invented prompt)
  request_value_not_used   the request names a model, export format or job id that no step uses
                           (e.g. the unsupported `yolov8n` was dropped and the default model is trained)

All matching is plain text processing (no LLM). It is deliberately conservative: case-insensitive whole-token
matching, singular/plural tolerance for SAM3 prompts, and digits/percentages/simple English number words
for numbers.
"""
from __future__ import annotations

import re
from typing import Any, Iterable

_FILLER = {"a", "an", "the", "all", "every", "each", "any", "of"}


def mentions(text: str, value: str) -> bool:
    """`value` occurs in `text` as a whole token (case-insensitive; a trailing slash is ignored)."""
    v = value.strip()
    if len(v) > 1:
        v = v.rstrip("/")
    if not v:
        return False
    return re.search(r"(?<![\w-])" + re.escape(v) + r"(?![\w-])", text, re.IGNORECASE) is not None


def _word_forms(w: str) -> set[str]:
    w = w.strip().lower()
    forms = {w}
    if w.endswith("ies") and len(w) > 4:
        forms.add(w[:-3] + "y")
    if w.endswith("es"):
        forms.add(w[:-2])
    if w.endswith("s") and not w.endswith("ss"):
        forms.add(w[:-1])
    return forms


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def phrase_grounded(text: str, phrase: str) -> bool:
    """Every content word of `phrase` appears in `text` (singular/plural tolerant)."""
    words = [w for w in _words(phrase) if w not in _FILLER]
    if not words:
        return False
    available: set[str] = set()
    for t in _words(text):
        available |= _word_forms(t)
    return all(_word_forms(w) & available for w in words)


_QUOTES = "\"'`\u201c\u201d\u2018\u2019\u00ab\u00bb"


def without_references(text: str, references: Iterable[str]) -> str:
    """`text` with unquoted occurrences of the given dataset references removed.

    A SAM3 prompt must not be derived from a dataset name ("Segment the cats dataset" does not say what to
    segment), so the words of the dataset reference do not count as a stated prompt. A quoted occurrence
    ('Segment cats for "cats"') is an explicit prompt and is kept.
    """
    for ref in references:
        v = ref.strip()
        if len(v) > 1:
            v = v.rstrip("/")
        if v:
            q = re.escape(_QUOTES)
            text = re.sub(rf"(?<![\w\-{q}]){re.escape(v)}/?(?![\w\-{q}])", " ", text, flags=re.IGNORECASE)
    return text


# ------------------------------------------------------------------ numbers

_NUM = re.compile(r"(?<![\w.\-])(-?\d+(?:[.,]\d+)?)\s*(%|percent\b|per cent\b|відсот\w*)?", re.IGNORECASE)
_UNITS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
    "seventeen eighteen nineteen".split())}
_TENS = {w: 10 * i for i, w in enumerate("_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()) if i >= 2}


def _word_numbers(text: str) -> set[float]:
    out: set[float] = set()
    total = current = 0
    active = False
    for tok in re.findall(r"[a-z]+", text.lower()):
        if tok in _UNITS or tok in _TENS:
            current += _UNITS.get(tok, 0) + _TENS.get(tok, 0)
            active = True
        elif tok == "hundred":
            current = max(current, 1) * 100
            active = True
        elif tok == "thousand":
            total += max(current, 1) * 1000
            current = 0
            active = True
        elif tok in ("and", "a") and (active or tok == "a"):
            continue
        else:
            if active:
                out.add(float(total + current))
            total = current = 0
            active = False
    if active:
        out.add(float(total + current))
    return out


def request_numbers(text: str) -> set[float]:
    nums: set[float] = set()
    for m in _NUM.finditer(text):
        v = float(m.group(1).replace(",", "."))
        nums.add(v)
        if m.group(2):
            nums.add(v / 100)
    return nums | _word_numbers(text)


def number_grounded(text: str, value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return True  # non-numbers are type errors, reported elsewhere
    return any(abs(float(value) - n) < 1e-9 for n in request_numbers(text))


# ------------------------------------------------------------------ explicit values named in the request

_MODEL_TOKEN = re.compile(
    r"(?<![\w-])(yolo\s?v?\d+[a-z0-9]*(?:-[a-z]+)?|rt-?detr\w*|detr\w*|faster[\s-]?r-?cnn|mask[\s-]?r-?cnn|"
    r"efficientdet\w*|sam\s?-?2(?:\.\d)?)(?![\w-])", re.IGNORECASE)
_FORMAT_TOKEN = re.compile(r"(?<![\w-])(coco|pascal[\s-]?voc|voc|polygons?|onnx|tensorrt|coreml|tflite)(?![\w-])",
                           re.IGNORECASE)
_JOB_TOKEN = re.compile(r"(?<![\w-])(train-\d+)(?![\w-])", re.IGNORECASE)


def requested_models(text: str) -> list[str]:
    return [re.sub(r"\s+", "", m.group(1).lower()) for m in _MODEL_TOKEN.finditer(text)]


def requested_formats(text: str) -> list[str]:
    return [m.group(1).lower() for m in _FORMAT_TOKEN.finditer(text)]


def requested_job_ids(text: str) -> list[str]:
    return [m.group(1).lower() for m in _JOB_TOKEN.finditer(text)]


def unused_request_values(text: str, used_models: Iterable[str], used_job_ids: Iterable[str],
                          supported_models: Iterable[str]) -> list[str]:
    """Messages for explicit models / formats / job ids in the request that the plan does not use."""
    used_models = {m.lower() for m in used_models}
    used_jobs = {j.lower() for j in used_job_ids}
    supported = [m.lower() for m in supported_models]
    out: list[str] = []
    for m in dict.fromkeys(requested_models(text)):
        if m in used_models:
            continue
        if m not in supported and any(s.startswith(m) for s in supported):
            continue  # a family name such as "YOLO11", not a specific model
        out.append(f"the request names the model '{m}', but no step uses it (a model must never be replaced)")
    for f in dict.fromkeys(requested_formats(text)):
        out.append(f"the request asks for '{f}', which no tool provides (export is YOLO detection only)")
    for j in dict.fromkeys(requested_job_ids(text)):
        if j not in used_jobs:
            out.append(f"the request names job '{j}', but no step uses it")
    return out
