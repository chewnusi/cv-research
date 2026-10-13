"""Case-level bootstrap confidence intervals.

The resampling unit is the benchmark case (repeats of the same case are averaged first), because repeats
of one prompt are not independent observations. Differences between two runs are paired by case id.
"""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Callable

Score = dict
Filter = Callable[[Score], bool]
Value = Callable[[Score], bool | None]


def per_case_values(records: list[dict], include: Filter, value: Value) -> dict[str, float]:
    acc: dict[str, list[float]] = defaultdict(list)
    for r in records:
        s = r["score"]
        if include(s):
            v = value(s)
            if v is not None:
                acc[s["case_id"]].append(float(v))
    return {cid: sum(vs) / len(vs) for cid, vs in acc.items()}


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def bootstrap_ci(values: dict[str, float], b: int = 2000, seed: int = 0, alpha: float = 0.05
                 ) -> tuple[float, float] | None:
    xs = list(values.values())
    if len(xs) < 2:
        return None
    rng = random.Random(seed)
    stats = sorted(_mean([xs[rng.randrange(len(xs))] for _ in xs]) for _ in range(b))
    return stats[int(alpha / 2 * b)], stats[min(b - 1, int((1 - alpha / 2) * b))]


def paired_diff_ci(a: dict[str, float], b: dict[str, float], n_boot: int = 2000, seed: int = 0,
                   alpha: float = 0.05) -> tuple[float, tuple[float, float] | None] | None:
    """Difference mean(b) - mean(a) over the cases present in both, with a paired bootstrap CI."""
    ids = sorted(set(a) & set(b))
    if not ids:
        return None
    diffs = [b[i] - a[i] for i in ids]
    point = _mean(diffs)
    if len(ids) < 2:
        return point, None
    rng = random.Random(seed)
    stats = sorted(_mean([diffs[rng.randrange(len(diffs))] for _ in diffs]) for _ in range(n_boot))
    return point, (stats[int(alpha / 2 * n_boot)], stats[min(n_boot - 1, int((1 - alpha / 2) * n_boot))])
