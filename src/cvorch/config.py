from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Literal

from dotenv import load_dotenv

Mode = Literal["baseline", "planned"]


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return default if raw in (None, "") else int(raw)


def _float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return default if raw in (None, "") else float(raw)


@dataclass(frozen=True)
class Settings:
    mode: Mode = "planned"

    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.0
    llm_seed: int | None = 42
    llm_timeout_s: float = 60.0

    baseline_max_rounds: int = 8

    planner_validate: bool = True
    planner_repair_attempts: int = 0
    planner_response_format: Literal["json_object", "json_schema"] = "json_object"
    planner_auto_prerequisites: bool = False

    tool_arg_validation: bool = True

    bench_concurrency: int = 1

    @classmethod
    def from_env(cls, load_dotenv_file: bool = True) -> Settings:
        if load_dotenv_file:
            load_dotenv()
        mode = os.getenv("ORCH_MODE", "planned").strip().lower()
        if mode not in ("baseline", "planned"):
            raise ValueError(f"ORCH_MODE must be 'baseline' or 'planned', got {mode!r}")
        fmt = os.getenv("PLANNER_RESPONSE_FORMAT", "json_object").strip().lower()
        if fmt not in ("json_object", "json_schema"):
            raise ValueError(f"PLANNER_RESPONSE_FORMAT must be json_object|json_schema, got {fmt!r}")
        seed_raw = os.getenv("LLM_SEED", "42")
        return cls(
            mode=mode,  # type: ignore[arg-type]
            llm_base_url=os.getenv("LLM_BASE_URL") or None,
            llm_api_key=os.getenv("LLM_API_KEY") or None,
            llm_model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            llm_temperature=_float("LLM_TEMPERATURE", 0.0),
            llm_seed=None if seed_raw.strip() in ("", "none") else int(seed_raw),
            llm_timeout_s=_float("LLM_TIMEOUT_S", 60.0),
            baseline_max_rounds=_int("BASELINE_MAX_ROUNDS", 8),
            planner_validate=_bool("PLANNER_VALIDATE", True),
            planner_repair_attempts=_int("PLANNER_REPAIR_ATTEMPTS", 0),
            planner_response_format=fmt,  # type: ignore[arg-type]
            planner_auto_prerequisites=_bool("PLANNER_AUTO_PREREQUISITES", False),
            tool_arg_validation=_bool("TOOL_ARG_VALIDATION", True),
            bench_concurrency=_int("BENCH_CONCURRENCY", 1),
        )

    def public_dict(self) -> dict:
        d = asdict(self)
        d.pop("llm_api_key", None)
        return d
