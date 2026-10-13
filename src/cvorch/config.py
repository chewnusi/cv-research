"""Configuration.

Orchestration variants are selected only through named experiment profiles (ORCH_PROFILE), so the
primary comparison cannot be contaminated by stray environment flags. Model and runtime settings
(shared by every profile) come from environment variables.
"""
from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass, replace
from typing import Literal

from dotenv import load_dotenv

Mode = Literal["baseline", "planned"]
StructuredOutput = Literal["json_schema", "strict_json_schema", "json_mode"]


@dataclass(frozen=True)
class Profile:
    name: str
    mode: Mode
    description: str
    planner_validate: bool = True
    planner_repair_attempts: int = 0
    planner_auto_prerequisites: bool = False
    planner_structured_output: StructuredOutput = "json_schema"
    primary: bool = False


PROFILES: dict[str, Profile] = {p.name: p for p in [
    Profile("baseline", "baseline", "LLM -> direct tool calling (primary baseline)", primary=True),
    Profile("planned", "planned", "LLM -> structured plan -> deterministic validation -> execution (primary proposed)",
            primary=True),
    Profile("planned_no_validator", "planned", "ablation: structured plan executed without validation",
            planner_validate=False),
    Profile("planned_strict_schema", "planned", "ablation: provider-side strict JSON schema for the plan",
            planner_structured_output="strict_json_schema"),
    Profile("planned_auto_prereq", "planned", "ablation: deterministic insertion of missing export_yolo steps",
            planner_auto_prerequisites=True),
    Profile("planned_repair", "planned", "ablation: one validator-feedback repair attempt",
            planner_repair_attempts=1),
]}

_LEGACY_FLAGS = ("PLANNER_VALIDATE", "PLANNER_REPAIR_ATTEMPTS", "PLANNER_AUTO_PREREQUISITES", "PLANNER_RESPONSE_FORMAT")


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
    profile: str = "planned"
    mode: Mode = "planned"
    planner_validate: bool = True
    planner_repair_attempts: int = 0
    planner_auto_prerequisites: bool = False
    planner_structured_output: StructuredOutput = "json_schema"

    llm_model: str = "gpt-4o-mini"
    llm_api_key: str | None = None
    llm_base_url: str | None = None
    llm_temperature: float = 0.0
    llm_seed: int | None = 42
    llm_timeout_s: float = 60.0
    llm_max_retries: int = 3

    baseline_max_rounds: int = 8
    tool_arg_validation: bool = True
    bench_concurrency: int = 1

    def with_profile(self, name: str) -> Settings:
        if name not in PROFILES:
            raise ValueError(f"unknown profile {name!r}; available: {', '.join(PROFILES)}")
        p = PROFILES[name]
        return replace(self, profile=p.name, mode=p.mode, planner_validate=p.planner_validate,
                       planner_repair_attempts=p.planner_repair_attempts,
                       planner_auto_prerequisites=p.planner_auto_prerequisites,
                       planner_structured_output=p.planner_structured_output)

    @classmethod
    def from_env(cls, profile: str | None = None, load_dotenv_file: bool = True) -> Settings:
        if load_dotenv_file:
            load_dotenv()
        for flag in _LEGACY_FLAGS:
            if os.getenv(flag):
                print(f"warning: {flag} is ignored; orchestration variants are selected with ORCH_PROFILE "
                      f"({', '.join(PROFILES)})", file=sys.stderr)
        name = profile or os.getenv("ORCH_PROFILE") or os.getenv("ORCH_MODE") or "planned"
        seed_raw = os.getenv("LLM_SEED", "42").strip()
        base = cls(
            llm_model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            llm_api_key=os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or None,
            llm_base_url=os.getenv("LLM_BASE_URL") or None,
            llm_temperature=_float("LLM_TEMPERATURE", 0.0),
            llm_seed=None if seed_raw.lower() in ("", "none") else int(seed_raw),
            llm_timeout_s=_float("LLM_TIMEOUT_S", 60.0),
            llm_max_retries=_int("LLM_MAX_RETRIES", 3),
            baseline_max_rounds=_int("BASELINE_MAX_ROUNDS", 8),
            tool_arg_validation=_bool("TOOL_ARG_VALIDATION", True),
            bench_concurrency=_int("BENCH_CONCURRENCY", 1),
        )
        return base.with_profile(name.strip().lower())

    def public_dict(self) -> dict:
        d = asdict(self)
        d.pop("llm_api_key", None)
        return d
