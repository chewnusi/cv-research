import pytest

from cvorch.config import PROFILES, Settings


def test_primary_profiles():
    primary = {p.name for p in PROFILES.values() if p.primary}
    assert primary == {"baseline", "planned"}
    planned = PROFILES["planned"]
    assert planned.planner_validate and planned.planner_repair_attempts == 0
    assert not planned.planner_auto_prerequisites and planned.planner_structured_output == "json_schema"


def test_profile_from_env(monkeypatch):
    monkeypatch.setenv("ORCH_PROFILE", "planned_repair")
    monkeypatch.setenv("LLM_MODEL", "m")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.3")
    monkeypatch.setenv("LLM_SEED", "7")
    s = Settings.from_env(load_dotenv_file=False)
    assert s.profile == "planned_repair" and s.mode == "planned" and s.planner_repair_attempts == 1
    assert (s.llm_model, s.llm_temperature, s.llm_seed) == ("m", 0.3, 7)
    assert Settings.from_env(profile="baseline", load_dotenv_file=False).mode == "baseline"


def test_legacy_flags_are_ignored_with_warning(monkeypatch, capsys):
    monkeypatch.delenv("ORCH_PROFILE", raising=False)
    monkeypatch.delenv("ORCH_MODE", raising=False)
    monkeypatch.setenv("PLANNER_REPAIR_ATTEMPTS", "3")
    s = Settings.from_env(load_dotenv_file=False)
    assert s.profile == "planned" and s.planner_repair_attempts == 0
    assert "PLANNER_REPAIR_ATTEMPTS is ignored" in capsys.readouterr().err


def test_unknown_profile():
    with pytest.raises(ValueError):
        Settings().with_profile("nope")


def test_public_dict_hides_key():
    assert "llm_api_key" not in Settings(llm_api_key="secret").public_dict()
