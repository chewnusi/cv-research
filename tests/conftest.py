import pytest

from cvorch.config import Settings
from cvorch.workspace import default_workspace


def make_settings(profile: str = "planned", **overrides) -> Settings:
    return Settings(llm_model="scripted", **overrides).with_profile(profile)


@pytest.fixture
def ws():
    return default_workspace()


@pytest.fixture(autouse=True)
def _no_real_api(monkeypatch):
    """No unit test may reach the OpenAI API."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
