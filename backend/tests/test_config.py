from pathlib import Path

import pytest
from pydantic import ValidationError

from app.config import REPO_ROOT, Settings
from tests.fakes import make_settings


def test_relative_and_default_corpus_paths_resolve_at_repo_root() -> None:
    assert make_settings().corpus_dir == REPO_ROOT / "data" / "corpus"
    assert make_settings(corpus_dir="custom").corpus_dir == REPO_ROOT / "custom"
    absolute = (REPO_ROOT / "elsewhere").resolve()
    assert make_settings(corpus_dir=absolute).corpus_dir == absolute
    assert isinstance(make_settings().corpus_dir, Path)


def test_defaults_use_hosted_models() -> None:
    settings = make_settings()
    assert settings.llm_model == "openai/gpt-oss-20b"
    assert settings.embedding_model == "openai/text-embedding-3-small"


@pytest.mark.parametrize(
    "setting",
    [{"retrieval_top_k": 0}, {"llm_timeout_seconds": 0}, {"llm_max_tokens": 0}],
)
def test_invalid_settings_fail_early(setting: dict) -> None:
    with pytest.raises(ValidationError):
        make_settings(**setting)


@pytest.mark.parametrize("key", [None, "", "  "])
def test_a_missing_or_blank_key_is_rejected(key: str | None) -> None:
    with pytest.raises(ValidationError, match="OPENROUTER_API_KEY"):
        make_settings(openrouter_api_key=key)


def test_app_creation_validates_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    from app import main

    def invalid_settings() -> Settings:
        return make_settings(openrouter_api_key=None)

    monkeypatch.setattr(main, "get_settings", invalid_settings)
    with pytest.raises(ValidationError, match="OPENROUTER_API_KEY"):
        main.create_app()


def test_env_example_matches_the_defaults() -> None:
    # A .env copied from a stale example would silently pin old values.
    example = {
        name.lower(): value
        for line in (REPO_ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
        for name, _, value in [line.partition("=")]
    }
    defaults = make_settings()
    assert example.pop("openrouter_api_key") == ""
    for name, value in example.items():
        # Parse each value as Settings would, so "30" and 30.0 compare equal.
        parsed = make_settings(**{name: value})
        assert getattr(parsed, name) == getattr(defaults, name), name
