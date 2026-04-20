from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from neuromod.config import configure, get_config, resolve_api_key, NeuromodConfig, _config
from neuromod.composition.thread import InMemoryThreadStore


@pytest.fixture(autouse=True)
def reset_config():
    """Reset the config contextvar before each test."""
    token = _config.set(None)
    yield
    _config.set(None)


class TestNeuromodConfig:
    def test_frozen(self):
        config = NeuromodConfig(api_keys={}, base_urls={}, thread_store=None)
        with pytest.raises(AttributeError):
            config.api_keys = {"anthropic": "sk-123"}


class TestConfigure:
    def test_sets_api_keys(self):
        configure(api_keys={"anthropic": "sk-123"})
        config = get_config()
        assert config.api_keys == {"anthropic": "sk-123"}

    def test_sets_base_urls(self):
        configure(base_urls={"openai": "https://my-proxy.com"})
        config = get_config()
        assert config.base_urls == {"openai": "https://my-proxy.com"}

    def test_sets_thread_store(self):
        store = InMemoryThreadStore()
        configure(thread_store=store)
        config = get_config()
        assert config.thread_store is store

    def test_defaults_to_empty_dicts(self):
        configure()
        config = get_config()
        assert config.api_keys == {}
        assert config.base_urls == {}
        assert config.thread_store is None

    def test_replaces_not_merges(self):
        configure(api_keys={"anthropic": "sk-1"})
        configure(api_keys={"openai": "sk-2"})
        config = get_config()
        assert config.api_keys == {"openai": "sk-2"}
        assert "anthropic" not in config.api_keys


class TestGetConfig:
    def test_returns_default_when_not_configured(self):
        config = get_config()
        assert config.api_keys == {}
        assert config.base_urls == {}
        assert config.thread_store is None


class TestResolveApiKey:
    def test_override_wins(self):
        configure(api_keys={"anthropic": "sk-from-configure"})
        assert resolve_api_key("anthropic", override="sk-override") == "sk-override"

    def test_configure_wins_over_env(self):
        configure(api_keys={"anthropic": "sk-from-configure"})
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-from-env"}):
            assert resolve_api_key("anthropic") == "sk-from-configure"

    def test_falls_back_to_env_var(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-from-env"}):
            assert resolve_api_key("anthropic") == "sk-from-env"

    def test_google_tries_gemini_key_first(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "sk-gemini"}):
            assert resolve_api_key("google") == "sk-gemini"

    def test_google_falls_back_to_google_ai_key(self):
        with patch.dict(os.environ, {"GOOGLE_AI_API_KEY": "sk-google"}, clear=False):
            # Make sure GEMINI_API_KEY is not set
            env = os.environ.copy()
            env.pop("GEMINI_API_KEY", None)
            env["GOOGLE_AI_API_KEY"] = "sk-google"
            with patch.dict(os.environ, env, clear=True):
                assert resolve_api_key("google") == "sk-google"

    def test_raises_when_no_key_found(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(RuntimeError, match="No API key for 'anthropic'"):
                resolve_api_key("anthropic")

    def test_raises_for_unknown_provider(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(RuntimeError, match="No API key for 'mistral'"):
                resolve_api_key("mistral")
