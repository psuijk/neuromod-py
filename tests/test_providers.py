from __future__ import annotations

import pytest

from neuromod.messages import Message, TextContent, user_message
from neuromod.models import Claude, Model
from neuromod.providers import (
    ProviderRequest,
    ProviderResponse,
    TokenUsage,
    TokenCount,
    ToolDefinition,
    TextDeltaEvent,
    ToolCallStartEvent,
    ToolCallDeltaEvent,
    ToolCallsReadyEvent,
    ToolCallInfo,
    NeuromodError,
    AuthError,
    RateLimitError,
    NetworkError,
    APIError,
    ErrorCode,
    is_neuromod_error,
    is_auth_error,
    is_rate_limit_error,
    is_network_error,
    is_api_error,
)
from neuromod.providers.factory import ProviderFactory, ProviderFactoryConfig


# ── Provider types ─────────────────────────────────


def test_provider_request_creation():
    msg = user_message("hello")
    req = ProviderRequest(model=Claude.Sonnet4_6, messages=[msg])
    assert req.model is Claude.Sonnet4_6
    assert len(req.messages) == 1
    assert req.tools is None
    assert req.temperature is None


def test_provider_response_creation():
    msg = Message(role="assistant", content=[TextContent(text="hi")])
    usage = TokenUsage(input_tokens=10, output_tokens=5)
    resp = ProviderResponse(message=msg, usage=usage)
    assert resp.message is msg
    assert resp.usage is usage


def test_token_usage_creation():
    u = TokenUsage(input_tokens=100, output_tokens=50)
    assert u.input_tokens == 100
    assert u.output_tokens == 50


def test_token_usage_optional_cache_fields():
    u1 = TokenUsage(input_tokens=100, output_tokens=50)
    assert u1.cache_read_tokens is None
    assert u1.cache_write_tokens is None

    u2 = TokenUsage(input_tokens=100, output_tokens=50, cache_read_tokens=20, cache_write_tokens=10)
    assert u2.cache_read_tokens == 20
    assert u2.cache_write_tokens == 10


def test_token_count_creation():
    tc = TokenCount(tokens=42, exact=True)
    assert tc.tokens == 42
    assert tc.exact is True


def test_tool_definition_creation():
    td = ToolDefinition(
        name="get_weather",
        description="Get weather",
        parameters={"type": "object", "properties": {"location": {"type": "string"}}},
    )
    assert td.name == "get_weather"
    assert td.description == "Get weather"
    assert "properties" in td.parameters


def test_provider_stream_event_types():
    e1 = TextDeltaEvent(text="hello")
    assert e1.type == "text_delta"
    assert e1.text == "hello"

    e2 = ToolCallStartEvent(id="1", name="foo")
    assert e2.type == "tool_call_start"

    e3 = ToolCallDeltaEvent(id="1", arguments_delta='{"x":')
    assert e3.type == "tool_call_delta"

    info = ToolCallInfo(id="1", name="foo", arguments={"x": 1})
    e4 = ToolCallsReadyEvent(calls=[info])
    assert e4.type == "tool_calls_ready"
    assert len(e4.calls) == 1


# ── Errors ─────────────────────────────────────────


def test_auth_error_fields():
    err = AuthError("anthropic")
    assert err.provider == "anthropic"
    assert err.code == "AUTH"


def test_auth_error_message_format():
    err = AuthError("openai")
    assert str(err) == "Authentication failed for openai"


def test_auth_error_is_neuromod_error():
    err = AuthError("anthropic")
    assert isinstance(err, NeuromodError)


def test_rate_limit_error_with_retry_after():
    err = RateLimitError("anthropic", retry_after_ms=5000)
    assert err.retry_after_ms == 5000


def test_rate_limit_error_without_retry_after():
    err = RateLimitError("openai")
    assert err.retry_after_ms is None


def test_network_error_fields():
    err = NetworkError("google")
    assert err.provider == "google"
    assert err.code == "NETWORK"
    assert str(err) == "Network error calling google"


def test_api_error_truncates_body():
    long_body = "x" * 500
    err = APIError("anthropic", 500, long_body)
    assert len(str(err)) < len(long_body)
    assert "x" * 200 in str(err)


def test_api_error_preserves_full_body():
    long_body = "x" * 500
    err = APIError("anthropic", 500, long_body)
    assert err.body == long_body
    assert len(err.body) == 500


def test_error_code_constants():
    assert ErrorCode.AUTH == "AUTH"
    assert ErrorCode.RATE_LIMIT == "RATE_LIMIT"
    assert ErrorCode.NETWORK == "NETWORK"
    assert ErrorCode.API == "API_ERROR"


def test_is_auth_error_true():
    assert is_auth_error(AuthError("x")) is True


def test_is_auth_error_false():
    assert is_auth_error(ValueError("nope")) is False
    assert is_auth_error(NetworkError("x")) is False


def test_is_rate_limit_error_true():
    assert is_rate_limit_error(RateLimitError("x")) is True


def test_is_network_error_true():
    assert is_network_error(NetworkError("x")) is True


def test_is_api_error_true():
    assert is_api_error(APIError("x", 500, "body")) is True


def test_is_neuromod_error_false_for_regular_exception():
    assert is_neuromod_error(ValueError("nope")) is False


def test_error_cause_chaining():
    original = ValueError("original error")
    err = AuthError("anthropic", cause=original)
    assert err.__cause__ is original


# ── Factory ────────────────────────────────────────


def test_factory_raises_not_implemented_for_unknown_provider():
    factory = ProviderFactory(ProviderFactoryConfig(api_keys={"openai": "sk-test"}))
    with pytest.raises(NotImplementedError):
        factory.get("openai")


def test_factory_builds_anthropic_provider():
    factory = ProviderFactory(ProviderFactoryConfig(api_keys={"anthropic": "sk-test"}))
    provider = factory.get("anthropic")
    assert provider is not None


def test_factory_key_from_config():
    factory = ProviderFactory(ProviderFactoryConfig(api_keys={"anthropic": "sk-config"}))
    provider = factory.get("anthropic")
    assert provider is not None


def test_factory_key_from_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-env")
    factory = ProviderFactory(ProviderFactoryConfig())
    provider = factory.get("anthropic")
    assert provider is not None


def test_factory_missing_key_raises(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    factory = ProviderFactory(ProviderFactoryConfig())
    with pytest.raises(NeuromodError):
        factory.get("openai")


def test_factory_caches_by_provider():
    factory = ProviderFactory(ProviderFactoryConfig(api_keys={"anthropic": "sk-test"}))
    provider1 = factory.get("anthropic")
    provider2 = factory.get("anthropic")
    assert provider1 is provider2


def test_factory_caches_by_api_key():
    factory = ProviderFactory(ProviderFactoryConfig(api_keys={"anthropic": "sk-test"}))
    provider1 = factory.get("anthropic")
    provider2 = factory.get("anthropic", api_key="sk-override")
    assert provider1 is not provider2


def test_factory_google_dual_env_var(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_AI_API_KEY", raising=False)

    # GEMINI_API_KEY takes priority
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    factory = ProviderFactory(ProviderFactoryConfig())
    with pytest.raises(NotImplementedError):
        factory.get("google")

    # Falls back to GOOGLE_AI_API_KEY
    monkeypatch.delenv("GEMINI_API_KEY")
    monkeypatch.setenv("GOOGLE_AI_API_KEY", "google-key")
    factory2 = ProviderFactory(ProviderFactoryConfig())
    with pytest.raises(NotImplementedError):
        factory2.get("google")

    # Neither set → raises NeuromodError
    monkeypatch.delenv("GOOGLE_AI_API_KEY")
    factory3 = ProviderFactory(ProviderFactoryConfig())
    with pytest.raises(NeuromodError):
        factory3.get("google")
