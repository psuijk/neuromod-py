from dataclasses import dataclass
from contextvars import ContextVar
from neuromod.composition.thread import ThreadStore
from os import environ

from neuromod.models.model import ProviderName
from neuromod.providers.errors import ConfigError
from neuromod.providers.factory import ProviderFactory

@dataclass(frozen=True)
class NeuromodConfig:
    """Configuration for Neuromod."""

    api_keys: dict[str, str]
    base_urls: dict[str, str]
    timeouts: dict[str, float]
    thread_store: ThreadStore | None

_config: ContextVar[NeuromodConfig | None] = ContextVar("neuromod_config", default=None)
_factory: ContextVar[ProviderFactory | None] = ContextVar("neuromod_factory", default=None)

def configure(
    *,
    api_keys: dict[str, str] | None = None,
    base_urls: dict[str, str] | None = None,
    timeouts: dict[str, float] | None = None,
    thread_store: ThreadStore | None = None,
) -> None:
    """Configure Neuromod with API keys, base URLs, timeouts, and thread store."""
    config = NeuromodConfig(
        api_keys=api_keys or {},
        base_urls=base_urls or {},
        timeouts=timeouts or {},
        thread_store=thread_store,
    )
    _config.set(config)
    _factory.set(None)

def get_config() -> NeuromodConfig:
    """Get the current Neuromod configuration."""
    config = _config.get()
    if config is None:
        return NeuromodConfig(api_keys={}, base_urls={}, timeouts={}, thread_store=None)
    return config

def get_factory() -> ProviderFactory:
    """Get or create the cached ProviderFactory for the current config."""
    from neuromod.providers.factory import ProviderFactory, ProviderFactoryConfig

    existing = _factory.get()
    if existing is not None:
        return existing

    cfg = get_config()
    factory = ProviderFactory(ProviderFactoryConfig(
        api_keys=cfg.api_keys if cfg.api_keys else None,
        base_urls=cfg.base_urls if cfg.base_urls else None,
    ))
    _factory.set(factory)
    return factory


_ENV_VAR_NAMES: dict[str, list[str]] = {
    "anthropic": ["ANTHROPIC_API_KEY"],
    "openai": ["OPENAI_API_KEY"],
    "google": ["GEMINI_API_KEY", "GOOGLE_AI_API_KEY"],
    "xai": ["XAI_API_KEY"],
}

_KEYLESS_PROVIDERS: set[str] = {"ollama"}


def resolve_api_key(provider: ProviderName, override: str | None = None) -> str:
    """Resolve an API key using precedence: override > configure() > env var."""
    if override is not None:
        return override

    config = get_config()
    if provider in config.api_keys:
        return config.api_keys[provider]

    for env_var in _ENV_VAR_NAMES.get(provider, []):
        value = environ.get(env_var)
        if value is not None:
            return value

    if provider in _KEYLESS_PROVIDERS:
        return ""

    raise ConfigError(
        f"No API key for '{provider}'. Set it via configure(), "
        f"agent config, or environment variable."
    )


def resolve_timeout(provider: ProviderName, override: float | None = None) -> float | None:
    """Resolve a timeout using precedence: override > configure()."""
    if override is not None:
        return override

    config = get_config()
    if provider in config.timeouts:
        return config.timeouts[provider]

    return None

