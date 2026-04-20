from dataclasses import dataclass
from contextvars import ContextVar
from neuromod.composition.thread import ThreadStore
from os import environ

@dataclass(frozen=True)
class NeuromodConfig:
    """Configuration for Neuromod."""

    api_keys: dict[str, str]
    base_urls: dict[str, str]
    thread_store: ThreadStore | None

_config: ContextVar[NeuromodConfig | None] = ContextVar("neuromod_config", default=None)

def configure(
    *,
    api_keys: dict[str, str] | None = None,
    base_urls: dict[str, str] | None = None,
    thread_store: ThreadStore | None = None,
) -> None:
    """Configure Neuromod with API keys, base URLs, and thread store."""
    config = NeuromodConfig(
        api_keys=api_keys or {},
        base_urls=base_urls or {},
        thread_store=thread_store,
    )
    _config.set(config)

def get_config() -> NeuromodConfig:
    """Get the current Neuromod configuration."""
    config = _config.get()
    if config is None:
        return NeuromodConfig(api_keys={}, base_urls={}, thread_store=None)
    return config


_ENV_VAR_NAMES: dict[str, list[str]] = {
    "anthropic": ["ANTHROPIC_API_KEY"],
    "openai": ["OPENAI_API_KEY"],
    "google": ["GEMINI_API_KEY", "GOOGLE_AI_API_KEY"],
    "xai": ["XAI_API_KEY"],
}


def resolve_api_key(provider: str, override: str | None = None) -> str:
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

    raise RuntimeError(
        f"No API key for '{provider}'. Set it via configure(), "
        f"agent config, or environment variable."
    )

