from __future__ import annotations

import os
from dataclasses import dataclass

from neuromod.models.model import ProviderName
from neuromod.providers.errors import NeuromodError
from neuromod.providers.provider import Provider


_ENV_VARS: dict[str, list[str]] = {
    "anthropic": ["ANTHROPIC_API_KEY"],
    "openai": ["OPENAI_API_KEY"],
    "google": ["GEMINI_API_KEY", "GOOGLE_AI_API_KEY"],
    "xai": ["XAI_API_KEY"],
}

_KEYLESS_PROVIDERS: set[str] = {"ollama"}


@dataclass
class ProviderFactoryConfig:
    api_keys: dict[str, str] | None = None
    base_urls: dict[str, str] | None = None


class ProviderFactory:
    def __init__(self, config: ProviderFactoryConfig) -> None:
        self._config = config
        self._cache: dict[str, Provider] = {}

    def get(
        self,
        provider: ProviderName,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> Provider:
        resolved_key = api_key or self._get_key(provider)
        cache_key = f"{provider}:{api_key}" if api_key else provider
        if cache_key not in self._cache:
            self._cache[cache_key] = self._build(
                provider, api_key=resolved_key, base_url=base_url,
            )
        return self._cache[cache_key]

    def _build(
        self,
        provider: ProviderName,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> Provider:
        resolved_base_url = base_url
        if resolved_base_url is None and self._config.base_urls:
            resolved_base_url = self._config.base_urls.get(provider)

        if provider == "anthropic":
            from neuromod.providers.anthropic import ClaudeProvider
            return ClaudeProvider(api_key=api_key or "", base_url=resolved_base_url)

        if provider == "google":
            from neuromod.providers.google import GeminiProvider
            return GeminiProvider(api_key=api_key or "", base_url=resolved_base_url)

        if provider == "ollama":
            from neuromod.providers.ollama import OllamaProvider
            return OllamaProvider(api_key=api_key or "", base_url=resolved_base_url)

        raise NotImplementedError(f"{provider} provider not yet implemented")

    def _get_key(self, provider: ProviderName) -> str:
        if self._config.api_keys and provider in self._config.api_keys:
            return self._config.api_keys[provider]
        env_key = self._env_key(provider)
        if env_key is not None:
            return env_key
        if provider in _KEYLESS_PROVIDERS:
            return ""
        raise NeuromodError(f"No API key found for {provider}")

    def _env_key(self, provider: ProviderName) -> str | None:
        for var in _ENV_VARS.get(provider, []):
            val = os.environ.get(var)
            if val:
                return val
        return None
