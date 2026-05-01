from __future__ import annotations

from dataclasses import dataclass

from neuromod.models.model import ProviderName
from neuromod.providers.provider import Provider


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
        cache_key = f"{provider}:{api_key}" if api_key else provider
        if cache_key not in self._cache:
            self._cache[cache_key] = self._build(
                provider, api_key=api_key or "", base_url=base_url,
            )
        return self._cache[cache_key]

    def _build(
        self,
        provider: ProviderName,
        *,
        api_key: str = "",
        base_url: str | None = None,
    ) -> Provider:
        resolved_base_url = base_url
        if resolved_base_url is None and self._config.base_urls:
            resolved_base_url = self._config.base_urls.get(provider)

        if provider == "anthropic":
            from neuromod.providers.anthropic import ClaudeProvider
            return ClaudeProvider(api_key=api_key, base_url=resolved_base_url)

        if provider == "google":
            from neuromod.providers.google import GeminiProvider
            return GeminiProvider(api_key=api_key, base_url=resolved_base_url)

        if provider == "ollama":
            from neuromod.providers.ollama import OllamaProvider
            return OllamaProvider(api_key=api_key, base_url=resolved_base_url)

        if provider == "openai":
            from neuromod.providers.openai import OpenAIProvider
            return OpenAIProvider(api_key=api_key, base_url=resolved_base_url)

        if provider == "xai":
            from neuromod.providers.openai import OpenAIProvider
            return OpenAIProvider(api_key=api_key, base_url=resolved_base_url or "https://api.x.ai/v1")

        raise NotImplementedError(f"{provider} provider not yet implemented")
