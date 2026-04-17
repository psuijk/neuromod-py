from __future__ import annotations

from enum import StrEnum


class NeuromodError(Exception):
    code: str = ""

    def __init__(self, message: str, *, cause: BaseException | None = None) -> None:
        super().__init__(message)
        self.__cause__ = cause


class AuthError(NeuromodError):
    code = "AUTH"

    def __init__(self, provider: str, *, cause: BaseException | None = None) -> None:
        self.provider = provider
        super().__init__(f"Authentication failed for {provider}", cause=cause)


class RateLimitError(NeuromodError):
    code = "RATE_LIMIT"

    def __init__(
        self,
        provider: str,
        *,
        retry_after_ms: int | None = None,
        cause: BaseException | None = None,
    ) -> None:
        self.provider = provider
        self.retry_after_ms = retry_after_ms
        super().__init__(f"Rate limited by {provider}", cause=cause)


class NetworkError(NeuromodError):
    code = "NETWORK"

    def __init__(self, provider: str, *, cause: BaseException | None = None) -> None:
        self.provider = provider
        super().__init__(f"Network error calling {provider}", cause=cause)


class APIError(NeuromodError):
    code = "API_ERROR"

    def __init__(
        self,
        provider: str,
        status_code: int,
        body: str,
        *,
        model_id: str | None = None,
    ) -> None:
        self.provider = provider
        self.status_code = status_code
        self.body = body
        self.model_id = model_id
        super().__init__(f"{provider} returned {status_code}: {body[:200]}")


class ErrorCode(StrEnum):
    AUTH = "AUTH"
    RATE_LIMIT = "RATE_LIMIT"
    NETWORK = "NETWORK"
    API = "API_ERROR"


def is_neuromod_error(err: BaseException) -> bool:
    return isinstance(err, NeuromodError)


def is_auth_error(err: BaseException) -> bool:
    return isinstance(err, AuthError)


def is_rate_limit_error(err: BaseException) -> bool:
    return isinstance(err, RateLimitError)


def is_network_error(err: BaseException) -> bool:
    return isinstance(err, NetworkError)


def is_api_error(err: BaseException) -> bool:
    return isinstance(err, APIError)
