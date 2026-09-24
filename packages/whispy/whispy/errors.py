"""Whispy error hierarchy."""


class WhispyError(Exception):
    """Base error for all whispy failures."""


class AuthError(WhispyError):
    """Authentication/authorization failure (401)."""


class EntitlementError(WhispyError):
    """Plan/scope does not permit the operation (403)."""


class NotFoundError(WhispyError):
    """Resource does not exist (404)."""


class APIError(WhispyError):
    """General API/transport failure."""

    def __init__(self, message: str, status_code: int = 0, detail: str = ""):
        super().__init__(message)
        self.status_code = status_code
        self.detail = detail


class SourceError(WhispyError):
    """Base class for observation-source resolution failures."""


class SourceNotFoundError(SourceError, KeyError):
    """No source matches the requested id/name/modality (404-ish)."""

    def __init__(self, key: str, available=None):
        self.key = key
        self.available = list(available or [])
        super().__init__(
            f"no source {key!r}; available: {self.available}")


class AmbiguousSourceError(SourceError, KeyError):
    """A modality/name matched more than one source; a stable id is required."""

    def __init__(self, key: str, candidates=None):
        self.key = key
        self.candidates = list(candidates or [])
        super().__init__(
            f"ambiguous source {key!r}: {self.candidates}; use a stable id")


class SourceUnavailableError(SourceError):
    """The source exists but cannot be opened right now (offline, busy)."""


__all__ = [
    "WhispyError", "AuthError", "EntitlementError", "NotFoundError", "APIError",
    "SourceError", "SourceNotFoundError", "AmbiguousSourceError",
    "SourceUnavailableError",
]
