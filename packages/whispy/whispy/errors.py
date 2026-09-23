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


__all__ = [
    "WhispyError", "AuthError", "EntitlementError", "NotFoundError", "APIError",
]
