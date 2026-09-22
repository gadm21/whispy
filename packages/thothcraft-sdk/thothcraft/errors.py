"""SDK error types."""


class ThothError(Exception):
    """Base error for all remote failures."""


class AuthError(ThothError):
    """Authentication failed or the token expired."""


class EntitlementError(AuthError):
    """The account's plan does not include this capability."""


class NotFoundError(ThothError):
    """The requested device, minute, or resource was not found."""


class APIError(ThothError):
    """The backend returned a non-success response."""

    def __init__(self, message, status_code=None, detail=None):
        super().__init__(message)
        self.status_code = status_code
        self.detail = detail
