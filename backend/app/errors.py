"""Domain errors.

These carry an HTTP status and a stable machine-readable `code` so the
frontend can react to specific failures rather than string-matching messages.
"""

from typing import Any, Dict, Optional


class AppError(Exception):
    status_code: int = 400
    code: str = "app_error"

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_payload(self) -> Dict[str, Any]:
        return {"code": self.code, "message": self.message, "details": self.details}


class UnsupportedFileType(AppError):
    status_code = 415
    code = "unsupported_file_type"


class FileTooLarge(AppError):
    status_code = 413
    code = "file_too_large"


class EmptyDocument(AppError):
    """Parsed successfully but yielded no usable text (e.g. a scanned PDF)."""

    status_code = 422
    code = "empty_document"


class SessionNotFound(AppError):
    status_code = 404
    code = "session_not_found"


class ResourceNotFound(AppError):
    status_code = 404
    code = "resource_not_found"


class PreconditionFailed(AppError):
    """The requested action needs documents that have not been uploaded yet."""

    status_code = 409
    code = "precondition_failed"


class LimitExceeded(AppError):
    status_code = 429
    code = "limit_exceeded"


class LLMUnavailable(AppError):
    status_code = 503
    code = "llm_unavailable"


class GuardrailBlocked(AppError):
    """Request refused by an input guardrail. Not an error the user should
    see as a failure -- the API turns this into a polite refusal answer."""

    status_code = 200
    code = "guardrail_blocked"
