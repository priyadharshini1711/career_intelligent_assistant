"""FastAPI application factory."""

import time
from contextlib import asynccontextmanager
from typing import Callable

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import analysis, chat, documents, system
from app.config import Settings, get_settings
from app.errors import AppError
from app.observability import (
    METRICS,
    configure_logging,
    get_logger,
    new_request_id,
    set_request_id,
)

logger = get_logger(__name__)

DESCRIPTION = """
Ask questions about how your resume lines up with specific job descriptions.

Upload one resume and up to ten job descriptions, then ask about fit, skill
gaps, experience alignment, or interview preparation. Answers are grounded in
the uploaded documents and cite the extracts they came from.
"""


def create_app(settings: Settings = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(level=settings.log_level, fmt=settings.log_format)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        logger.info(
            "service starting",
            extra={
                "environment": settings.environment,
                "llm_provider": settings.llm_provider,
                "embedding_model": settings.embedding_model,
            },
        )
        # Load the embedding model now so the first user upload does not pay
        # for it. Failing here should not stop the service -- it will retry
        # lazily and /api/system/ready reports the state either way.
        try:
            from app.api.deps import warm_embedder

            warm_embedder()
        except Exception as exc:  # pragma: no cover - depends on the environment
            logger.warning("embedding model warmup failed", extra={"error": str(exc)})

        logger.info("service ready")
        yield
        logger.info("service stopping")

    app = FastAPI(
        title=settings.app_name,
        description=DESCRIPTION,
        version="1.0.0",
        docs_url="/docs",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-Id"],
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next: Callable):
        """Attach a request id, time the request, log the outcome.

        The id is echoed in the response header so a user reporting "the answer
        was wrong" can hand over something that finds the exact trace in the
        logs.
        """
        request_id = request.headers.get("X-Request-Id") or new_request_id()
        set_request_id(request_id)
        started = time.perf_counter()

        try:
            response = await call_next(request)
        except Exception:
            duration_ms = (time.perf_counter() - started) * 1000
            METRICS.increment("requests_failed")
            logger.exception(
                "unhandled error",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "duration_ms": round(duration_ms, 1),
                },
            )
            raise

        duration_ms = (time.perf_counter() - started) * 1000
        response.headers["X-Request-Id"] = request_id
        METRICS.increment("requests_total")
        METRICS.observe("request_latency_ms", duration_ms)

        # Health checks would otherwise dominate the log at one line per probe.
        if not request.url.path.startswith("/api/system/health"):
            logger.info(
                "request",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                    "duration_ms": round(duration_ms, 1),
                },
            )
        return response

    @app.exception_handler(AppError)
    async def handle_app_error(_: Request, exc: AppError) -> JSONResponse:
        logger.info(
            "request rejected", extra={"code": exc.code, "detail": exc.message, **exc.details}
        )
        return JSONResponse(status_code=exc.status_code, content=exc.to_payload())

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        # Pydantic puts the original exception object in `ctx`, which is not
        # JSON-serialisable and turns a 422 into a 500 on the way out.
        errors = [
            {
                "field": ".".join(str(part) for part in error.get("loc", [])),
                "message": error.get("msg", ""),
                "type": error.get("type", ""),
            }
            for error in exc.errors()[:5]
        ]
        return JSONResponse(
            status_code=422,
            content={
                "code": "validation_error",
                "message": "The request was not in the expected shape.",
                "details": {"errors": errors},
            },
        )

    @app.exception_handler(Exception)
    async def handle_unexpected(_: Request, exc: Exception) -> JSONResponse:
        # Never leak an internal message to the client; the request id is the
        # bridge between what the user sees and what the logs hold.
        logger.exception("unhandled exception", extra={"error": str(exc)})
        return JSONResponse(
            status_code=500,
            content={
                "code": "internal_error",
                "message": "Something went wrong on our side. Please try again.",
                "details": {},
            },
        )

    app.include_router(system.router)
    app.include_router(documents.router)
    app.include_router(chat.router)
    app.include_router(analysis.router)

    @app.get("/", include_in_schema=False)
    async def root():
        return {
            "service": settings.app_name,
            "docs": "/docs",
            "health": "/api/system/health",
        }

    return app


app = create_app()
