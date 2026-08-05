"""Logging and per-request tracing.

Two things live here:

1. `configure_logging` -- structured logging. In production we emit one JSON
   object per line so a log shipper can index the fields; locally we emit a
   readable line because staring at JSON while developing is miserable.

2. `QueryTrace` -- a lightweight span recorder for a single RAG query. Rather
   than pulling in OpenTelemetry for a take-home, this captures the same shape
   of information (named stages, durations, attributes) in a form we can both
   log and return to the UI. The frontend renders it as a "how did we get this
   answer" inspector, which doubles as the observability story and as a
   debugging tool. Swapping this for real OTel spans later is a contained
   change: the call sites are already stage-shaped.
"""

import json
import logging
import sys
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List

_request_id: ContextVar[str] = ContextVar("request_id", default="-")


def set_request_id(value: str) -> None:
    _request_id.set(value)


def get_request_id() -> str:
    return _request_id.get()


def new_request_id() -> str:
    return uuid.uuid4().hex[:12]


class _RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = get_request_id()
        return True


class _JsonFormatter(logging.Formatter):
    """One JSON object per line, with any `extra=` fields merged in."""

    _RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {
        "asctime",
        "message",
        "taskName",
    }

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "request_id": getattr(record, "request_id", "-"),
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in self._RESERVED and key != "request_id":
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class _ConsoleFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = (
            f"{self.formatTime(record, '%H:%M:%S')} "
            f"{record.levelname:<5} "
            f"[{getattr(record, 'request_id', '-')}] "
            f"{record.name}: {record.getMessage()}"
        )
        extras = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _JsonFormatter._RESERVED and key != "request_id"
        }
        if extras:
            base += " | " + " ".join(f"{k}={v}" for k, v in extras.items())
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base


def configure_logging(level: str = "INFO", fmt: str = "console") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_JsonFormatter() if fmt == "json" else _ConsoleFormatter())
    handler.addFilter(_RequestIdFilter())

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())

    # uvicorn installs its own handlers; route them through ours instead.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(name)
        logger.handlers = []
        logger.propagate = True

    # These are chatty at INFO and tell us nothing we want.
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


@dataclass
class Stage:
    name: str
    duration_ms: float
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryTrace:
    """Records what happened during one question, stage by stage."""

    request_id: str = field(default_factory=get_request_id)
    stages: List[Stage] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    _started: float = field(default_factory=time.perf_counter)

    @contextmanager
    def stage(self, name: str) -> Iterator[Dict[str, Any]]:
        """Time a block and attach arbitrary attributes to it.

        Usage:
            with trace.stage("retrieval") as attrs:
                ...
                attrs["chunks"] = len(hits)
        """
        attrs: Dict[str, Any] = {}
        started = time.perf_counter()
        try:
            yield attrs
        finally:
            elapsed = (time.perf_counter() - started) * 1000
            self.stages.append(Stage(name=name, duration_ms=round(elapsed, 2), attributes=attrs))

    def set(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    @property
    def total_ms(self) -> float:
        return round((time.perf_counter() - self._started) * 1000, 2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "total_ms": self.total_ms,
            "stages": [
                {"name": s.name, "duration_ms": s.duration_ms, "attributes": s.attributes}
                for s in self.stages
            ],
            "attributes": self.attributes,
        }

    def log(self, logger: logging.Logger, message: str = "query completed") -> None:
        logger.info(
            message,
            extra={
                "total_ms": self.total_ms,
                "stage_ms": {s.name: s.duration_ms for s in self.stages},
                **self.attributes,
            },
        )


@dataclass
class Counters:
    """Deliberately tiny in-process metrics.

    Real deployments should scrape Prometheus or push to CloudWatch; this
    exists so `/api/system/metrics` shows something useful in a demo without
    adding a metrics backend to the compose file.
    """

    values: Dict[str, float] = field(default_factory=dict)

    def increment(self, name: str, amount: float = 1.0) -> None:
        self.values[name] = self.values.get(name, 0.0) + amount

    def observe(self, name: str, value: float) -> None:
        self.increment(f"{name}_count")
        self.increment(f"{name}_sum", value)

    def snapshot(self) -> Dict[str, float]:
        out = dict(self.values)
        for key in list(out):
            if key.endswith("_sum"):
                base = key[: -len("_sum")]
                count = out.get(f"{base}_count", 0.0)
                if count:
                    out[f"{base}_avg"] = round(out[key] / count, 2)
        return out


METRICS = Counters()


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


__all__ = [
    "configure_logging",
    "get_logger",
    "get_request_id",
    "new_request_id",
    "set_request_id",
    "QueryTrace",
    "METRICS",
]
