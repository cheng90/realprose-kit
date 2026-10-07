"""Leak-safe structured logging for services that handle user text.

A text-scoring service logs request metrics next to the payload it was asked to
score, and the payload is somebody's private draft. The usual mitigation --
"don't log the text field" -- is a denylist, and denylists lose: the field gets
renamed, an exception message carries the document, a nested dict quietly
stringifies.

This module inverts it. Only scalars and short identifier-like strings in an
explicit whitelist survive; anything long enough to be prose raises rather than
being truncated, so a mistake surfaces in tests instead of in a log aggregator.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

#: Reasonable default for a scoring service: counters, identifiers, labels.
DEFAULT_ALLOWED_FIELDS = frozenset(
    {
        "request_id",
        "model",
        "tier",
        "word_count",
        "span_count",
        "duration_ms",
        "queue_ms",
        "confidence",
        "score",
        "error_type",
        "status",
    }
)

#: Anything longer than this is prose or a token, not an identifier.
MAX_STR_LEN = 64

#: Loggers named under this prefix are neutered: a library that logs its own
#: inputs downstream of us cannot be trusted to have sanitised them.
FORBIDDEN_LOGGER_PREFIX = "textscore.raw"


class TextLeakError(RuntimeError):
    """A field looked like user content and was refused instead of logged."""


forbid = logging.getLogger(FORBIDDEN_LOGGER_PREFIX)
forbid.disabled = True


class EventLogger:
    """Whitelist-enforcing front door for request-path logging."""

    def __init__(
        self,
        logger: logging.Logger,
        allowed_fields: frozenset[str] = DEFAULT_ALLOWED_FIELDS,
        max_str_len: int = MAX_STR_LEN,
    ) -> None:
        self.logger = logger
        self.allowed = set(allowed_fields)
        self.max_str_len = max_str_len

    def event(self, name: str, **fields: Any) -> dict[str, Any]:
        """Emit one JSON line; returns the safe record (useful in tests).

        Raises:
            TextLeakError: a whitelisted value is a string longer than
                ``max_str_len``, i.e. it is probably user content.
        """
        safe: dict[str, Any] = {"event": name, "ts": round(time.time(), 3)}
        for key, value in fields.items():
            if key not in self.allowed:
                continue  # whitelist, not denylist: unknown keys are dropped
            if isinstance(value, bool) or value is None:
                safe[key] = value
            elif isinstance(value, (int, float)):
                safe[key] = value
            elif isinstance(value, str):
                if len(value) > self.max_str_len:
                    raise TextLeakError(f"field {key!r} looks like user text, not an identifier")
                safe[key] = value
            elif isinstance(value, (list, tuple)) and all(
                isinstance(v, str) and len(v) <= self.max_str_len for v in value
            ):
                safe[key] = list(value)
            else:
                continue  # nested structures can hide text; drop them
        self.logger.info(json.dumps(safe, ensure_ascii=False))
        return safe


def get_logger(name: str = "textscore") -> logging.Logger:
    """Logger pre-set to INFO so structured lines are not swallowed by the root level."""
    logger = logging.getLogger(name)
    if logger.level == logging.NOTSET or logger.level > logging.INFO:
        logger.setLevel(logging.INFO)
    return logger
