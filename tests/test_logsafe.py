"""Fault injection: user text must not survive into a log line.

The point of these tests is not "the helper drops the text field" -- any
denylist does that. They check the three ways a real service leaks anyway:
a renamed field, a nested structure, and an exception message that happens to
carry the document.
"""

import json
import logging

import pytest

from textscore.logsafe import DEFAULT_ALLOWED_FIELDS, EventLogger, TextLeakError, get_logger

SECRET = "This sentence contains a UNIQUE-NEEDLE-2026 fragment of prose that must never leak."


class CapturingHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(record.getMessage())


@pytest.fixture
def logger():
    log = logging.getLogger("test.logsafe")
    log.handlers.clear()
    log.setLevel(logging.INFO)
    handler = CapturingHandler()
    log.addHandler(handler)
    yield log, handler
    log.handlers.clear()


def assert_no_needle(blob: str) -> None:
    # sliding 20-char windows: catches a leak even if a field was renamed or truncated
    for i in range(0, len(SECRET) - 20, 10):
        frag = SECRET[i : i + 20]
        assert frag not in blob, f"leaked fragment: {frag!r}"


def test_whitelisted_scalars_are_emitted(logger):
    log, handler = logger
    rec = EventLogger(log).event("detect_ok", request_id="req_1", word_count=1234, confidence="high", score=0.87)
    assert json.loads(handler.lines[0])["event"] == "detect_ok"
    assert rec["word_count"] == 1234


def test_unknown_field_is_dropped_not_logged(logger):
    log, handler = logger
    EventLogger(log).event("detect_ok", tier="free", rogue_key="short-value")
    line = handler.lines[0]
    assert "rogue_key" not in line and '"tier": "free"' in line


def test_long_string_raises_instead_of_truncating(logger):
    log, _ = logger
    with pytest.raises(TextLeakError):
        EventLogger(log).event("detect_ok", model=SECRET)


def test_nested_structure_is_dropped(logger):
    log, handler = logger
    EventLogger(log).event("detect_ok", status="ok", payload={"text": SECRET})
    assert_no_needle("\n".join(handler.lines))


def test_exception_message_cannot_smuggle_text(logger):
    log, handler = logger
    events = EventLogger(log)
    try:
        raise RuntimeError(SECRET)
    except RuntimeError as exc:
        # the tempting call: passing str(exc) straight through
        events.event("detect_error", error_type=type(exc).__name__, status="failed")
        joined = "\n".join(handler.lines)
        assert_no_needle(joined)
        assert "RuntimeError" in joined


def test_list_of_short_identifiers_is_allowed(logger):
    log, handler = logger
    EventLogger(log).event("loaded", model="e5-small")
    assert '"model": "e5-small"' in handler.lines[0]


def test_boundary_length_is_inclusive():
    log = logging.getLogger("unused")
    events = EventLogger(log)
    ok = "x" * 64
    assert events.event("detect_ok", model=ok)["model"] == ok
    with pytest.raises(TextLeakError):
        events.event("detect_ok", model="x" * 65)


def test_custom_whitelist():
    log = logging.getLogger("unused")
    events = EventLogger(log, allowed_fields=frozenset({"tenant"}))
    assert "tenant" in events.event("evt", tenant="acme")
    assert "request_id" not in events.event("evt", request_id="req_1", tenant="acme")


def test_default_whitelist_covers_the_common_case():
    assert {"request_id", "word_count", "duration_ms", "error_type"} <= set(DEFAULT_ALLOWED_FIELDS)


def test_get_logger_promotes_info_level():
    log = get_logger("textscore.demo")
    assert log.isEnabledFor(logging.INFO)


def test_forbidden_logger_is_disabled():
    from textscore.logsafe import FORBIDDEN_LOGGER_PREFIX

    assert logging.getLogger(FORBIDDEN_LOGGER_PREFIX).disabled is True
