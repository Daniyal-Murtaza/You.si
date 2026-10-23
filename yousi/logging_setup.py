"""Logging with secret redaction. Console plus a per-vault log file."""

from __future__ import annotations

import logging
import re
from pathlib import Path

_MASK = "***REDACTED***"
_TOKEN_PATTERNS = (re.compile(r"sk-[A-Za-z0-9_\-]{8,}"),)
_LABELLED_PATTERNS = (
    re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._\-]{8,}"),
    re.compile(r"(?i)(api[_-]?key[\"']?\s*[:=]\s*[\"']?)[A-Za-z0-9._\-]{8,}"),
)


def redact(text: str) -> str:
    for pattern in _TOKEN_PATTERNS:
        text = pattern.sub(_MASK, text)
    for pattern in _LABELLED_PATTERNS:
        text = pattern.sub(lambda m: m.group(1) + _MASK, text)
    return text


class RedactingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact(str(record.msg))
        if record.args:
            record.args = tuple(redact(a) if isinstance(a, str) else a for a in record.args)
        return True


def configure_logging(level: str = "INFO", *, quiet_console: bool = False) -> None:
    # Quiet the ML stack before it is imported; its progress bars are noise in a CLI.
    import os

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")
    root = logging.getLogger("yousi")
    if getattr(root, "_configured", False):
        root.setLevel(level.upper())
        return
    root.setLevel(level.upper())
    root.propagate = False
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S")
    )
    handler.addFilter(RedactingFilter())
    if quiet_console:
        handler.setLevel(logging.WARNING)
    root.addHandler(handler)
    root._configured = True  # type: ignore[attr-defined]
    for noisy in ("httpx", "openai", "urllib3", "sentence_transformers", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    # torch logs a WARNING on every import on macOS ("Redirects are not supported");
    # it is irrelevant to us and lands in the middle of the interview.
    logging.getLogger("torch").setLevel(logging.ERROR)


def attach_file(path: Path) -> logging.Handler:
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
    handler.addFilter(RedactingFilter())
    handler.setLevel(logging.DEBUG)
    logging.getLogger("yousi").addHandler(handler)
    return handler


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"yousi.{name}")
