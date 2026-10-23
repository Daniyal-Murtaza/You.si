"""Turn the files a person actually has into plain text units with provenance.

Supported today (all parsed with the standard library, no paid services):

* ``.md`` / ``.txt``             - notes, journals, essays
* ``.json``                      - ChatGPT exports and generic message arrays
* ``.eml`` / ``.mbox``           - email
* WhatsApp ``.txt`` exports      - detected by line format
* ``.pdf``                       - text layer only, via pypdf if installed

Each parser yields :class:`Unit` objects: a piece of text, who said it (when
known), when, and a reference that points back into the source file. Units
are the raw material for memory extraction; nothing here calls a model.
"""

from __future__ import annotations

import email
import json
import mailbox
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from email import policy
from pathlib import Path

from yousi.logging_setup import get_logger

log = get_logger("ingest.parsers")


@dataclass
class Unit:
    text: str
    ref: str  # e.g. "notes.md#p3", "chat.json#m12", "mail.mbox#7"
    speaker: str = ""  # "owner", a name, or "" when unknown
    when: str = ""  # ISO-ish date string when available
    meta: dict[str, str] = field(default_factory=dict)


class UnsupportedFileError(ValueError):
    pass


_WHATSAPP = re.compile(
    r"^\[?(\d{1,2}/\d{1,2}/\d{2,4}),? (\d{1,2}:\d{2}(?::\d{2})?(?: ?[AP]M)?)\]? ?[-–]? ?"
    r"([^:]+?): (.*)$"
)
_SYSTEM_WA = ("Messages and calls are end-to-end encrypted", "<Media omitted>", "image omitted")


def _paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


# --------------------------------------------------------------------------
# text / markdown
# --------------------------------------------------------------------------


def parse_text(path: Path, data: bytes, *, owner_name: str) -> Iterator[Unit]:
    text = data.decode("utf-8", errors="replace")
    if _looks_like_whatsapp(text):
        yield from parse_whatsapp(path, text, owner_name=owner_name)
        return
    for i, para in enumerate(_paragraphs(text), 1):
        yield Unit(text=para, ref=f"{path.name}#p{i}", speaker="owner")


def _looks_like_whatsapp(text: str) -> bool:
    lines = [ln for ln in text.splitlines()[:40] if ln.strip()]
    return bool(lines) and sum(bool(_WHATSAPP.match(ln)) for ln in lines) >= max(3, len(lines) // 2)


def parse_whatsapp(path: Path, text: str, *, owner_name: str) -> Iterator[Unit]:
    """Group consecutive lines by speaker so a thought split over messages stays together."""
    owner_key = owner_name.strip().lower()
    buffer: list[str] = []
    speaker = when = ""
    n = 0

    def flush() -> Iterator[Unit]:
        nonlocal buffer, n
        if buffer:
            n += 1
            role = (
                "owner"
                if speaker.lower().startswith(owner_key[: max(3, len(owner_key))])
                else speaker
            )
            yield Unit(
                text="\n".join(buffer),
                ref=f"{path.name}#m{n}",
                speaker=role,
                when=when,
                meta={"channel": "whatsapp"},
            )
            buffer = []

    for line in text.splitlines():
        m = _WHATSAPP.match(line)
        if m:
            date, time, who, body = m.groups()
            if any(s in body for s in _SYSTEM_WA):
                continue
            if who != speaker:
                yield from flush()
                speaker, when = who, f"{date} {time}"
            buffer.append(body.strip())
        elif buffer and line.strip():
            buffer.append(line.strip())  # continuation line
    yield from flush()


# --------------------------------------------------------------------------
# json (ChatGPT export, generic messages)
# --------------------------------------------------------------------------


def parse_json(path: Path, data: bytes, *, owner_name: str) -> Iterator[Unit]:
    try:
        payload = json.loads(data.decode("utf-8", errors="replace"))
    except json.JSONDecodeError as exc:
        raise UnsupportedFileError(f"{path.name}: not valid JSON ({exc})") from exc

    # ChatGPT "conversations.json": list of {title, mapping: {id: {message: {...}}}}
    if (
        isinstance(payload, list)
        and payload
        and isinstance(payload[0], dict)
        and "mapping" in payload[0]
    ):
        yield from _parse_chatgpt(path, payload)
        return
    # Generic: list of {role|speaker|from, content|text, timestamp?}
    if isinstance(payload, dict) and isinstance(payload.get("messages"), list):
        payload = payload["messages"]
    if isinstance(payload, list):
        for i, item in enumerate(payload, 1):
            if not isinstance(item, dict):
                continue
            text = item.get("content") or item.get("text") or item.get("body") or ""
            if isinstance(text, list):  # content blocks
                text = " ".join(
                    str(b.get("text", "")) if isinstance(b, dict) else str(b) for b in text
                )
            text = str(text).strip()
            if not text:
                continue
            who = str(item.get("role") or item.get("speaker") or item.get("from") or "")
            speaker = "owner" if who.lower() in ("user", "owner", owner_name.lower()) else who
            yield Unit(
                text=text,
                ref=f"{path.name}#m{i}",
                speaker=speaker,
                when=str(item.get("timestamp") or item.get("date") or ""),
            )
        return
    raise UnsupportedFileError(f"{path.name}: unrecognised JSON shape")


def _parse_chatgpt(path: Path, conversations: list[dict]) -> Iterator[Unit]:
    for c_idx, convo in enumerate(conversations, 1):
        title = str(convo.get("title") or f"conversation {c_idx}")
        nodes = convo.get("mapping") or {}
        messages = []
        for node in nodes.values():
            msg = (node or {}).get("message") or {}
            role = ((msg.get("author") or {}).get("role")) or ""
            parts = ((msg.get("content") or {}).get("parts")) or []
            text = " ".join(str(p) for p in parts if isinstance(p, str)).strip()
            if role == "user" and text:
                messages.append((msg.get("create_time") or 0, text))
        for m_idx, (ts, text) in enumerate(sorted(messages), 1):
            # Only the person's own messages: that is what tells us about them.
            yield Unit(
                text=text,
                ref=f"{path.name}#c{c_idx}m{m_idx}",
                speaker="owner",
                when=str(ts),
                meta={"title": title, "channel": "chatgpt"},
            )


# --------------------------------------------------------------------------
# email
# --------------------------------------------------------------------------


def _email_units(msg: email.message.EmailMessage, ref: str, owner_name: str) -> Iterator[Unit]:
    body = ""
    try:
        part = msg.get_body(preferencelist=("plain", "html"))
        if part is not None:
            body = part.get_content()
            if part.get_content_type() == "text/html":
                body = re.sub(r"<[^>]+>", " ", body)
    except Exception:  # noqa: BLE001 - malformed mail is data, not a crash
        body = ""
    body = re.sub(r"\n>.*", "", body)  # drop quoted replies
    body = re.sub(r"\n--\s*\n.*", "", body, flags=re.S)  # drop signatures
    body = body.strip()
    if not body:
        return
    sender = str(msg.get("From", ""))
    speaker = "owner" if owner_name.lower() in sender.lower() else sender
    yield Unit(
        text=body[:4000],
        ref=ref,
        speaker=speaker,
        when=str(msg.get("Date", "")),
        meta={"subject": str(msg.get("Subject", "")), "channel": "email"},
    )


def parse_eml(path: Path, data: bytes, *, owner_name: str) -> Iterator[Unit]:
    msg = email.message_from_bytes(data, policy=policy.default)
    yield from _email_units(msg, f"{path.name}#1", owner_name)


def parse_mbox(path: Path, data: bytes, *, owner_name: str) -> Iterator[Unit]:
    box = mailbox.mbox(
        str(path), factory=lambda f: email.message_from_binary_file(f, policy=policy.default)
    )
    for i, msg in enumerate(box, 1):
        yield from _email_units(msg, f"{path.name}#{i}", owner_name)


# --------------------------------------------------------------------------
# pdf (optional dependency)
# --------------------------------------------------------------------------


def parse_pdf(path: Path, data: bytes, *, owner_name: str) -> Iterator[Unit]:
    try:
        from io import BytesIO

        from pypdf import PdfReader
    except ImportError as exc:
        raise UnsupportedFileError("PDF support needs `pip install pypdf`") from exc
    reader = PdfReader(BytesIO(data))
    for page_no, page in enumerate(reader.pages, 1):
        text = (page.extract_text() or "").strip()
        for i, para in enumerate(_paragraphs(text), 1):
            yield Unit(text=para, ref=f"{path.name}#page{page_no}p{i}", speaker="owner")


# --------------------------------------------------------------------------
# dispatch
# --------------------------------------------------------------------------

PARSERS = {
    ".md": parse_text,
    ".txt": parse_text,
    ".markdown": parse_text,
    ".json": parse_json,
    ".eml": parse_eml,
    ".mbox": parse_mbox,
    ".pdf": parse_pdf,
}


def parse_file(path: Path, data: bytes, *, owner_name: str) -> list[Unit]:
    parser = PARSERS.get(path.suffix.lower())
    if parser is None:
        raise UnsupportedFileError(f"{path.name}: unsupported type {path.suffix or '(none)'}")
    units = [u for u in parser(path, data, owner_name=owner_name) if len(u.text.split()) >= 3]
    log.debug("parsed %s -> %d units", path.name, len(units))
    return units
