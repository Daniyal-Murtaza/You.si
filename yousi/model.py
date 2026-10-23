"""Provider-agnostic chat model.

The twin must outlive any vendor. Everything above this module talks to the
small :class:`ChatModel` protocol; swapping OpenAI for a local Ollama model
(or anything else) is a one-line settings change and touches nothing else.

Three implementations:

* :class:`OpenAIChat` - the default for now.
* :class:`OllamaChat` - open-weights model running locally; no account, no key.
* :class:`ScriptedChat` - deterministic replay for tests and offline demos.

:func:`complete_json` layers "give me JSON matching this schema, with one
repair attempt" on top of any of them.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from yousi.config import Settings
from yousi.logging_setup import get_logger

log = get_logger("model")

T = TypeVar("T", bound=BaseModel)


@dataclass
class Message:
    role: str  # system | user | assistant
    content: str


@dataclass
class Reply:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0


class ChatModel(Protocol):
    name: str

    def chat(
        self, messages: Sequence[Message], *, temperature: float = 0.4, max_tokens: int = 800
    ) -> Reply: ...


class ModelError(RuntimeError):
    """Any provider failure, with a flag saying whether retrying could help."""

    def __init__(self, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable


# --------------------------------------------------------------------------
# OpenAI
# --------------------------------------------------------------------------


class OpenAIChat:
    def __init__(self, api_key: str, model: str, *, timeout_s: float = 90.0) -> None:
        import openai

        self._openai = openai
        self._client = openai.OpenAI(api_key=api_key, timeout=timeout_s, max_retries=2)
        self.name = f"openai:{model}"
        self._model = model

    def chat(
        self, messages: Sequence[Message], *, temperature: float = 0.4, max_tokens: int = 800
    ) -> Reply:
        try:
            r = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": m.role, "content": m.content} for m in messages],
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except (
            self._openai.AuthenticationError,
            self._openai.PermissionDeniedError,
            self._openai.BadRequestError,
        ) as exc:
            raise ModelError(f"OpenAI rejected the request: {exc}", retryable=False) from exc
        except self._openai.OpenAIError as exc:
            raise ModelError(f"OpenAI error: {exc}", retryable=True) from exc
        usage = getattr(r, "usage", None)
        return Reply(
            text=(r.choices[0].message.content or "").strip(),
            model=self.name,
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
        )


# --------------------------------------------------------------------------
# Ollama (local, open weights)
# --------------------------------------------------------------------------


class OllamaChat:
    """Talks to a local Ollama server over its plain HTTP API. No SDK needed."""

    def __init__(
        self, model: str, url: str = "http://localhost:11434", *, timeout_s: float = 180.0
    ) -> None:
        self.name = f"ollama:{model}"
        self._model = model
        self._url = url.rstrip("/")
        self._timeout = timeout_s

    def chat(
        self, messages: Sequence[Message], *, temperature: float = 0.4, max_tokens: int = 800
    ) -> Reply:
        import urllib.error
        import urllib.request

        body = json.dumps(
            {
                "model": self._model,
                "messages": [{"role": m.role, "content": m.content} for m in messages],
                "stream": False,
                "options": {"temperature": temperature, "num_predict": max_tokens},
            }
        ).encode()
        req = urllib.request.Request(
            f"{self._url}/api/chat", data=body, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                data = json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            raise ModelError(
                f"Ollama HTTP {exc.code}: {exc.reason}", retryable=exc.code >= 500
            ) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise ModelError(
                f"cannot reach Ollama at {self._url} ({exc}); is `ollama serve` running?",
                retryable=True,
            ) from exc
        return Reply(
            text=(data.get("message", {}).get("content") or "").strip(),
            model=self.name,
            input_tokens=int(data.get("prompt_eval_count", 0) or 0),
            output_tokens=int(data.get("eval_count", 0) or 0),
        )


# --------------------------------------------------------------------------
# Scripted (tests / offline)
# --------------------------------------------------------------------------


@dataclass
class ScriptedChat:
    """Returns canned replies in order; records every prompt it was given.

    A step may be a string, an exception to raise, or a callable that receives
    the messages and returns a string (so tests can react to the prompt).
    """

    script: list[Any]
    name: str = "scripted"
    calls: list[list[Message]] = field(default_factory=list)

    def chat(
        self, messages: Sequence[Message], *, temperature: float = 0.4, max_tokens: int = 800
    ) -> Reply:
        self.calls.append(list(messages))
        idx = min(len(self.calls) - 1, len(self.script) - 1)
        step = self.script[idx]
        if isinstance(step, Exception):
            raise step
        text = step(list(messages)) if callable(step) else str(step)
        return Reply(text=text, model=self.name)


# --------------------------------------------------------------------------
# Factory and JSON helper
# --------------------------------------------------------------------------


def build_model(settings: Settings) -> ChatModel:
    if settings.provider == "openai":
        return OpenAIChat(
            settings.require_openai(), settings.openai_model, timeout_s=settings.model_timeout_s
        )
    if settings.provider == "ollama":
        return OllamaChat(
            settings.ollama_model,
            settings.ollama_url,
            timeout_s=max(settings.model_timeout_s, 180.0),
        )
    return ScriptedChat(script=["(fake model: no script loaded)"])


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def extract_json(text: str) -> str:
    """Pull the first JSON object/array out of a reply, tolerating code fences."""
    cleaned = _FENCE.sub("", text.strip())
    start = min((i for i in (cleaned.find("{"), cleaned.find("[")) if i >= 0), default=-1)
    if start < 0:
        raise ValueError("no JSON found in model reply")
    depth, in_str, esc = 0, False, False
    opener = cleaned[start]
    closer = "}" if opener == "{" else "]"
    for i in range(start, len(cleaned)):
        ch = cleaned[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
            if depth == 0 and ch == closer:
                return cleaned[start : i + 1]
    raise ValueError("unterminated JSON in model reply")


def complete_json(
    model: ChatModel,
    messages: Sequence[Message],
    schema: type[T],
    *,
    temperature: float = 0.2,
    max_tokens: int = 1200,
) -> T:
    """Ask for JSON that validates against ``schema``; one repair attempt on failure."""
    convo = list(messages)
    schema_hint = json.dumps(schema.model_json_schema(), ensure_ascii=False)
    convo[-1] = Message(
        convo[-1].role,
        f"{convo[-1].content}\n\nReply with ONLY a JSON object matching this JSON Schema, "
        f"no prose:\n{schema_hint}",
    )
    last_error = ""
    for attempt in (1, 2):
        reply = model.chat(convo, temperature=temperature, max_tokens=max_tokens)
        try:
            return schema.model_validate_json(extract_json(reply.text))
        except (ValueError, ValidationError) as exc:
            last_error = str(exc)[:600]
            log.warning("model JSON invalid (attempt %d): %s", attempt, last_error[:160])
            convo = [
                *convo,
                Message("assistant", reply.text),
                Message(
                    "user",
                    f"That was not valid. Error:\n{last_error}\n"
                    "Return ONLY the corrected JSON object.",
                ),
            ]
    raise ModelError(f"model returned invalid JSON twice: {last_error}", retryable=False)
