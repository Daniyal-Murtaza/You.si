"""Tests for the model adapter: provider selection, JSON helper, repair, failures."""

import pytest
from pydantic import BaseModel

from yousi.config import Settings
from yousi.model import (
    Message,
    ModelError,
    OllamaChat,
    OpenAIChat,
    ScriptedChat,
    build_model,
    complete_json,
    extract_json,
)


class Answer(BaseModel):
    name: str
    year: int


def test_build_model_by_provider(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-1234567890")
    assert isinstance(build_model(Settings(_env_file=None, YOUSI_PROVIDER="openai")), OpenAIChat)
    assert isinstance(build_model(Settings(_env_file=None, YOUSI_PROVIDER="ollama")), OllamaChat)
    assert isinstance(build_model(Settings(_env_file=None, YOUSI_PROVIDER="fake")), ScriptedChat)


def test_build_openai_requires_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        build_model(Settings(_env_file=None, YOUSI_PROVIDER="openai"))


@pytest.mark.parametrize(
    "text",
    [
        '{"name": "D", "year": 1999}',
        'Sure! Here you go:\n```json\n{"name": "D", "year": 1999}\n```',
        'prefix {"name": "D", "year": 1999} suffix with } brace',
        '{"name": "D }{", "year": 1999}',  # braces inside a string
    ],
)
def test_extract_json_tolerates_noise(text):
    assert Answer.model_validate_json(extract_json(text)) == Answer(
        name="D }{" if "D }{" in text else "D", year=1999
    )


def test_extract_json_fails_clearly():
    with pytest.raises(ValueError):
        extract_json("no json here")
    with pytest.raises(ValueError):
        extract_json('{"unterminated": ')


def test_complete_json_repairs_once_then_succeeds():
    model = ScriptedChat(
        script=['{"name": "D", "year": "nineteen"}', '{"name": "D", "year": 1999}']
    )
    out = complete_json(model, [Message("user", "who?")], Answer)
    assert out == Answer(name="D", year=1999)
    assert len(model.calls) == 2
    # The repair turn shows the model its own output and the validation error.
    repair_prompt = model.calls[1][-1].content
    assert "not valid" in repair_prompt and "year" in repair_prompt
    # The schema lives in the system turn, never mixed into the user's data.
    assert "JSON Schema" in model.calls[0][0].content and model.calls[0][0].role == "system"
    assert "JSON Schema" not in model.calls[0][-1].content


def test_complete_json_gives_up_after_two_failures():
    model = ScriptedChat(script=["garbage", "still garbage"])
    with pytest.raises(ModelError) as exc:
        complete_json(model, [Message("user", "who?")], Answer)
    assert exc.value.retryable is False and len(model.calls) == 2


def test_scripted_chat_can_raise_and_react():
    model = ScriptedChat(
        script=[ModelError("boom", retryable=True), lambda msgs: f"echo: {msgs[-1].content}"]
    )
    with pytest.raises(ModelError):
        model.chat([Message("user", "x")])
    assert model.chat([Message("user", "hello")]).text == "echo: hello"


def test_ollama_unreachable_is_retryable_error():
    model = OllamaChat("llama3.1:8b", url="http://127.0.0.1:1", timeout_s=1)
    with pytest.raises(ModelError) as exc:
        model.chat([Message("user", "hi")])
    assert exc.value.retryable is True and "ollama serve" in str(exc.value)
