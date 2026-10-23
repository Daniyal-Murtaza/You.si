"""Tests for persona inference, evidence enforcement and owner verdicts."""

import json

import pytest

from yousi.model import ScriptedChat
from yousi.persona import (
    Persona,
    Trait,
    apply_verdict,
    infer_persona,
    load_persona,
    save_persona,
)
from yousi.vault import Memory, Vault


def _vault(tmp_path):
    vault = Vault.create(tmp_path / "v", "Daniyal")
    for text, domain, kind in [
        ("I usually sleep around midnight.", "routines", "habit"),
        ("I'd rather fix something than replace it.", "values", "belief"),
        ("I use short sentences and dry humour.", "communication", "style"),
    ]:
        vault.add_memory(
            Memory(
                memory_id=Memory.make_id(text, "t"),
                text=text,
                kind=kind,
                domain=domain,
                source_kind="interview",
                source_ref="t",
                excerpt=text,
            )
        )
    return vault


def _ids(vault):
    return [m.memory_id for m in vault.all_memories()]


def test_infer_keeps_only_evidenced_traits(tmp_path):
    vault = _vault(tmp_path)
    sleep, fix, style = _ids(vault)
    reply = json.dumps(
        {
            "summary": "A builder who fixes things.",
            "values": [
                {"text": "Prefers repairing to replacing", "evidence": [fix]},
                {"text": "Loves luxury", "evidence": ["m000000000000"]},
            ],  # bogus id
            "routines": [{"text": "Sleeps around midnight", "evidence": [sleep]}],
            "speech_style": [{"text": "Short sentences, dry humour", "evidence": [style]}],
            "traits": [{"text": "Patient", "evidence": []}],  # no evidence
            "big_five": {
                "openness": 70,
                "conscientiousness": 80,
                "extraversion": 40,
                "agreeableness": 60,
                "neuroticism": 30,
                "rationale": "guess",
            },
        }
    )
    persona = infer_persona(ScriptedChat(script=[reply]), vault)
    assert persona.summary == "A builder who fixes things."
    assert [t.text for t in persona.values] == ["Prefers repairing to replacing"]
    assert persona.traits == []
    assert all(t.status == "proposed" for _, t in persona.all_traits())
    assert persona.confirmation_rate() == 0.0
    assert persona.big_five.conscientiousness == 80


def test_empty_vault_gives_empty_persona(tmp_path):
    vault = Vault.create(tmp_path / "v", "D")
    persona = infer_persona(ScriptedChat(script=["should not be called"]), vault)
    assert persona.owner_name == "D" and persona.all_traits() == []


def test_verdicts_persist_and_survive_reinference(tmp_path):
    vault = _vault(tmp_path)
    sleep, fix, style = _ids(vault)
    first = json.dumps(
        {
            "values": [
                {"text": "Prefers repairing to replacing", "evidence": [fix]},
                {"text": "Values punctuality", "evidence": [sleep]},
            ],
            "routines": [{"text": "Sleeps around midnight", "evidence": [sleep]}],
        }
    )
    persona = infer_persona(ScriptedChat(script=[first]), vault)
    apply_verdict(persona, "values", 0, "confirmed")
    apply_verdict(persona, "values", 1, "rejected", note="not really")
    apply_verdict(persona, "routines", 0, "edited", edited_text="Sleeps around 1am lately")
    save_persona(vault, persona)
    loaded = load_persona(vault)
    assert loaded.values[0].status == "confirmed"
    assert loaded.routines[0].text == "Sleeps around 1am lately"
    assert loaded.routines[0].evidence == [sleep]  # edits keep evidence
    assert loaded.confirmation_rate() == pytest.approx(2 / 3)

    # Re-infer: the model proposes the rejected one again and forgets a confirmed one.
    second = json.dumps(
        {
            "values": [
                {"text": "Values punctuality", "evidence": [sleep]},
                {"text": "Curious about how things work", "evidence": [fix]},
            ],
            "routines": [],
        }
    )
    again = infer_persona(ScriptedChat(script=[second]), vault, previous=loaded)
    texts = {t.text for _, t in again.all_traits()}
    assert "Values punctuality" not in texts  # rejected stays rejected
    assert "Prefers repairing to replacing" in texts  # confirmed is never lost
    assert "Sleeps around 1am lately" in texts
    assert "Curious about how things work" in texts
    assert again.values[-1].status in ("confirmed", "proposed")


def test_prompt_rendering_labels_guesses():
    p = Persona(
        owner_name="D",
        summary="Quiet builder.",
        values=[Trait(text="Honesty", evidence=["m1"], status="confirmed")],
        fears=[Trait(text="Being forgotten", evidence=["m2"])],
    )
    confirmed_only = p.to_prompt()
    assert "Honesty" in confirmed_only and "Being forgotten" not in confirmed_only
    with_guesses = p.to_prompt(include_unconfirmed=True)
    assert "Being forgotten (unconfirmed guess)" in with_guesses
    assert "uncertain" in with_guesses


def test_edit_requires_text():
    p = Persona(owner_name="D", values=[Trait(text="x", evidence=["m1"])])
    with pytest.raises(ValueError):
        apply_verdict(p, "values", 0, "edited")
