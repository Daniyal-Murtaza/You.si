"""Tests for the twin: grounding, citations, don't-know, corrections, off switch."""

import json

import pytest

from yousi.ingest.retrieval import HashEmbedder
from yousi.model import ScriptedChat
from yousi.persona import Persona, Trait, save_persona
from yousi.twin import Twin, TwinDisabledError
from yousi.vault import Memory, Vault

NO_CORRECTION = json.dumps({"is_correction": False})


def _vault(tmp_path):
    vault = Vault.create(tmp_path / "v", "Daniyal")
    emb = HashEmbedder()
    ids = {}
    for key, text, domain in [
        ("sleep", "I usually sleep around midnight.", "routines"),
        ("father", "My father is called Murtaza.", "family"),
        ("city", "I live in Seattle.", "places"),
    ]:
        m = Memory(
            memory_id=Memory.make_id(text, "t"),
            text=text,
            kind="fact",
            domain=domain,
            source_kind="interview",
            source_ref="t",
            excerpt=text,
        )
        vault.add_memory(m, emb.encode([text])[0])
        ids[key] = m.memory_id
    save_persona(
        vault,
        Persona(
            owner_name="Daniyal",
            summary="Dry, direct.",
            speech_style=[
                Trait(text="Short sentences", evidence=[ids["sleep"]], status="confirmed")
            ],
        ),
    )
    return vault, emb, ids


def test_grounded_reply_cites_retrieved_memories(tmp_path):
    vault, emb, ids = _vault(tmp_path)
    model = ScriptedChat(
        script=[NO_CORRECTION, lambda msgs: f"Around midnight, usually [{ids['sleep']}]."]
    )
    twin = Twin(vault, model, emb)
    r = twin.reply("When do you normally sleep?")
    assert r.grounded and [m.memory_id for m in r.cited] == [ids["sleep"]]
    system = model.calls[1][0].content
    assert "Short sentences" in system and ids["sleep"] in system
    assert "Dry, direct." in system
    assert "I am an AI twin of Daniyal" in r.disclosure
    # transcript persisted
    files = list((vault.root / "sessions").glob("chat-owner-*.json"))
    assert files and json.loads(files[0].read_text())["turns"][1]["cited"] == [ids["sleep"]]


def test_fabricated_citation_is_stripped_and_reply_marked_ungrounded(tmp_path):
    vault, emb, ids = _vault(tmp_path)
    model = ScriptedChat(script=[NO_CORRECTION, "I was born in 1990 [m000000000000]."])
    twin = Twin(vault, model, emb)
    r = twin.reply("When were you born?")
    assert "[m000000000000]" not in r.text
    assert r.grounded is False and r.cited == []


def test_no_relevant_memory_means_no_grounding(tmp_path):
    vault, emb, ids = _vault(tmp_path)
    model = ScriptedChat(script=[NO_CORRECTION, "I don't know that about myself yet."])
    twin = Twin(vault, model, emb)
    r = twin.reply("What is your favourite opera?")
    assert r.grounded is False and r.retrieved == [] and r.cited == []
    assert "(none)" in model.calls[1][0].content


def test_owner_correction_updates_vault_and_is_cited(tmp_path):
    vault, emb, ids = _vault(tmp_path)
    correction = json.dumps(
        {
            "is_correction": True,
            "wrong_memory_id": ids["father"],
            "corrected_text": "My father is called Murtaza Zaidi.",
        }
    )

    def reply(msgs):
        # cite whichever father memory the twin was given this turn
        system = msgs[0].content
        new_id = [
            line.split("]")[0][1:] for line in system.splitlines() if "Murtaza Zaidi" in line
        ][0]
        return f"Right, Murtaza Zaidi [{new_id}]."

    model = ScriptedChat(script=[correction, reply])
    twin = Twin(vault, model, emb)
    r = twin.reply("No, my father's name is Murtaza Zaidi, not just Murtaza.")
    assert vault.get(ids["father"]).retracted is True
    assert r.grounded and r.cited[0].text == "My father is called Murtaza Zaidi."
    assert r.cited[0].source_kind == "correction" and r.cited[0].supersedes == ids["father"]
    assert "Corrected memory" in r.text


def test_owner_new_fact_is_remembered(tmp_path):
    vault, emb, ids = _vault(tmp_path)
    new_fact = json.dumps(
        {
            "is_correction": False,
            "new_fact": "I have a cat called Biscuit.",
            "domain": "relationships",
        }
    )
    model = ScriptedChat(script=[new_fact, "Noted. Biscuit it is."])
    twin = Twin(vault, model, emb)
    before = vault.count()
    r = twin.reply("By the way I have a cat called Biscuit.")
    assert vault.count() == before + 1 and "Remembered as" in r.text
    added = [m for m in vault.all_memories() if "Biscuit" in m.text][0]
    assert added.source_kind == "chat" and added.excerpt.startswith("By the way")


def test_visitor_mode_never_learns_and_respects_permissions(tmp_path):
    vault, emb, ids = _vault(tmp_path)
    model = ScriptedChat(script=[lambda msgs: f"Seattle [{ids['city']}]."])
    twin = Twin(vault, model, emb, mode="visitor")
    r = twin.reply("Where did he live?")
    assert r.grounded and len(model.calls) == 1  # no correction check in visitor mode
    assert "Do not promise to learn" in model.calls[0][0].content

    vault.update_meta(allow_posthumous=False)
    with pytest.raises(TwinDisabledError):
        Twin(vault, model, emb, mode="visitor").reply("hello")


def test_off_switch_silences_both_modes(tmp_path):
    vault, emb, ids = _vault(tmp_path)
    vault.update_meta(off_switch_engaged=True)
    for mode in ("owner", "visitor"):
        with pytest.raises(TwinDisabledError, match="switched this twin off"):
            Twin(vault, ScriptedChat(script=["x"]), emb, mode=mode).reply("hi")


def test_history_is_passed_to_model(tmp_path):
    vault, emb, ids = _vault(tmp_path)
    model = ScriptedChat(
        script=[NO_CORRECTION, f"Midnight [{ids['sleep']}].", NO_CORRECTION, "Because I work late."]
    )
    twin = Twin(vault, model, emb)
    twin.reply("When do you sleep?")
    twin.reply("Why so late?")
    roles = [m.role for m in model.calls[3]]
    assert roles == ["system", "user", "assistant", "user"]
