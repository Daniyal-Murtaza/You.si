"""Tests for parsers, retrieval and grounded memory extraction (offline)."""

import json
from pathlib import Path

import pytest

from yousi.ingest.extract import IngestReport, extract_memories, ingest_path
from yousi.ingest.parsers import UnsupportedFileError, parse_file
from yousi.ingest.retrieval import HashEmbedder, Retriever, tokenize
from yousi.model import ScriptedChat
from yousi.vault import Memory, Vault

OWNER = "Daniyal"

WHATSAPP = """\
12/03/2023, 21:14 - Daniyal: I'll be late tonight, stuck at work again
12/03/2023, 21:15 - Daniyal: honestly I think I work too much
12/03/2023, 21:20 - Sara: you always say that
12/03/2023, 21:21 - Sara: come home, dinner is ready
12/03/2023, 21:30 - Daniyal: on my way. love you
13/03/2023, 08:02 - Daniyal: <Media omitted>
"""

NOTES = """\
# Journal

I usually sleep around midnight and wake at seven. Coffee first, always.

My father, Murtaza, taught me to fix things before replacing them. I still do that.

Random reminder: buy batteries.
"""


def test_parse_markdown_paragraphs(tmp_path):
    units = parse_file(Path("journal.md"), NOTES.encode(), owner_name=OWNER)
    assert [u.ref for u in units] == ["journal.md#p2", "journal.md#p3", "journal.md#p4"]
    assert all(u.speaker == "owner" for u in units)


def test_parse_whatsapp_groups_turns_and_marks_speakers():
    units = parse_file(Path("chat.txt"), WHATSAPP.encode(), owner_name=OWNER)
    speakers = [u.speaker for u in units]
    assert speakers == ["owner", "Sara", "owner"]
    assert "I'll be late tonight" in units[0].text and "work too much" in units[0].text
    assert units[0].when.startswith("12/03/2023")
    assert not any("Media omitted" in u.text for u in units)


def test_parse_chatgpt_export_keeps_only_user_turns():
    export = [
        {
            "title": "Career advice",
            "mapping": {
                "a": {
                    "message": {
                        "author": {"role": "user"},
                        "create_time": 1.0,
                        "content": {"parts": ["Should I leave my job to start a company?"]},
                    }
                },
                "b": {
                    "message": {
                        "author": {"role": "assistant"},
                        "create_time": 2.0,
                        "content": {"parts": ["That depends on..."]},
                    }
                },
            },
        }
    ]
    units = parse_file(Path("conversations.json"), json.dumps(export).encode(), owner_name=OWNER)
    assert len(units) == 1 and units[0].speaker == "owner"
    assert units[0].meta["title"] == "Career advice"


def test_parse_generic_messages_and_eml():
    msgs = {
        "messages": [
            {"role": "user", "content": "I grew up in Karachi."},
            {"role": "assistant", "content": "That sounds like a big city."},
        ]
    }
    units = parse_file(Path("m.json"), json.dumps(msgs).encode(), owner_name=OWNER)
    assert [u.speaker for u in units] == ["owner", "assistant"]

    eml = (
        b"From: Daniyal <d@example.com>\nTo: boss@example.com\nSubject: Leave\n"
        b"Date: Mon, 2 Jan 2023 09:00:00 +0000\n\nI need Friday off for my sister's wedding.\n"
        b"\n> old quoted text\n"
    )
    units = parse_file(Path("leave.eml"), eml, owner_name=OWNER)
    assert len(units) == 1 and units[0].speaker == "owner"
    assert "quoted" not in units[0].text and units[0].meta["subject"] == "Leave"


def test_unsupported_and_bad_files():
    with pytest.raises(UnsupportedFileError):
        parse_file(Path("x.exe"), b"\x00", owner_name=OWNER)
    with pytest.raises(UnsupportedFileError):
        parse_file(Path("x.json"), b"{not json", owner_name=OWNER)


# --------------------------------------------------------------------------
# retrieval
# --------------------------------------------------------------------------


def _vault_with(tmp_path, texts):
    vault = Vault.create(tmp_path / "v", OWNER)
    emb = HashEmbedder()
    for text, domain in texts:
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
    return vault, emb


def test_retriever_hybrid_search_and_domain_filter(tmp_path):
    vault, emb = _vault_with(
        tmp_path,
        [
            ("I usually sleep around midnight.", "routines"),
            ("My father is called Murtaza.", "family"),
            ("I live in Seattle.", "places"),
        ],
    )
    r = Retriever(vault, emb)
    hits = r.search("what time do I sleep", top_k=2)
    assert hits and hits[0].memory.text.startswith("I usually sleep")
    assert hits[0].why in ("both", "words", "meaning")
    assert r.search("Murtaza", domain="family")[0].memory.domain == "family"
    assert r.search("Murtaza", domain="places") == []
    assert tokenize("The father's name") == ["father's", "name"]


def test_retriever_refreshes_when_vault_changes(tmp_path):
    vault, emb = _vault_with(tmp_path, [("I love mangoes.", "tastes")])
    r = Retriever(vault, emb)
    assert len(r.search("mangoes")) == 1
    text = "I hate cold weather."
    vault.add_memory(
        Memory(
            memory_id="m2",
            text=text,
            kind="preference",
            domain="tastes",
            source_kind="interview",
            source_ref="t",
            excerpt=text,
        ),
        emb.encode([text])[0],
    )
    assert r.search("cold weather")[0].memory.memory_id == "m2"


# --------------------------------------------------------------------------
# extraction
# --------------------------------------------------------------------------


def _proposal(text, excerpt, ref, **kw):
    base = {
        "text": text,
        "kind": "habit",
        "domain": "routines",
        "excerpt": excerpt,
        "unit_ref": ref,
        "confidence": 0.9,
        "tags": ["sleep"],
    }
    base.update(kw)
    return base


def test_extract_accepts_grounded_and_rejects_fabricated(tmp_path):
    units = parse_file(Path("journal.md"), NOTES.encode(), owner_name=OWNER)
    reply = json.dumps(
        {
            "memories": [
                _proposal(
                    "I usually sleep around midnight and wake at seven.",
                    "I usually sleep around midnight and wake at seven.",
                    "journal.md#p2",
                ),
                _proposal(
                    "My father is called Murtaza.",
                    "My father, Murtaza,",
                    "journal.md#p3",
                    kind="relationship",
                    domain="family",
                ),
                _proposal(
                    "I was born in 1990.",
                    "born in 1990",
                    "journal.md#p3",
                    kind="fact",
                    domain="identity",
                ),  # excerpt not in source -> rejected
                _proposal("I like coffee.", "Coffee first, always.", "journal.md#p99"),  # bad ref
            ]
        }
    )
    model = ScriptedChat(script=[reply])
    memories, rejected = extract_memories(model, units, owner_name=OWNER, source_file="journal.md")
    assert [m.text for m in memories] == [
        "I usually sleep around midnight and wake at seven.",
        "My father is called Murtaza.",
    ]
    assert rejected == 2
    assert memories[0].source_ref == "journal.md:journal.md#p2"
    assert memories[0].excerpt == "I usually sleep around midnight and wake at seven."
    assert "speaker=owner" in model.calls[0][-1].content


def test_extract_caps_confidence_for_other_speakers():
    units = parse_file(Path("chat.txt"), WHATSAPP.encode(), owner_name=OWNER)
    sara = next(u for u in units if u.speaker == "Sara")
    reply = json.dumps(
        {
            "memories": [
                _proposal(
                    "Sara is my partner and cooks dinner for us.",
                    "dinner is ready",
                    sara.ref,
                    kind="relationship",
                    domain="relationships",
                    confidence=0.95,
                )
            ]
        }
    )
    memories, _ = extract_memories(
        ScriptedChat(script=[reply]), units, owner_name=OWNER, source_file="chat.txt"
    )
    assert memories[0].confidence == 0.6


def test_ingest_path_skips_duplicates_and_unsupported(tmp_path):
    vault = Vault.create(tmp_path / "v", OWNER)
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    (inbox / "journal.md").write_text(NOTES)
    (inbox / "photo.jpg").write_bytes(b"\xff\xd8")
    reply = json.dumps(
        {
            "memories": [
                _proposal(
                    "I usually sleep around midnight and wake at seven.",
                    "I usually sleep around midnight and wake at seven.",
                    "journal.md#p2",
                )
            ]
        }
    )
    model = ScriptedChat(script=[reply])
    report = ingest_path(vault, model, HashEmbedder(), inbox)
    assert isinstance(report, IngestReport)
    assert (report.files_seen, report.files_skipped, report.memories_added) == (2, 1, 1)
    assert any("unsupported" in e for e in report.errors)
    assert vault.count() == 1 and len(vault.embeddings()[0]) == 1

    # Second run: same content, nothing re-sent to the model.
    again = ingest_path(vault, model, HashEmbedder(), inbox)
    assert again.memories_added == 0 and again.files_skipped == 2
    assert len(model.calls) == 1
