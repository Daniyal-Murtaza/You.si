"""Tests for the vault: ownership, provenance, corrections, rebuild, export/import."""

import json

import numpy as np
import pytest

from yousi.logging_setup import redact
from yousi.vault import Memory, Vault, VaultNotFoundError


def _mem(text: str, ref: str = "q1", **kw) -> Memory:
    base = dict(
        kind="fact", domain="identity", source_kind="interview", source_ref=ref, excerpt=text
    )
    base.update(kw)
    return Memory(memory_id=Memory.make_id(text, ref), text=text, **base)


def test_create_open_and_meta(tmp_path):
    Vault.create(tmp_path / "me", "Daniyal", birth_year=1999)
    assert (tmp_path / "me" / "vault.json").exists()
    for sub in ("memories", "inbox", "sources", "sessions", "exports"):
        assert (tmp_path / "me" / sub).is_dir()
    meta = Vault.open(tmp_path / "me").load_meta()
    assert meta.owner_name == "Daniyal" and meta.birth_year == 1999
    assert meta.off_switch_engaged is False
    with pytest.raises(FileExistsError):
        Vault.create(tmp_path / "me", "Again")
    with pytest.raises(VaultNotFoundError):
        Vault.open(tmp_path / "nope")


def test_memory_roundtrip_and_markdown_mirror(tmp_path):
    vault = Vault.create(tmp_path / "me", "D")
    m = _mem("I usually sleep around midnight.", domain="routines", kind="habit", tags=["sleep"])
    vault.add_memory(m, np.array([0.1, 0.2, 0.3], dtype=np.float32))
    got = vault.get(m.memory_id)
    assert got == m
    md_files = list((tmp_path / "me" / "memories").rglob("*.md"))
    assert len(md_files) == 1
    text = md_files[0].read_text()
    assert "I usually sleep around midnight." in text
    assert Memory.from_markdown(text) == m  # the file alone is enough to recreate it
    ids, mat = vault.embeddings()
    assert ids == [m.memory_id] and mat.shape == (1, 3)


def test_retract_and_correct_keep_history(tmp_path):
    vault = Vault.create(tmp_path / "me", "D")
    old = vault.add_memory(_mem("My father's name is Murtaza Ali."))
    new = vault.correct(old.memory_id, "My father's name is Murtaza Zaidi.", source_ref="chat-7")
    assert new.supersedes == old.memory_id and new.source_kind == "correction"
    assert vault.get(old.memory_id).retracted is True
    assert "corrected" in vault.get(old.memory_id).retracted_reason
    assert [m.memory_id for m in vault.all_memories()] == [new.memory_id]
    assert len(vault.all_memories(include_retracted=True)) == 2
    assert vault.count() == 1


def test_domain_counts_cover_all_domains(tmp_path):
    vault = Vault.create(tmp_path / "me", "D")
    vault.add_memory(_mem("I live in Seattle.", domain="places"))
    counts = vault.domain_counts()
    assert counts["places"] == 1 and counts["family"] == 0
    assert len(counts) >= 16


def test_rebuild_index_from_markdown_only(tmp_path):
    vault = Vault.create(tmp_path / "me", "D")
    for i in range(5):
        vault.add_memory(_mem(f"memory {i}", ref=f"q{i}"))
    (tmp_path / "me" / "memories.sqlite").unlink()
    assert Vault.open(tmp_path / "me").rebuild_index() == 5
    assert Vault.open(tmp_path / "me").count() == 5


def test_export_and_import_are_lossless(tmp_path):
    vault = Vault.create(tmp_path / "me", "D", executors=["sister@example.com"])
    a = vault.add_memory(_mem("I was born in Karachi.", domain="identity"))
    vault.retract(a.memory_id, "test")
    vault.add_memory(_mem("I love mangoes.", domain="tastes", kind="preference"))
    vault.write_json("persona.json", {"values": ["honesty"]})
    out = vault.export_all()
    payload = json.loads(out.read_text())
    assert payload["format"] == "yousi-vault-export"
    assert len(payload["memories"]) == 2  # retracted ones travel too

    restored = Vault.import_from(out, tmp_path / "copy")
    assert restored.load_meta().executors == ["sister@example.com"]
    assert restored.count() == 1
    assert restored.read_json("persona.json") == {"values": ["honesty"]}


def test_sources_dedupe_by_content(tmp_path):
    vault = Vault.create(tmp_path / "me", "D")
    data = b"hello notes"
    assert not vault.has_source(Vault.content_hash(data))
    stored = vault.archive_source(tmp_path / "notes.md", data, memory_count=3)
    assert stored.exists() and vault.has_source(Vault.content_hash(data))


def test_counters_are_thread_safe(tmp_path):
    import threading

    vault = Vault.create(tmp_path / "me", "D")

    def work():
        for _ in range(20):
            vault.bump("questions")

    threads = [threading.Thread(target=work) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert vault.load_meta().counters["questions"] == 120


def test_redact_masks_keys():
    assert "sk-abcdefghijklmnop" not in redact("key sk-abcdefghijklmnop here")
    assert "REDACTED" in redact("Authorization: Bearer abcdefghijklmnop")


def test_missing_embeddings_can_be_filled_later(tmp_path):
    from yousi.ingest.retrieval import HashEmbedder, Retriever, embed_missing

    vault = Vault.create(tmp_path / "me", "D")
    vault.add_memory(_mem("I live in Seattle.", domain="places"))  # no embedding
    assert len(vault.missing_embeddings()) == 1
    r = Retriever(vault, HashEmbedder())
    assert r.search("Seattle")  # BM25 still finds it
    assert embed_missing(vault, HashEmbedder()) == 1
    assert vault.missing_embeddings() == [] and vault.embeddings()[1].shape == (1, 64)
    assert r.search("Seattle")[0].why in ("both", "words", "meaning")


def test_write_json_refuses_to_escape_vault(tmp_path):
    vault = Vault.create(tmp_path / "me", "D")
    with pytest.raises(ValueError, match="outside the vault"):
        vault.write_json("../escape.json", {})
    assert not (tmp_path / "escape.json").exists()


def test_disclosure_with_braces_does_not_crash():
    from yousi.vault import VaultMeta

    meta = VaultMeta(owner_name="D", disclosure="Twin of {owner} {not a format field}")
    assert meta.disclosure_text() == "Twin of D {not a format field}"
