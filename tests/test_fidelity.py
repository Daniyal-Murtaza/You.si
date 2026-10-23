"""Tests for the fidelity test and the scenario sandbox."""

import json
import random

import pytest

from yousi.fidelity import FidelityTest, Sandbox
from yousi.ingest.retrieval import HashEmbedder
from yousi.interview import SEED_QUESTIONS
from yousi.model import ScriptedChat
from yousi.vault import Memory, Vault


def _vault(tmp_path):
    vault = Vault.create(tmp_path / "v", "Daniyal")
    emb = HashEmbedder()
    text = "I usually sleep around midnight."
    m = Memory(
        memory_id=Memory.make_id(text, "t"),
        text=text,
        kind="habit",
        domain="routines",
        source_kind="interview",
        source_ref="t",
        excerpt=text,
    )
    vault.add_memory(m, emb.encode([text])[0])
    return vault, emb, m.memory_id


def test_fidelity_round_hides_ids_and_scores_judge(tmp_path):
    vault, emb, mid = _vault(tmp_path)
    model = ScriptedChat(script=[f"Midnight, give or take [{mid}, m000000000000]."])
    ft = FidelityTest(vault, model, emb, rng=random.Random(3))
    assert ft.pick_question() in {q for qs in SEED_QUESTIONS.values() for q in qs}
    rnd = ft.new_round("When do you usually sleep?", "Usually about 12, sometimes later.")
    assert rnd.twin_grounded and rnd.twin_cited == [mid]
    assert "[m" not in rnd.twin_answer  # judge never sees citations
    a, b = FidelityTest.pair(rnd)
    assert {a, b} == {rnd.twin_answer, rnd.owner_answer}
    ft.judge(rnd, rnd.twin_shown_as)  # judge picks the twin as the real person
    assert rnd.fooled is True
    rnd2 = ft.new_round("What food could you eat every day?", "Biryani.")
    ft.judge(rnd2, "A" if rnd2.twin_shown_as == "B" else "B")
    assert rnd2.fooled is False
    assert ft.report.fool_rate == 0.5
    assert "50%" in ft.report.summary()
    saved = list((vault.root / "sessions").glob("fidelity-*.json"))
    assert saved and len(json.loads(saved[0].read_text())["rounds"]) == 2
    with pytest.raises(ValueError):
        ft.judge(rnd, "C")


def test_fidelity_ungrounded_answer_is_recorded_as_such(tmp_path):
    vault, emb, mid = _vault(tmp_path)
    ft = FidelityTest(vault, ScriptedChat(script=["No idea, honestly [m000000000000]."]), emb)
    rnd = ft.new_round("What is your favourite opera?", "Don't have one.")
    assert rnd.twin_grounded is False and rnd.twin_cited == []
    assert "m000000000000" not in rnd.twin_answer


def test_owner_answers_become_memories_once(tmp_path):
    vault, emb, mid = _vault(tmp_path)
    ft = FidelityTest(vault, ScriptedChat(script=["x"]), emb)
    ft.new_round("Q1?", "My honest answer one.")
    ft.new_round("Q2?", "")
    assert ft.owner_answers_as_memories() == 1
    assert ft.owner_answers_as_memories() == 0
    assert vault.count() == 2


def test_sandbox_prediction_is_labelled_and_confidence_capped(tmp_path):
    vault, emb, mid = _vault(tmp_path)
    good = json.dumps(
        {
            "reaction": "I'd say no and go to bed.",
            "reasoning": "sleep habit",
            "evidence": [mid],
            "confidence": 0.7,
            "what_would_change_it": "a crisis",
        }
    )
    sb = Sandbox(vault, ScriptedChat(script=[good]), emb)
    r = sb.predict("A friend calls around midnight asking you to come out instead of sleep.")
    assert r.label.startswith("PREDICTION")
    # One strongly relevant memory caps confidence at 0.5 however sure the model sounds.
    assert r.prediction.confidence == 0.5 and [m.memory_id for m in r.cited] == [mid]
    log_lines = (vault.root / "sessions" / "scenarios.jsonl").read_text().splitlines()
    assert len(log_lines) == 1 and "PREDICTION" in log_lines[0]

    bogus = json.dumps(
        {
            "reaction": "I'd buy a boat.",
            "reasoning": "vibes",
            "evidence": ["m000000000000"],
            "confidence": 0.95,
        }
    )
    r2 = Sandbox(vault, ScriptedChat(script=[bogus]), emb).predict("You win the lottery.")
    assert r2.prediction.evidence == [] and r2.prediction.confidence == 0.3


def test_sandbox_confirmation_becomes_memory(tmp_path):
    vault, emb, mid = _vault(tmp_path)
    sb = Sandbox(vault, ScriptedChat(script=["garbage", "garbage"]), emb)
    r = sb.predict("Someone cuts in front of you in a queue.")
    assert r.prediction.confidence == 0.0  # model failed; honest fallback
    m = sb.confirm(r, "I'd say something, politely, once.")
    assert m.source_kind == "interview" and "Scenario:" in m.excerpt
    assert vault.get(m.memory_id) is not None
