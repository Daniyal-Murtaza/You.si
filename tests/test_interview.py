"""Tests for the interview engine: gap-driven questions, answers, exit/resume."""

import json
import random

from yousi.ingest.retrieval import HashEmbedder
from yousi.interview import (
    SEED_QUESTIONS,
    InterviewEngine,
    classify_answer,
    pick_domain,
)
from yousi.model import ScriptedChat
from yousi.vault import Memory, Vault


def _answer_json(*texts, domain="routines", kind="habit"):
    return json.dumps({"memories": [{"text": t, "kind": kind, "domain": domain} for t in texts]})


def _engine(tmp_path, script, max_questions=3):
    vault = Vault.create(tmp_path / "v", "Daniyal")
    model = ScriptedChat(script=script)
    # rng seeded so follow-ups (35% chance) never fire unless a test wants them
    return (
        vault,
        model,
        InterviewEngine(
            vault, model, HashEmbedder(), max_questions=max_questions, rng=random.Random(7)
        ),
    )


def test_classify_answer():
    assert classify_answer("skip") == ("skipped", "")
    assert classify_answer("I don't know.") == ("unknown", "")
    assert classify_answer("exit") == ("exit", "")
    assert classify_answer("Around midnight") == ("answered", "Around midnight")


def test_pick_domain_prefers_least_covered(tmp_path):
    vault = Vault.create(tmp_path / "v", "D")
    for d in ("identity", "family", "work"):
        text = f"something about {d}"
        vault.add_memory(
            Memory(
                memory_id=Memory.make_id(text, d),
                text=text,
                kind="fact",
                domain=d,
                source_kind="interview",
                source_ref="x",
                excerpt=text,
            )
        )
    assert pick_domain(vault, rng=random.Random(1)) not in ("identity", "family", "work")


def test_answers_become_memories_with_verbatim_excerpt(tmp_path):
    vault, model, eng = _engine(
        tmp_path, [_answer_json("I usually sleep around midnight.", "I wake at seven.")]
    )
    session = eng.start()
    q = eng.ask(session)
    assert q is not None and q.text in {t for qs in SEED_QUESTIONS.values() for t in qs}
    assert q.status == "pending"
    # Durable before the person answers: a crash here must not lose the question.
    assert eng.load(session.session_id).pending.text == q.text

    status, memories = eng.answer(session, "I sleep around midnight, up at 7.")
    assert status == "answered" and len(memories) == 2
    assert all(m.excerpt == "I sleep around midnight, up at 7." for m in memories)
    assert all(m.source_kind == "interview" and m.source_ref == q.question_id for m in memories)
    assert vault.count() == 2 and len(vault.embeddings()[0]) == 2
    assert eng.load(session.session_id).questions[0].memory_ids == [m.memory_id for m in memories]


def test_skip_and_unknown_store_no_memory_and_are_not_reasked(tmp_path):
    vault, model, eng = _engine(tmp_path, ["unused"], max_questions=5)
    s = eng.start()
    q1 = eng.ask(s)
    assert eng.answer(s, "skip") == ("skipped", [])
    q2 = eng.ask(s)
    assert eng.answer(s, "I don't know") == ("unknown", [])
    assert vault.count() == 0
    texts = {q1.text, q2.text}
    q3 = eng.ask(s)
    assert q3.text not in texts
    assert model.calls == []  # nothing was sent to the model for skipped answers


def test_exit_leaves_question_pending_and_resume_reasks_it(tmp_path):
    vault, model, eng = _engine(
        tmp_path, [_answer_json("I was born in Karachi.", domain="identity", kind="fact")]
    )
    s = eng.start()
    q = eng.ask(s)
    assert eng.answer(s, "exit") == ("exit", [])
    assert eng.load(s.session_id).pending.text == q.text

    # "New process": a fresh engine finds the unfinished session and the same question.
    eng2 = InterviewEngine(vault, model, HashEmbedder(), rng=random.Random(7))
    resumed = eng2.unfinished()
    assert resumed is not None and resumed.session_id == s.session_id
    assert eng2.ask(resumed).text == q.text
    status, mems = eng2.answer(resumed, "Karachi, 1999")
    assert status == "answered" and mems[0].text == "I was born in Karachi."
    eng2.finish(resumed)
    assert eng2.unfinished() is None


def test_question_budget_and_no_repeats_across_sessions(tmp_path):
    vault, model, eng = _engine(tmp_path, ["unused"], max_questions=2)
    s1 = eng.start()
    asked = []
    while (q := eng.ask(s1)) is not None:
        asked.append(q.text)
        eng.answer(s1, "skip")
    assert len(asked) == 2
    eng.finish(s1)
    s2 = eng.start()
    q = eng.ask(s2)
    assert q.text not in asked


def test_follow_up_uses_model_and_builds_on_answer(tmp_path):
    vault = Vault.create(tmp_path / "v", "D")
    model = ScriptedChat(
        script=[
            _answer_json("My father is called Murtaza.", domain="family", kind="relationship"),
            json.dumps({"question": "What did Murtaza teach you?", "reason": "you mentioned him"}),
        ]
    )

    class AlwaysFollowUp(random.Random):
        def random(self):  # force the 35% branch
            return 0.0

    eng = InterviewEngine(vault, model, HashEmbedder(), rng=AlwaysFollowUp(1))
    s = eng.start()
    eng.ask(s)
    eng.answer(s, "My dad is Murtaza, an engineer.")
    q2 = eng.ask(s)
    assert q2.text == "What did Murtaza teach you?" and q2.reason == "you mentioned him"
    assert "They said: My dad is Murtaza" in model.calls[1][-1].content


def test_unstructurable_answer_is_stored_verbatim(tmp_path):
    vault, model, eng = _engine(tmp_path, ["not json", "still not json"])
    s = eng.start()
    eng.ask(s)
    status, mems = eng.answer(s, "Honestly, it depends on the day.")
    assert status == "answered" and len(mems) == 1
    assert mems[0].text == "Honestly, it depends on the day."
