"""The interview: the twin asks, the person answers, answers become memories.

Gap-driven: the engine looks at which life domains have the least coverage
in the vault and asks there first, using a seed bank of questions plus model-
generated follow-ups that build on what was just said. The person can answer,
``skip``, say ``I don't know``, or ``exit`` and resume later; the session file
records every question and verdict so nothing is asked twice.

Everything the person says is stored verbatim as the ``excerpt``; the model
only rephrases it into clean first-person memories and may never add facts.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from yousi.ingest.retrieval import Embedder
from yousi.logging_setup import get_logger
from yousi.model import ChatModel, Message, ModelError, complete_json
from yousi.vault import DOMAINS, Memory, Vault

log = get_logger("interview")

AnswerStatus = Literal["answered", "skipped", "unknown"]
SKIP_WORDS = {"skip", "s", "pass", "next"}
UNKNOWN_WORDS = {
    "i don't know",
    "i dont know",
    "dont know",
    "don't know",
    "idk",
    "unknown",
    "not sure",
    "no idea",
    "?",
}
EXIT_WORDS = {"exit", "quit", "q", "stop", "bye"}

# Seed questions per domain. Short, concrete, answerable in a sentence or two.
SEED_QUESTIONS: dict[str, list[str]] = {
    "identity": [
        "What is your full name, and is there a name people actually call you?",
        "When and where were you born?",
        "How would you describe yourself in three words?",
    ],
    "family": [
        "Tell me about your father: his name, what he does, what he is like.",
        "Tell me about your mother.",
        "Do you have siblings? What are they like?",
    ],
    "relationships": [
        "Who are the two or three people you trust most, and why them?",
        "Is there someone you have lost touch with and still think about?",
    ],
    "work": [
        "What do you do for work, and how did you end up there?",
        "What part of your work do you actually enjoy?",
    ],
    "health": [
        "How is your health? Anything you manage day to day?",
        "What do you do when you are stressed?",
    ],
    "routines": [
        "When do you normally sleep and wake up?",
        "Walk me through an ordinary weekday morning.",
    ],
    "beliefs": [
        "What do you believe that most people around you do not?",
        "Are you religious or spiritual? How does that show up in your life?",
    ],
    "values": [
        "What would you never do, even if it paid well?",
        "What is the quality you admire most in other people?",
    ],
    "goals": [
        "What are you aiming for in life right now?",
        "Where do you want to be in ten years?",
    ],
    "fears": ["What are you afraid of?", "What is a regret you carry?"],
    "humour": [
        "What makes you laugh? Give me an example of a joke you would make.",
        "Who do you tease, and how?",
    ],
    "tastes": ["What food could you eat every day?", "What music do you return to?"],
    "money": [
        "How do you think about money: saver, spender, somewhere in between?",
        "What is the best thing you ever spent money on?",
    ],
    "places": [
        "Where do you live now, and what do you like about it?",
        "Which place feels most like home?",
    ],
    "stories": [
        "Tell me a story from your childhood that you still think about.",
        "What is the hardest thing you have been through?",
    ],
    "communication": [
        "How do you usually show someone you care about them?",
        "How do you argue? What do you do when you are angry?",
    ],
}


@dataclass
class Question:
    question_id: str
    domain: str
    text: str
    reason: str = ""
    status: AnswerStatus | Literal["pending"] = "pending"
    answer: str = ""
    asked_at: str = ""
    answered_at: str = ""
    memory_ids: list[str] = field(default_factory=list)

    @staticmethod
    def make_id(domain: str, text: str) -> str:
        return "q" + hashlib.sha1(f"{domain}\n{text}".encode()).hexdigest()[:10]


@dataclass
class Session:
    session_id: str
    started_at: str
    questions: list[Question] = field(default_factory=list)
    finished: bool = False

    @property
    def pending(self) -> Question | None:
        return next((q for q in self.questions if q.status == "pending"), None)

    @property
    def asked(self) -> int:
        return len([q for q in self.questions if q.status != "pending"])


def classify_answer(raw: str) -> tuple[AnswerStatus | Literal["exit"], str]:
    text = raw.strip()
    low = text.lower().rstrip(".!")
    if low in EXIT_WORDS:
        return "exit", ""
    if low in SKIP_WORDS:
        return "skipped", ""
    if low in UNKNOWN_WORDS:
        return "unknown", ""
    return "answered", text


# --------------------------------------------------------------------------
# answer -> memories
# --------------------------------------------------------------------------


class _AnswerMemory(BaseModel):
    text: str
    kind: Literal[
        "fact",
        "preference",
        "habit",
        "relationship",
        "event",
        "belief",
        "goal",
        "feeling",
        "story",
        "style",
    ] = "fact"
    domain: str
    happened_at: str = ""
    tags: list[str] = Field(default_factory=list)


class _AnswerMemories(BaseModel):
    memories: list[_AnswerMemory] = Field(default_factory=list)


ANSWER_SYSTEM = """\
A person answered a question about themselves. Turn the answer into 1-3 memory sentences.
Rules:
- FIRST PERSON ONLY, as the person would say it: "I ...", "My ...". Never "Daniyal is ...",
  never "they", never commentary like "showing a clear preference".
- Each memory must stand alone: name the subject explicitly ("My sister Hira is the funny
  one", never "She is the funny one"). Resolve every pronoun.
- Record what the answer SAYS, not what the question asked. If asked about fears and they
  talk about their knee, the memory is about the knee ("I have a bad knee from football"),
  not "My fear is my knee". Never invent a link to the question's topic.
- Prefer one richer sentence over fragments. Only split for clearly separate facts.
- Use only what they said; keep distinctive wording; never add facts, dates or names.
- If the answer reveals how they talk (humour, bluntness, a turn of phrase), add ONE memory
  of kind "style" that quotes the phrase, e.g. I say things like "saver by instinct, spender
  on tools".
Domain must be one of: %s
"""


def answer_to_memories(
    model: ChatModel, question: Question, answer: str, *, owner_name: str
) -> list[Memory]:
    prompt = (
        f"Person: {owner_name}\nQuestion ({question.domain}): {question.text}\nAnswer: {answer}"
    )
    try:
        result = complete_json(
            model,
            [Message("system", ANSWER_SYSTEM % ", ".join(DOMAINS)), Message("user", prompt)],
            _AnswerMemories,
        )
        proposed = result.memories
    except ModelError as exc:
        log.warning("could not structure answer, storing it verbatim: %s", exc)
        proposed = []
    if not proposed:
        proposed = [_AnswerMemory(text=answer, domain=question.domain)]
    memories = []
    for p in proposed:
        domain = p.domain if p.domain in DOMAINS else question.domain
        memories.append(
            Memory(
                memory_id=Memory.make_id(p.text, question.question_id),
                text=p.text.strip(),
                kind=p.kind,
                domain=domain,
                source_kind="interview",
                source_ref=question.question_id,
                excerpt=answer,
                confidence=1.0,
                happened_at=p.happened_at,
                tags=[t.lower() for t in p.tags][:6],
            )
        )
    return memories


# --------------------------------------------------------------------------
# question selection
# --------------------------------------------------------------------------


class _FollowUp(BaseModel):
    question: str
    reason: str = ""


FOLLOWUP_SYSTEM = """\
You are interviewing a person to understand them deeply, so that a twin of them can speak as
they would. Given what they just said, ask ONE natural follow-up a close friend would ask about
what they ACTUALLY said - not about the original question's topic if their answer went
elsewhere. Short, warm, specific, no preamble. Do not ask anything already known. If the
answer was thin, off-topic, or there is nothing worth asking, return an empty question.
"""


def _asked_texts(vault: Vault) -> set[str]:
    out: set[str] = set()
    for path in (vault.root / "sessions").glob("interview-*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        for q in data.get("questions", []):
            if q.get("status") != "pending":
                out.add(q["text"].strip().lower())
    return out


def pick_domain(vault: Vault, *, rng: random.Random | None = None) -> str:
    """Least-covered domain, with a little randomness so sessions feel natural."""
    rng = rng or random.Random()
    counts = vault.domain_counts()
    ordered = sorted(DOMAINS, key=lambda d: (counts.get(d, 0), rng.random()))
    return ordered[0]


def next_question(
    vault: Vault, model: ChatModel, session: Session, *, rng: random.Random | None = None
) -> Question | None:
    rng = rng or random.Random()
    asked = _asked_texts(vault) | {q.text.strip().lower() for q in session.questions}

    # Follow up on the last answer sometimes - but only if the LAST question was a seed
    # question that was actually answered. Never chain follow-up on follow-up, and never
    # follow up on a skip or "I don't know": that is how interviews turn into interrogations.
    last = session.questions[-1] if session.questions else None
    can_follow = (
        last is not None
        and last.status == "answered"
        and not last.reason.startswith("follow-up")
        and rng.random() < 0.35
    )
    if can_follow:
        known = "\n".join(f"- {m.text}" for m in vault.by_domain(last.domain)[-12:])
        prompt = (
            f"Domain: {last.domain}\nThey were asked: {last.text}\nThey said: {last.answer}\n"
            f"Already known in this domain:\n{known or '- nothing yet'}"
        )
        try:
            fu = complete_json(
                model, [Message("system", FOLLOWUP_SYSTEM), Message("user", prompt)], _FollowUp
            )
            if fu.question.strip() and fu.question.strip().lower() not in asked:
                return Question(
                    question_id=Question.make_id(last.domain, fu.question),
                    domain=last.domain,
                    text=fu.question.strip(),
                    reason="follow-up: " + (fu.reason or "on what you just said"),
                )
        except ModelError as exc:
            log.warning("follow-up generation failed: %s", exc)

    # Otherwise, the least covered domain that still has an unasked seed question.
    counts = vault.domain_counts()
    for domain in sorted(DOMAINS, key=lambda d: (counts.get(d, 0), rng.random())):
        for text in SEED_QUESTIONS.get(domain, []):
            if text.strip().lower() not in asked:
                return Question(
                    question_id=Question.make_id(domain, text),
                    domain=domain,
                    text=text,
                    reason=f"I know little about your {domain} yet",
                )
    return None


# --------------------------------------------------------------------------
# engine
# --------------------------------------------------------------------------


class InterviewEngine:
    def __init__(
        self,
        vault: Vault,
        model: ChatModel,
        embedder: Embedder,
        *,
        max_questions: int = 8,
        rng: random.Random | None = None,
    ) -> None:
        self.vault = vault
        self.model = model
        self.embedder = embedder
        self.max_questions = max_questions
        self.rng = rng or random.Random()

    # -- sessions ------------------------------------------------------------

    def _path(self, session_id: str) -> Path:
        return self.vault.root / "sessions" / f"interview-{session_id}.json"

    def save(self, session: Session) -> None:
        self.vault.write_json(f"sessions/interview-{session.session_id}.json", asdict(session))

    def load(self, session_id: str) -> Session:
        data = json.loads(self._path(session_id).read_text(encoding="utf-8"))
        data["questions"] = [Question(**q) for q in data["questions"]]
        return Session(**data)

    def unfinished(self) -> Session | None:
        paths = sorted((self.vault.root / "sessions").glob("interview-*.json"))
        for path in reversed(paths):
            session = self.load(path.stem.removeprefix("interview-"))
            if not session.finished:
                return session
        return None

    def start(self) -> Session:
        sid = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        session = Session(
            session_id=sid, started_at=datetime.now(UTC).isoformat(timespec="seconds")
        )
        self.save(session)
        return session

    # -- the loop --------------------------------------------------------------

    def ask(self, session: Session) -> Question | None:
        """Return the pending question, or pick a new one. None when done."""
        if session.pending is not None:
            return session.pending
        if session.asked >= self.max_questions:
            return None
        q = next_question(self.vault, self.model, session, rng=self.rng)
        if q is None:
            return None
        q.asked_at = datetime.now(UTC).isoformat(timespec="seconds")
        session.questions.append(q)
        self.save(session)  # the question is durable before the person sees it
        return q

    def answer(self, session: Session, raw: str) -> tuple[str, list[Memory]]:
        """Commit an answer. Returns (status, memories added). 'exit' leaves it pending."""
        q = session.pending
        if q is None:
            raise RuntimeError("no pending question")
        status, text = classify_answer(raw)
        if status == "exit":
            return "exit", []
        q.status, q.answer = status, text
        q.answered_at = datetime.now(UTC).isoformat(timespec="seconds")
        memories: list[Memory] = []
        if status == "answered":
            meta = self.vault.load_meta()
            memories = answer_to_memories(self.model, q, text, owner_name=meta.owner_name)
            vecs = self.embedder.encode([m.text for m in memories])
            self.vault.add_memories(list(zip(memories, vecs, strict=True)))
            q.memory_ids = [m.memory_id for m in memories]
            self.vault.bump("answers")
        self.save(session)
        log.info("question %s -> %s (%d memories)", q.question_id, status, len(memories))
        return status, memories

    def finish(self, session: Session) -> Session:
        if session.pending is not None:
            session.questions.remove(session.pending)  # never leave a dangling question
        session.finished = True
        self.save(session)
        return session
