"""Fidelity test and scenario sandbox.

**Fidelity test** - the only honest answer to "how much like me is it" is a
measurement. The owner answers a question; the twin answers the same question
from memory; a judge (the owner, or a friend who knows them) sees both answers
in random order and picks which one is the real person. Over many rounds the
twin's score is how often the judge could *not* tell (50% = indistinguishable).
Each round also records whether the twin's answer was grounded, so a twin that
bluffs well is still visible in the data.

**Scenario sandbox** - "how would I react if ...". The twin predicts, and the
output is labelled a prediction with the memories it rests on and a confidence
that is low when little evidence applies. The owner can then confirm or
correct, and the correction becomes a memory. Nothing from the sandbox is ever
presented as something the person actually did or said.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime

from pydantic import BaseModel, Field

from yousi.ingest.retrieval import Embedder, Retriever
from yousi.interview import SEED_QUESTIONS
from yousi.logging_setup import get_logger
from yousi.model import ChatModel, Message, ModelError, complete_json
from yousi.persona import Persona, load_persona
from yousi.twin import MIN_RELEVANCE
from yousi.vault import Memory, Vault

log = get_logger("fidelity")


# --------------------------------------------------------------------------
# fidelity test
# --------------------------------------------------------------------------


@dataclass
class Round:
    question: str
    owner_answer: str
    twin_answer: str
    twin_cited: list[str]
    twin_grounded: bool
    twin_shown_as: str  # "A" or "B"
    judge_pick: str = ""  # "A" or "B"
    fooled: bool | None = None  # judge picked the twin as the real one


@dataclass
class FidelityReport:
    started_at: str
    rounds: list[Round] = field(default_factory=list)

    @property
    def judged(self) -> list[Round]:
        return [r for r in self.rounds if r.fooled is not None]

    @property
    def fool_rate(self) -> float:
        j = self.judged
        return sum(r.fooled for r in j) / len(j) if j else 0.0

    @property
    def grounded_rate(self) -> float:
        return sum(r.twin_grounded for r in self.rounds) / len(self.rounds) if self.rounds else 0.0

    def summary(self) -> str:
        n = len(self.judged)
        if not n:
            return "no judged rounds yet"
        return (
            f"{n} rounds: judge mistook the twin for the real person "
            f"{self.fool_rate:.0%} of the time (50% = indistinguishable); "
            f"{self.grounded_rate:.0%} of twin answers were grounded in memories."
        )


TWIN_ANSWER_SYSTEM = """\
You are the recorded twin of {owner}. Answer the question in one to three sentences exactly as
{owner} would write it - their rhythm, their words, their humour - using ONLY these memories and
this description of them. Cite memory ids in square brackets after the facts you use. If the
memories do not cover it, answer in their voice that you do not know / are not sure, and cite
nothing. Do not mention being a twin.

WHO THEY ARE:
{persona}

MEMORIES:
{memories}
"""


class FidelityTest:
    def __init__(
        self,
        vault: Vault,
        model: ChatModel,
        embedder: Embedder,
        *,
        rng: random.Random | None = None,
    ) -> None:
        self.vault = vault
        self.model = model
        self.embedder = embedder
        self.retriever = Retriever(vault, embedder)
        self.rng = rng or random.Random()
        self.report = FidelityReport(started_at=datetime.now(UTC).isoformat(timespec="seconds"))

    def pick_question(self, *, exclude: set[str] = frozenset()) -> str:
        pool = [q for qs in SEED_QUESTIONS.values() for q in qs if q not in exclude]
        return self.rng.choice(pool)

    def twin_answer(self, question: str) -> tuple[str, list[str], bool]:
        meta = self.vault.load_meta()
        persona = load_persona(self.vault) or Persona(owner_name=meta.owner_name)
        hits = [h for h in self.retriever.search(question, top_k=8) if h.score >= MIN_RELEVANCE]
        allowed = {h.memory.memory_id for h in hits}
        memories = "\n".join(f"[{h.memory.memory_id}] {h.memory.text}" for h in hits) or "(none)"
        system = TWIN_ANSWER_SYSTEM.format(
            owner=meta.owner_name,
            persona=persona.to_prompt(include_unconfirmed=True),
            memories=memories,
        )
        text = self.model.chat(
            [Message("system", system), Message("user", question)], temperature=0.6, max_tokens=200
        ).text
        import re

        cited = [c for c in dict.fromkeys(re.findall(r"\[(m[0-9a-f]{12})\]", text)) if c in allowed]
        clean = re.sub(r"\s*\[m[0-9a-f]{12}\]", "", text).strip()  # judge must not see ids
        return clean, cited, bool(cited)

    def new_round(self, question: str, owner_answer: str) -> Round:
        twin_text, cited, grounded = self.twin_answer(question)
        shown_as = self.rng.choice(["A", "B"])
        rnd = Round(
            question=question,
            owner_answer=owner_answer,
            twin_answer=twin_text,
            twin_cited=cited,
            twin_grounded=grounded,
            twin_shown_as=shown_as,
        )
        self.report.rounds.append(rnd)
        return rnd

    @staticmethod
    def pair(rnd: Round) -> tuple[str, str]:
        """(answer shown as A, answer shown as B)."""
        return (
            (rnd.twin_answer, rnd.owner_answer)
            if rnd.twin_shown_as == "A"
            else (rnd.owner_answer, rnd.twin_answer)
        )

    def judge(self, rnd: Round, pick: str) -> Round:
        """The judge says which answer (A/B) they believe is the real person."""
        pick = pick.strip().upper()
        if pick not in ("A", "B"):
            raise ValueError("pick must be A or B")
        rnd.judge_pick = pick
        rnd.fooled = pick == rnd.twin_shown_as
        self.save()
        return rnd

    def save(self) -> None:
        self.vault.write_json(
            f"sessions/fidelity-{self.report.started_at[:19].replace(':', '')}.json",
            asdict(self.report),
        )

    def owner_answers_as_memories(self) -> int:
        """The owner's real answers are valuable data; store them as memories."""
        n = 0
        for rnd in self.report.rounds:
            if not rnd.owner_answer.strip():
                continue
            ref = f"fidelity-{self.report.started_at}"
            m = Memory(
                memory_id=Memory.make_id(rnd.owner_answer, ref + rnd.question),
                text=rnd.owner_answer.strip(),
                kind="story",
                domain="stories",
                source_kind="interview",
                source_ref=ref,
                excerpt=rnd.owner_answer,
                tags=["fidelity-test"],
            )
            if self.vault.get(m.memory_id) is None:
                self.vault.add_memory(m, self.embedder.encode([m.text])[0])
                n += 1
        return n


# --------------------------------------------------------------------------
# scenario sandbox
# --------------------------------------------------------------------------


class Prediction(BaseModel):
    reaction: str = Field(description="What the person would most likely do/say, first person.")
    reasoning: str = Field(description="Which memories/traits drive this and how.")
    evidence: list[str] = Field(default_factory=list, description="memory ids used")
    confidence: float = Field(ge=0.0, le=1.0)
    what_would_change_it: str = Field("", description="Info that would flip this prediction.")


SCENARIO_SYSTEM = """\
You are predicting how {owner} would react to a hypothetical, using ONLY their memories and the
description of them below. This is a PREDICTION, not a memory. Rules:
- reaction: first person, in their voice, 1-4 sentences.
- evidence: the memory ids you actually relied on. No ids -> confidence must be <= 0.3.
- confidence: honest. Many directly relevant memories -> higher; thin or indirect -> low.
- Never state as fact that they did or said something; this is what they WOULD likely do.

WHO THEY ARE:
{persona}

MEMORIES:
{memories}
"""


@dataclass
class ScenarioResult:
    scenario: str
    prediction: Prediction
    cited: list[Memory]
    label: str = "PREDICTION - not something the person actually did or said"
    at: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))


class Sandbox:
    def __init__(self, vault: Vault, model: ChatModel, embedder: Embedder) -> None:
        self.vault = vault
        self.model = model
        self.embedder = embedder
        self.retriever = Retriever(vault, embedder)

    def predict(self, scenario: str) -> ScenarioResult:
        meta = self.vault.load_meta()
        persona = load_persona(self.vault) or Persona(owner_name=meta.owner_name)
        hits = [h for h in self.retriever.search(scenario, top_k=10) if h.score >= MIN_RELEVANCE]
        allowed = {h.memory.memory_id for h in hits}
        memories = (
            "\n".join(f"[{h.memory.memory_id}] ({h.memory.domain}) {h.memory.text}" for h in hits)
            or "(none)"
        )
        system = SCENARIO_SYSTEM.format(
            owner=meta.owner_name,
            persona=persona.to_prompt(include_unconfirmed=True),
            memories=memories,
        )
        try:
            pred = complete_json(
                self.model,
                [Message("system", system), Message("user", f"Scenario: {scenario}")],
                Prediction,
                temperature=0.5,
            )
        except ModelError as exc:
            log.warning("prediction failed: %s", exc)
            pred = Prediction(
                reaction="I can't predict that yet; I know too little about this.",
                reasoning=str(exc)[:200],
                evidence=[],
                confidence=0.0,
            )
        pred.evidence = [e for e in pred.evidence if e in allowed]
        if not pred.evidence:
            pred.confidence = min(pred.confidence, 0.3)
        cited = [m for m in (self.vault.get(i) for i in pred.evidence) if m is not None]
        result = ScenarioResult(scenario=scenario, prediction=pred, cited=cited)
        self._log(result)
        return result

    def confirm(self, result: ScenarioResult, owner_reaction: str) -> Memory:
        """The owner says what they would really do; that becomes a memory."""
        ref = f"sandbox-{result.at}"
        text = owner_reaction.strip()
        m = Memory(
            memory_id=Memory.make_id(text, ref),
            text=text,
            kind="belief",
            domain="values",
            source_kind="interview",
            source_ref=ref,
            excerpt=f"Scenario: {result.scenario}\nOwner: {owner_reaction}",
            tags=["scenario"],
        )
        self.vault.add_memory(m, self.embedder.encode([m.text])[0])
        return m

    def _log(self, result: ScenarioResult) -> None:
        path = self.vault.root / "sessions" / "scenarios.jsonl"
        path.parent.mkdir(exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    {
                        "at": result.at,
                        "scenario": result.scenario,
                        "prediction": result.prediction.model_dump(),
                        "label": result.label,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
