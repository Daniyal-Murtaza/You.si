"""The persona: a confirmed description of who the person is.

"Consciousness" is not a thing software can hold. What it *can* hold, and
test, is a structured description of a person: their values, traits, how
they speak, what they believe, their routines and relationships. This module
infers that description from memories, asks the owner to confirm or correct
every part of it, and keeps the link from each trait back to the memories it
rests on. Unconfirmed traits are marked as such and the twin treats them as
guesses.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import ClassVar, Literal

from pydantic import BaseModel, Field

from yousi.logging_setup import get_logger
from yousi.model import ChatModel, Message, complete_json
from yousi.vault import Memory, Vault

log = get_logger("persona")

Status = Literal["proposed", "confirmed", "rejected", "edited"]


class Trait(BaseModel):
    """One statement about the person, with its evidence and the owner's verdict."""

    text: str
    evidence: list[str] = Field(default_factory=list, description="memory ids")
    status: Status = "proposed"
    owner_note: str = ""


class BigFive(BaseModel):
    """0-100 estimates. Always labelled as estimates; never shown as fact."""

    openness: int = Field(50, ge=0, le=100)
    conscientiousness: int = Field(50, ge=0, le=100)
    extraversion: int = Field(50, ge=0, le=100)
    agreeableness: int = Field(50, ge=0, le=100)
    neuroticism: int = Field(50, ge=0, le=100)
    rationale: str = ""


class Persona(BaseModel):
    owner_name: str
    updated_at: str = ""
    summary: str = Field("", description="Two or three sentences: who this person is.")
    values: list[Trait] = Field(default_factory=list)
    traits: list[Trait] = Field(default_factory=list)
    speech_style: list[Trait] = Field(
        default_factory=list, description="How they write/talk: rhythm, humour, words"
    )
    beliefs: list[Trait] = Field(default_factory=list)
    routines: list[Trait] = Field(default_factory=list)
    relationships: list[Trait] = Field(default_factory=list)
    goals: list[Trait] = Field(default_factory=list)
    fears: list[Trait] = Field(default_factory=list)
    big_five: BigFive = Field(default_factory=BigFive)

    SECTIONS: ClassVar[tuple[str, ...]] = (
        "values",
        "traits",
        "speech_style",
        "beliefs",
        "routines",
        "relationships",
        "goals",
        "fears",
    )

    def all_traits(self) -> list[tuple[str, Trait]]:
        return [(s, t) for s in self.SECTIONS for t in getattr(self, s)]

    def confirmed(self) -> list[tuple[str, Trait]]:
        return [(s, t) for s, t in self.all_traits() if t.status in ("confirmed", "edited")]

    def pending(self) -> list[tuple[str, Trait]]:
        return [(s, t) for s, t in self.all_traits() if t.status == "proposed"]

    def confirmation_rate(self) -> float:
        total = len(self.all_traits())
        return len(self.confirmed()) / total if total else 0.0

    def to_prompt(self, *, include_unconfirmed: bool = False) -> str:
        """Render for the twin's system prompt. Confirmed first; guesses labelled."""
        lines = [f"Who {self.owner_name} is: {self.summary}".strip()]
        for section in self.SECTIONS:
            items = [
                t
                for t in getattr(self, section)
                if t.status in ("confirmed", "edited")
                or (include_unconfirmed and t.status == "proposed")
            ]
            if not items:
                continue
            lines.append(f"\n{section.replace('_', ' ').title()}:")
            for t in items:
                flag = "" if t.status in ("confirmed", "edited") else " (unconfirmed guess)"
                lines.append(f"- {t.text}{flag}")
        bf = self.big_five
        lines.append(
            f"\nPersonality estimate (0-100, uncertain): openness {bf.openness}, "
            f"conscientiousness {bf.conscientiousness}, extraversion {bf.extraversion}, "
            f"agreeableness {bf.agreeableness}, neuroticism {bf.neuroticism}."
        )
        return "\n".join(lines)


# --------------------------------------------------------------------------
# inference
# --------------------------------------------------------------------------


class _Proposed(BaseModel):
    summary: str = ""
    values: list[Trait] = Field(default_factory=list)
    traits: list[Trait] = Field(default_factory=list)
    speech_style: list[Trait] = Field(default_factory=list)
    beliefs: list[Trait] = Field(default_factory=list)
    routines: list[Trait] = Field(default_factory=list)
    relationships: list[Trait] = Field(default_factory=list)
    goals: list[Trait] = Field(default_factory=list)
    fears: list[Trait] = Field(default_factory=list)
    big_five: BigFive = Field(default_factory=BigFive)


INFER_SYSTEM = """\
You study a person's own recorded memories and describe who they are.
Rules:
- Every statement must be supported by the memories. Put the supporting memory ids in
  `evidence`. A statement with no evidence ids will be discarded.
- Write statements in third person about the person ("Values loyalty above comfort").
- speech_style describes HOW they write and talk: sentence length, humour, formality,
  favourite words, languages mixed, how they show affection or anger. Quote from memories
  where you can.
- Be specific and modest. 3-8 items per section; leave a section empty if the memories
  do not support it. Big Five numbers are rough estimates; explain them in `rationale`.
- Do not moralise, flatter, or diagnose.
"""


def _render_memories(memories: list[Memory], *, limit: int = 400) -> str:
    lines = []
    for m in memories[:limit]:
        tag = f"{m.domain}/{m.kind}"
        conf = "" if m.confidence >= 0.9 else f" conf={m.confidence}"
        lines.append(f"[{m.memory_id}] ({tag}{conf}) {m.text}")
    return "\n".join(lines)


def infer_persona(model: ChatModel, vault: Vault, *, previous: Persona | None = None) -> Persona:
    """Propose a persona from the vault. Keeps the owner's earlier verdicts."""
    meta = vault.load_meta()
    memories = vault.all_memories()
    if not memories:
        return previous or Persona(owner_name=meta.owner_name)
    valid_ids = {m.memory_id for m in memories}
    prompt = (
        f"The person is {meta.owner_name}. Memories:\n\n{_render_memories(memories)}\n\n"
        "Describe who they are."
    )
    proposed = complete_json(
        model,
        [Message("system", INFER_SYSTEM), Message("user", prompt)],
        _Proposed,
        max_tokens=2500,
    )

    persona = Persona(
        owner_name=meta.owner_name,
        summary=proposed.summary,
        big_five=proposed.big_five,
        updated_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    dropped = 0
    for section in Persona.SECTIONS:
        kept: list[Trait] = []
        for t in getattr(proposed, section):
            t.evidence = [e for e in t.evidence if e in valid_ids]
            if not t.evidence:
                dropped += 1
                continue
            t.status = "proposed"
            kept.append(t)
        setattr(persona, section, kept)
    if dropped:
        log.info("dropped %d unsupported trait(s)", dropped)
    if previous is not None:
        _carry_verdicts(previous, persona)
    return persona


def _carry_verdicts(previous: Persona, new: Persona) -> None:
    """Keep what the owner already confirmed/edited/rejected across re-inference."""
    for section in Persona.SECTIONS:
        old_by_text = {t.text.strip().lower(): t for t in getattr(previous, section)}
        merged: list[Trait] = []
        seen: set[str] = set()
        for t in getattr(new, section):
            key = t.text.strip().lower()
            old = old_by_text.get(key)
            if old is not None and old.status != "proposed":
                t.status, t.owner_note = old.status, old.owner_note
            if old is not None and old.status == "rejected":
                continue  # the owner said no; do not propose it again
            merged.append(t)
            seen.add(key)
        # Confirmed/edited traits the model forgot still belong to the person.
        for key, old in old_by_text.items():
            if key not in seen and old.status in ("confirmed", "edited"):
                merged.append(old)
        setattr(new, section, merged)


# --------------------------------------------------------------------------
# persistence and confirmation
# --------------------------------------------------------------------------


def load_persona(vault: Vault) -> Persona | None:
    raw = vault.read_json("persona.json")
    return Persona.model_validate(raw) if raw else None


def save_persona(vault: Vault, persona: Persona) -> None:
    persona.updated_at = datetime.now(UTC).isoformat(timespec="seconds")
    vault.write_json("persona.json", persona.model_dump())


def apply_verdict(
    persona: Persona,
    section: str,
    index: int,
    verdict: Status,
    *,
    edited_text: str = "",
    note: str = "",
) -> Trait:
    """Record the owner's decision on one trait. Edits keep the original evidence."""
    trait: Trait = getattr(persona, section)[index]
    if verdict == "edited":
        if not edited_text.strip():
            raise ValueError("edited verdict needs new text")
        trait.text = edited_text.strip()
    trait.status = verdict
    trait.owner_note = note
    return trait
