"""Memory extraction: from raw text units to grounded memories.

The model reads a batch of the person's own text and proposes memories.
Code then enforces the rules the model cannot be trusted with:

* every memory must quote an exact ``excerpt`` that appears in the batch
  (so the twin can never "remember" something that was not there)
* only text the person wrote themselves (``speaker == "owner"``) can become a
  first-person memory; other people's words become relationship context at
  lower confidence
* duplicates (same text, same source) are skipped
* each memory is embedded locally and stored with provenance
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from yousi.ingest.parsers import Unit, UnsupportedFileError, parse_file
from yousi.ingest.retrieval import Embedder
from yousi.logging_setup import get_logger
from yousi.model import ChatModel, Message, ModelError, complete_json
from yousi.vault import DOMAINS, Memory, Vault

log = get_logger("ingest.extract")

BATCH_WORDS = 1200  # roughly 1.6k tokens of source text per model call


class ProposedMemory(BaseModel):
    text: str = Field(description="One clear sentence about the person, first person.")
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
    ]
    domain: str
    excerpt: str = Field(description="Exact substring of the source that supports this.")
    unit_ref: str = Field(description="The [ref] tag of the unit the excerpt came from.")
    happened_at: str = ""
    confidence: float = Field(ge=0.0, le=1.0, default=0.9)
    tags: list[str] = Field(default_factory=list)


class ExtractionResult(BaseModel):
    memories: list[ProposedMemory] = Field(default_factory=list)


EXTRACT_SYSTEM = """\
You read a person's own writing and messages and extract what they reveal about that person.
Rules:
- Write each memory as ONE clear first-person sentence ("I ...", "My ...").
- Only state what the text supports. Never guess, infer biography, or fill gaps.
- `excerpt` must be an EXACT substring copied from the source. If you cannot quote it, do not
  include the memory.
- Prefer durable things: identity, family, relationships, work, health, routines, beliefs,
  values, goals, fears, humour, tastes, money, places, stories, and how the person
  writes/speaks (kind "style": e.g. "I use short sentences and dry humour").
- Skip trivia with no lasting meaning about the person (logistics, one-off scheduling).
- Text marked speaker=other is someone else talking TO the person: use it only for what it
  reveals about the relationship, at confidence <= 0.6.
- Domain must be one of: %s
Return 0-12 memories per batch; quality over quantity.
"""


@dataclass
class IngestReport:
    files_seen: int = 0
    files_skipped: int = 0
    units: int = 0
    memories_added: int = 0
    memories_rejected: int = 0
    errors: list[str] = field(default_factory=list)


def _batches(units: list[Unit]) -> list[list[Unit]]:
    batches, current, words = [], [], 0
    for u in units:
        w = len(u.text.split())
        if current and words + w > BATCH_WORDS:
            batches.append(current)
            current, words = [], 0
        current.append(u)
        words += w
    if current:
        batches.append(current)
    return batches


def _render(units: list[Unit]) -> str:
    lines = []
    for u in units:
        who = "owner" if u.speaker == "owner" else f"other:{u.speaker or 'unknown'}"
        when = f" when={u.when}" if u.when else ""
        lines.append(f"[{u.ref}] speaker={who}{when}\n{u.text}\n")
    return "\n".join(lines)


def extract_memories(
    model: ChatModel, units: list[Unit], *, owner_name: str, source_file: str
) -> tuple[list[Memory], int]:
    """Return (accepted memories, rejected count) for a list of units."""
    accepted: list[Memory] = []
    rejected = 0
    by_ref = {u.ref: u for u in units}
    for batch in _batches(units):
        prompt = f"The person is {owner_name}. Extract memories from these units:\n\n" + _render(
            batch
        )
        try:
            result = complete_json(
                model,
                [Message("system", EXTRACT_SYSTEM % ", ".join(DOMAINS)), Message("user", prompt)],
                ExtractionResult,
            )
        except ModelError as exc:
            log.warning("extraction batch failed: %s", exc)
            rejected += 1
            continue
        for p in result.memories:
            unit = by_ref.get(p.unit_ref)
            if unit is None or p.excerpt.strip() not in unit.text:
                rejected += 1  # fabricated or misattributed excerpt
                continue
            if p.domain not in DOMAINS:
                p.domain = "stories"
            confidence = p.confidence if unit.speaker == "owner" else min(p.confidence, 0.6)
            ref = f"{source_file}:{unit.ref}"
            accepted.append(
                Memory(
                    memory_id=Memory.make_id(p.text, ref),
                    text=p.text.strip(),
                    kind=p.kind,
                    domain=p.domain,
                    source_kind="file",
                    source_ref=ref,
                    excerpt=p.excerpt.strip(),
                    confidence=round(confidence, 2),
                    happened_at=p.happened_at or unit.when,
                    tags=[t.lower() for t in p.tags][:6],
                )
            )
    return accepted, rejected


def ingest_path(
    vault: Vault,
    model: ChatModel,
    embedder: Embedder,
    path: Path,
    *,
    report: IngestReport | None = None,
) -> IngestReport:
    """Ingest one file or every supported file under a folder."""
    report = report or IngestReport()
    meta = vault.load_meta()
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    for file in files:
        if file.name.startswith("."):
            continue
        report.files_seen += 1
        data = file.read_bytes()
        digest = Vault.content_hash(data)
        if vault.has_source(digest):
            log.info("skip %s: already ingested", file.name)
            report.files_skipped += 1
            continue
        try:
            units = parse_file(file, data, owner_name=meta.owner_name)
        except UnsupportedFileError as exc:
            report.files_skipped += 1
            report.errors.append(str(exc))
            continue
        report.units += len(units)
        memories, rejected = extract_memories(
            model, units, owner_name=meta.owner_name, source_file=file.name
        )
        report.memories_rejected += rejected
        new = [m for m in memories if vault.get(m.memory_id) is None]
        if new:
            vecs = embedder.encode([m.text for m in new])
            vault.add_memories(list(zip(new, vecs, strict=True)))
        vault.archive_source(file, data, memory_count=len(new))
        report.memories_added += len(new)
        log.info(
            "ingested %s: %d units -> %d memories (%d rejected)",
            file.name,
            len(units),
            len(new),
            rejected,
        )
    return report
