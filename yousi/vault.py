"""The vault: everything the twin knows, in a folder the person owns.

Layout of ``<vault>/``::

    vault.json            identity, created date, settings, legacy protocol
    memories.sqlite       memory records + embeddings (fast queries)
    memories/YYYY/MM/     one Markdown file per memory (human readable mirror)
    persona.json          confirmed personality spec
    inbox/                drop files here to ingest
    sources/              ingested files, archived by content hash
    sessions/             interview and chat transcripts
    exports/              full-vault exports
    vault.log

Design rules:

* The SQLite database is a cache of the Markdown files, not the other way
  round. ``rebuild_index()`` can regenerate it from the files alone, so the
  vault survives any software change and can be read with a text editor.
* Every memory carries provenance: where it came from (a file, an interview
  answer, a chat correction), when, and the exact source excerpt.
* Nothing is ever hard-deleted. Retracting a memory marks it retracted and
  records why; the twin stops using it but the history stays.
* Writes are atomic.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import sqlite3
import tempfile
import threading
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np

MemoryKind = Literal[
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
SourceKind = Literal["interview", "file", "chat", "correction", "inferred"]

DOMAINS: tuple[str, ...] = (
    "identity",
    "family",
    "relationships",
    "work",
    "health",
    "routines",
    "beliefs",
    "values",
    "goals",
    "fears",
    "humour",
    "tastes",
    "money",
    "places",
    "stories",
    "communication",
)


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _atomic_write(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(data)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def _dump(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False) + "\n"


# --------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------


@dataclass
class Memory:
    """One thing the twin knows about the person."""

    memory_id: str
    text: str
    kind: MemoryKind
    domain: str
    source_kind: SourceKind
    source_ref: str  # file path, question id, session id
    excerpt: str = ""  # exact source text this was derived from
    confidence: float = 1.0  # 1.0 = stated by the person; lower = inferred
    created_at: str = field(default_factory=now_iso)
    happened_at: str = ""  # when the thing itself happened/applies, if known
    tags: list[str] = field(default_factory=list)
    retracted: bool = False
    retracted_reason: str = ""
    supersedes: str = ""  # memory_id this replaces (corrections)

    @staticmethod
    def make_id(text: str, source_ref: str) -> str:
        return "m" + hashlib.sha1(f"{source_ref}\n{text}".encode()).hexdigest()[:12]

    def to_markdown(self) -> str:
        front = {k: v for k, v in asdict(self).items() if k != "text"}
        lines = ["---"]
        for k, v in front.items():
            lines.append(f"{k}: {json.dumps(v, ensure_ascii=False)}")
        lines += ["---", "", self.text, ""]
        return "\n".join(lines)

    @classmethod
    def from_markdown(cls, text: str) -> Memory:
        head, _, body = text.partition("\n---\n")
        head = head.removeprefix("---\n")
        data: dict[str, Any] = {}
        for line in head.splitlines():
            if ":" in line:
                k, _, v = line.partition(":")
                data[k.strip()] = json.loads(v.strip())
        data["text"] = body.strip()
        return cls(**data)


@dataclass
class VaultMeta:
    owner_name: str
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)
    birth_year: int | None = None
    language: str = "en"
    # Legacy protocol
    executors: list[str] = field(default_factory=list)  # names/emails allowed to run it after
    allow_posthumous: bool = True
    off_switch_engaged: bool = False  # owner can disable the twin for good
    disclosure: str = (
        "I am an AI twin of {owner}. I speak from their recorded memories, not as them."
    )
    counters: dict[str, int] = field(default_factory=dict)

    def disclosure_text(self) -> str:
        """The disclosure with the owner's name filled in; safe for any text."""
        return self.disclosure.replace("{owner}", self.owner_name)


# --------------------------------------------------------------------------
# Vault
# --------------------------------------------------------------------------


class VaultNotFoundError(FileNotFoundError):
    pass


class Vault:
    SCHEMA = """
    CREATE TABLE IF NOT EXISTS memories (
        memory_id TEXT PRIMARY KEY,
        text TEXT NOT NULL,
        kind TEXT NOT NULL,
        domain TEXT NOT NULL,
        source_kind TEXT NOT NULL,
        source_ref TEXT NOT NULL,
        excerpt TEXT NOT NULL DEFAULT '',
        confidence REAL NOT NULL DEFAULT 1.0,
        created_at TEXT NOT NULL,
        happened_at TEXT NOT NULL DEFAULT '',
        tags TEXT NOT NULL DEFAULT '[]',
        retracted INTEGER NOT NULL DEFAULT 0,
        retracted_reason TEXT NOT NULL DEFAULT '',
        supersedes TEXT NOT NULL DEFAULT '',
        embedding BLOB
    );
    CREATE INDEX IF NOT EXISTS idx_mem_domain ON memories(domain);
    CREATE INDEX IF NOT EXISTS idx_mem_source ON memories(source_ref);
    CREATE TABLE IF NOT EXISTS sources (
        content_hash TEXT PRIMARY KEY,
        original_name TEXT NOT NULL,
        stored_path TEXT NOT NULL,
        ingested_at TEXT NOT NULL,
        memory_count INTEGER NOT NULL DEFAULT 0
    );
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self._lock = threading.RLock()
        self._conn: sqlite3.Connection | None = None

    # --- lifecycle ---------------------------------------------------------

    @classmethod
    def create(cls, root: Path, owner_name: str, **meta: Any) -> Vault:
        root = Path(root)
        if (root / "vault.json").exists():
            raise FileExistsError(f"a vault already exists at {root}")
        for sub in ("memories", "inbox", "sources", "sessions", "exports"):
            (root / sub).mkdir(parents=True, exist_ok=True)
        vault = cls(root)
        vault.save_meta(VaultMeta(owner_name=owner_name, **meta))
        vault.conn()  # creates the database
        return vault

    @classmethod
    def open(cls, root: Path) -> Vault:
        root = Path(root)
        if not (root / "vault.json").exists():
            raise VaultNotFoundError(f"no vault at {root}; run `yousi init` first")
        return cls(root)

    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.root / "memories.sqlite", check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.executescript(self.SCHEMA)
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    @property
    def log_path(self) -> Path:
        return self.root / "vault.log"

    # --- meta ---------------------------------------------------------------

    def load_meta(self) -> VaultMeta:
        return VaultMeta(**json.loads((self.root / "vault.json").read_text(encoding="utf-8")))

    def save_meta(self, meta: VaultMeta) -> None:
        meta.updated_at = now_iso()
        _atomic_write(self.root / "vault.json", _dump(asdict(meta)))

    def update_meta(self, **changes: Any) -> VaultMeta:
        with self._lock:
            meta = self.load_meta()
            for k, v in changes.items():
                setattr(meta, k, v)
            self.save_meta(meta)
            return meta

    def bump(self, counter: str, by: int = 1) -> int:
        with self._lock:
            meta = self.load_meta()
            meta.counters[counter] = meta.counters.get(counter, 0) + by
            self.save_meta(meta)
            return meta.counters[counter]

    # --- memories -----------------------------------------------------------

    def _md_path(self, memory: Memory) -> Path:
        d = datetime.fromisoformat(memory.created_at)
        return self.root / "memories" / f"{d:%Y}" / f"{d:%m}" / f"{memory.memory_id}.md"

    def add_memory(self, memory: Memory, embedding: np.ndarray | None = None) -> Memory:
        """Insert or update a memory and its Markdown mirror."""
        with self._lock:
            _atomic_write(self._md_path(memory), memory.to_markdown())
            blob = embedding.astype(np.float32).tobytes() if embedding is not None else None
            self.conn().execute(
                """INSERT OR REPLACE INTO memories VALUES
                   (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    memory.memory_id,
                    memory.text,
                    memory.kind,
                    memory.domain,
                    memory.source_kind,
                    memory.source_ref,
                    memory.excerpt,
                    memory.confidence,
                    memory.created_at,
                    memory.happened_at,
                    json.dumps(memory.tags),
                    int(memory.retracted),
                    memory.retracted_reason,
                    memory.supersedes,
                    blob,
                ),
            )
            self.conn().commit()
            return memory

    def add_memories(self, items: list[tuple[Memory, np.ndarray | None]]) -> int:
        for memory, emb in items:
            self.add_memory(memory, emb)
        return len(items)

    @staticmethod
    def _row_to_memory(row: sqlite3.Row) -> Memory:
        return Memory(
            memory_id=row["memory_id"],
            text=row["text"],
            kind=row["kind"],
            domain=row["domain"],
            source_kind=row["source_kind"],
            source_ref=row["source_ref"],
            excerpt=row["excerpt"],
            confidence=row["confidence"],
            created_at=row["created_at"],
            happened_at=row["happened_at"],
            tags=json.loads(row["tags"]),
            retracted=bool(row["retracted"]),
            retracted_reason=row["retracted_reason"],
            supersedes=row["supersedes"],
        )

    def get(self, memory_id: str) -> Memory | None:
        row = (
            self.conn().execute("SELECT * FROM memories WHERE memory_id=?", (memory_id,)).fetchone()
        )
        return self._row_to_memory(row) if row else None

    def all_memories(self, *, include_retracted: bool = False) -> list[Memory]:
        sql = "SELECT * FROM memories" + ("" if include_retracted else " WHERE retracted=0")
        return [self._row_to_memory(r) for r in self.conn().execute(sql + " ORDER BY created_at")]

    def by_domain(self, domain: str) -> list[Memory]:
        rows = self.conn().execute(
            "SELECT * FROM memories WHERE domain=? AND retracted=0 ORDER BY created_at", (domain,)
        )
        return [self._row_to_memory(r) for r in rows]

    def count(self) -> int:
        return self.conn().execute("SELECT COUNT(*) FROM memories WHERE retracted=0").fetchone()[0]

    def domain_counts(self) -> dict[str, int]:
        rows = self.conn().execute(
            "SELECT domain, COUNT(*) FROM memories WHERE retracted=0 GROUP BY domain"
        )
        counts = {d: 0 for d in DOMAINS}
        for domain, n in rows:
            counts[domain] = n
        return counts

    def missing_embeddings(self) -> list[Memory]:
        rows = self.conn().execute(
            "SELECT * FROM memories WHERE retracted=0 AND embedding IS NULL ORDER BY created_at"
        )
        return [self._row_to_memory(r) for r in rows]

    def set_embedding(self, memory_id: str, embedding: np.ndarray) -> None:
        with self._lock:
            self.conn().execute(
                "UPDATE memories SET embedding=? WHERE memory_id=?",
                (embedding.astype(np.float32).tobytes(), memory_id),
            )
            self.conn().commit()

    def embeddings(self) -> tuple[list[str], np.ndarray]:
        """All active memory ids and their embedding matrix (rows without one are skipped)."""
        rows = (
            self.conn()
            .execute(
                "SELECT memory_id, embedding FROM memories "
                "WHERE retracted=0 AND embedding IS NOT NULL"
            )
            .fetchall()
        )
        if not rows:
            return [], np.zeros((0, 0), dtype=np.float32)
        ids = [r["memory_id"] for r in rows]
        mat = np.vstack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
        return ids, mat

    def retract(self, memory_id: str, reason: str) -> Memory | None:
        with self._lock:
            memory = self.get(memory_id)
            if memory is None:
                return None
            memory.retracted, memory.retracted_reason = True, reason
            return self.add_memory(memory)

    def correct(self, old_id: str, new_text: str, *, source_ref: str, embedding=None) -> Memory:
        """The person says a memory is wrong: retract it and add the corrected one."""
        with self._lock:
            old = self.get(old_id)
            if old is None:
                raise KeyError(old_id)
            self.retract(old_id, f"corrected by owner -> {new_text[:60]}")
            new = Memory(
                memory_id=Memory.make_id(new_text, source_ref),
                text=new_text,
                kind=old.kind,
                domain=old.domain,
                source_kind="correction",
                source_ref=source_ref,
                excerpt=new_text,
                confidence=1.0,
                supersedes=old_id,
                tags=list(old.tags),
            )
            return self.add_memory(new, embedding)

    # --- sources --------------------------------------------------------------

    @staticmethod
    def content_hash(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()[:20]

    def has_source(self, content_hash: str) -> bool:
        row = (
            self.conn()
            .execute("SELECT 1 FROM sources WHERE content_hash=?", (content_hash,))
            .fetchone()
        )
        return row is not None

    def archive_source(self, original: Path, data: bytes, memory_count: int) -> Path:
        h = self.content_hash(data)
        stored = self.root / "sources" / f"{h}{original.suffix.lower()}"
        if not stored.exists():
            stored.write_bytes(data)
        with self._lock:
            self.conn().execute(
                "INSERT OR REPLACE INTO sources VALUES (?,?,?,?,?)",
                (h, original.name, str(stored.relative_to(self.root)), now_iso(), memory_count),
            )
            self.conn().commit()
        return stored

    # --- rebuild / export ------------------------------------------------------

    def rebuild_index(self) -> int:
        """Regenerate the database from the Markdown mirror (no embeddings)."""
        with self._lock:
            self.close()
            (self.root / "memories.sqlite").unlink(missing_ok=True)
            n = 0
            for path in sorted((self.root / "memories").rglob("*.md")):
                memory = Memory.from_markdown(path.read_text(encoding="utf-8"))
                self.add_memory(memory)
                n += 1
            return n

    def export_all(self) -> Path:
        """One JSON file with everything: meta, persona, memories. Portable, vendor-free."""
        payload = {
            "format": "yousi-vault-export",
            "version": 1,
            "exported_at": now_iso(),
            "meta": asdict(self.load_meta()),
            "persona": json.loads((self.root / "persona.json").read_text(encoding="utf-8"))
            if (self.root / "persona.json").exists()
            else None,
            "memories": [asdict(m) for m in self.all_memories(include_retracted=True)],
        }
        out = self.root / "exports" / f"yousi-export-{datetime.now(UTC):%Y%m%d-%H%M%S}.json"
        _atomic_write(out, _dump(payload))
        return out

    @classmethod
    def import_from(cls, export_path: Path, root: Path) -> Vault:
        payload = json.loads(Path(export_path).read_text(encoding="utf-8"))
        if payload.get("format") != "yousi-vault-export":
            raise ValueError("not a You.si export file")
        meta = payload["meta"]
        vault = cls.create(
            root,
            meta.pop("owner_name"),
            **{k: v for k, v in meta.items() if k not in ("created_at", "updated_at")},
        )
        for m in payload["memories"]:
            vault.add_memory(Memory(**m))
        if payload.get("persona"):
            _atomic_write(root / "persona.json", _dump(payload["persona"]))
        return vault

    # --- generic json files -------------------------------------------------------

    def _inside(self, name: str) -> Path:
        path = (self.root / name).resolve()
        if self.root.resolve() not in path.parents:
            raise ValueError(f"refusing to write outside the vault: {name}")
        return path

    def write_json(self, name: str, obj: Any) -> None:
        _atomic_write(self._inside(name), _dump(obj))

    def read_json(self, name: str, default: Any = None) -> Any:
        path = self._inside(name)
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default
