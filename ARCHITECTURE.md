# Architecture

You.si builds a *twin* of one person: software that answers as they would,
from what they actually said and wrote. It does not claim to hold a
consciousness. It holds three things that can be inspected, tested and
handed on: **memories** with provenance, a **persona** the person confirmed,
and a **model adapter** that can be swapped for any vendor or a local model.

```
 you ──► interview ─┐                              ┌─► chat (owner / visitor)
 files ─► ingest ───┼─► VAULT (your folder) ◄──────┼─► fidelity test
                    │     memories/*.md + sqlite    ├─► scenario sandbox
                    │     persona.json              └─► export (one JSON file)
                    │     vault.json (legacy rules)
                    └── model adapter: OpenAI | Ollama (local) | scripted (tests)
                        embeddings: local open model, never leaves the machine
```

## The three honest claims

| Claim the product makes | How the code keeps it true |
|---|---|
| **Every statement rests on something you said.** | Memories are extracted only with an exact `excerpt` that must appear in the source (`ingest/extract.py`); interview answers are stored verbatim as the excerpt. The twin may cite only memory ids it was given this turn; invented ids are stripped and the reply is flagged ungrounded (`twin.py`). |
| **It says "I don't know yet" instead of guessing.** | Retrieval runs before generation; with no memory above the relevance floor the model is told it has none. Predictions in the sandbox with no evidence are capped at 30% confidence and always labelled `PREDICTION`. |
| **It is yours and it outlives any company.** | The vault is a folder of Markdown + JSON; SQLite is a rebuildable cache (`yousi reindex`). `yousi export` writes one portable JSON file; `yousi import` recreates everything. The model is behind a four-method protocol; Ollama runs open weights locally with no account. Embeddings are computed on your machine. |

## Modules

| Path | Role |
|---|---|
| `vault.py` | Folder layout, `Memory` and `VaultMeta` records, atomic writes, corrections that retract-and-link rather than delete, export/import, index rebuild. |
| `model.py` | `ChatModel` protocol; `OpenAIChat`, `OllamaChat`, `ScriptedChat`; `complete_json` with one repair attempt. |
| `ingest/parsers.py` | Standard-library parsers: Markdown/text, WhatsApp exports, ChatGPT exports, generic message JSON, `.eml`/`.mbox`, PDF text. Each unit tagged with speaker and reference. |
| `ingest/extract.py` | Model proposes memories; code enforces excerpt-in-source, caps confidence for other people's words, dedupes by content hash. |
| `ingest/retrieval.py` | Local embeddings (bge-small via sentence-transformers) and hybrid BM25 + vector search. |
| `persona.py` | Trait schema with evidence ids; inference; owner verdicts (confirm/edit/reject) that survive re-inference. |
| `interview.py` | Gap-driven questions by least-covered domain; seed bank plus model follow-ups; skip / don't know / exit / resume; answers → memories. |
| `twin.py` | Grounded chat in owner and visitor modes; citation enforcement; in-chat corrections become memories; off-switch and disclosure. |
| `fidelity.py` | Paired-answer judge test with fool rate and grounded rate; scenario sandbox with evidence-capped confidence. |
| `cli.py` | Typer commands; the legacy protocol group. |

## What "like you" means here, measurably

- **Memories** answer *what* you know, did, like, fear.
- **Persona** answers *how* you are: values, traits, speech style, routines, relationships, goals, fears, plus a Big Five estimate always shown as uncertain. Every trait carries evidence ids; the owner confirms or rejects each one, and the twin treats unconfirmed traits as guesses.
- **Fidelity test** answers *how close is it*: you and the twin answer the same question, a judge who knows you picks the real one. 50% is indistinguishable. The grounded rate is reported beside it so a twin that bluffs convincingly is still visible.

## Data flow for one chat turn (owner mode)

1. Retrieve the top-k memories for the message (hybrid search, local).
2. Ask the model whether the message is a correction or a new fact about the owner. If yes, update the vault first and force the new memory into the context.
3. Build the system prompt: disclosure, persona (confirmed first, guesses labelled), retrieved memories with ids.
4. Generate. Strip any cited id that was not in the retrieved set; mark grounded/ungrounded.
5. Persist the transcript under `sessions/`.

Visitor mode skips step 2 and never changes memories or persona; it still records the transcript under `sessions/` so the owner or executors can see what was asked. It is blocked entirely by the off switch or when posthumous access is disabled.

## Legacy protocol

Stored in `vault.json`, editable with `yousi legacy ...`:

- `executors`: who may run the twin after the owner (recorded for the people inheriting the folder; enforcement is organisational, not cryptographic, in Phase 0).
- `allow_posthumous`: whether visitor mode works at all.
- `off_switch_engaged`: silences the twin for everyone, including the owner. Memories stay readable as files; nothing speaks for the person.
- `disclosure`: the sentence every conversation starts with. Default: "I am an AI twin of {owner}. I speak from their recorded memories, not as them."

## Tradeoffs

- **Retrieval over fine-tuning.** Memories stay editable, citable and deletable; a fine-tuned model would bake them in. Phase 1 may add a small style adapter for voice, with the memory store still authoritative for facts.
- **BM25 + small local embeddings.** Deterministic, free, offline. Weaker on nuance than a large embedding API; acceptable for a few thousand short personal memories.
- **OpenAI as default brain.** Best quality today. The twin's data never depends on it; switching to Ollama is one setting.
- **No voice or face in Phase 0.** Text fidelity first. A convincing face on a twin that does not yet sound like you is the failure mode of every "digital afterlife" product before this one.
- **Executor enforcement is social, not cryptographic.** A proper scheme (encrypted vault, key escrow, dead-man's switch) is Phase 1 work.
- **Correction detection costs a model call per owner turn.** Worth it: the owner correcting the twin is the single most valuable training signal.

## What is deliberately not claimed

The twin does not remember anything you did not tell it or write down. It does not have experiences, continuity, or feelings. It predicts, with stated confidence, how you would react; it never reports a prediction as something you did. These are limits of the technology, and the product is more trustworthy for stating them.

## Grounded is not true

The strongest guarantee here is provenance: every cited id maps to a memory, every memory to the owner's own words. The guarantee that is *not* made is that the sentence connecting two true memories is itself true. The live run showed the twin joining "stuck debugging" and "headaches without coffee" into a stress-coping story the owner never told. Both citations were real; the link was invented. Retrieval grounding bounds *what* the twin can say, not *how* it combines it. The mitigations are structural: the owner corrects it in chat and the correction becomes a memory; the fidelity test measures the gap; visitor mode always shows the sources so a reader can check the join themselves.

## Time spent

Roughly nine hours: research into the existing products and why they failed or pivoted, a spike confirming local embeddings run on this machine, the build in nine commits with tests at every step, then three hours of live runs against gpt-4.1-mini that found six real defects (each fixed with a regression test) and produced the fidelity numbers in the README. 71 offline tests run in about three seconds.
