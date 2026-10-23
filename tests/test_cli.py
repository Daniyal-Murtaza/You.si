"""End-to-end tests through the real CLI with a scripted model and offline embedder.

These cover the journeys a person actually takes: create a vault, be
interviewed, pause and resume, see and confirm the persona, chat and correct
the twin, run the fidelity test, try a scenario, set legacy rules, export and
import. No network, no model API.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

import yousi.cli as cli
from yousi.ingest.retrieval import HashEmbedder
from yousi.model import ScriptedChat
from yousi.vault import Vault

runner = CliRunner()


def fake_model_step(msgs):
    """A stand-in model that answers each prompt type plausibly and groundedly."""
    system = msgs[0].content
    user = msgs[-1].content
    if "Turn the answer into 1-3 memory sentences" in system:
        answer = user.split("Answer: ", 1)[1].strip()
        domain = (
            "family"
            if "father" in answer.lower()
            else "places"
            if "live" in answer.lower()
            else "identity"
        )
        return json.dumps({"memories": [{"text": answer, "kind": "fact", "domain": domain}]})
    if "ask ONE natural follow-up" in system:
        return json.dumps({"question": ""})
    if "Describe who they are" in user:
        ids = re.findall(r"\[(m[0-9a-f]{12})\]", user)
        return json.dumps(
            {
                "summary": "Direct and family-minded.",
                "values": [{"text": "Values family", "evidence": ids[:1]}],
                "speech_style": [{"text": "Short, plain sentences", "evidence": ids[:1]}],
                "big_five": {
                    "openness": 60,
                    "conscientiousness": 70,
                    "extraversion": 40,
                    "agreeableness": 65,
                    "neuroticism": 35,
                    "rationale": "test",
                },
            }
        )
    if "Classify their message" in system:
        if "Zaidi" in user:
            mid = re.search(r"\[(m[0-9a-f]{12})\] \(family\)", user)
            return json.dumps(
                {
                    "is_correction": True,
                    "wrong_memory_id": mid.group(1) if mid else "",
                    "corrected_text": "My father is called Murtaza Zaidi.",
                }
            )
        return json.dumps({"is_correction": False})
    if "You are the recorded twin" in system:
        ids = re.findall(r"^\[(m[0-9a-f]{12})\]", user, re.M)  # memories are in the user turn
        return f"Murtaza, my father [{ids[0]}]." if ids else "I don't know that about myself yet."
    if "predicting how" in system:
        ids = re.findall(r"\[(m[0-9a-f]{12})\]", user)
        return json.dumps(
            {
                "reaction": "I'd call my father first.",
                "reasoning": "family matters",
                "evidence": ids[:1],
                "confidence": 0.6 if ids else 0.1,
            }
        )
    return "ok"


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "build_model", lambda settings: ScriptedChat(script=[fake_model_step]))
    monkeypatch.setattr(cli, "LocalEmbedder", lambda name: HashEmbedder())
    return {"YOUSI_VAULT": str(tmp_path / "vault"), "YOUSI_PROVIDER": "fake"}


def run(env, args, input=None):
    return runner.invoke(cli.app, args, input=input, env=env, catch_exceptions=False)


def test_full_journey(env):
    vault_path = Path(env["YOUSI_VAULT"])

    r = run(env, ["init", "--name", "Daniyal", "--birth-year", "1999"])
    assert r.exit_code == 0 and "Vault created" in r.output
    assert run(env, ["init", "--name", "Again"]).exit_code == 1  # refuses to overwrite

    # Interview: answer one, exit -> paused with exit code 3, question stays pending.
    r = run(env, ["interview", "-n", "3"], input="My father is called Murtaza.\nexit\n")
    assert r.exit_code == 3 and "+ remembered: My father is called Murtaza." in r.output
    assert "Paused" in r.output
    vault = Vault.open(vault_path)
    assert vault.count() == 1

    # Resume: same session, the pending question is asked again, skip works, budget ends it.
    r = run(env, ["interview", "-n", "3"], input="I live in Seattle.\nskip\n")
    assert r.exit_code == 0 and "Resuming session" in r.output
    assert "Session complete: 3 questions, 1 new memories" in r.output
    assert vault.count() == 2

    # Persona: inferred with evidence, then confirmed/edited/rejected interactively.
    r = run(env, ["persona"])
    assert r.exit_code == 0 and "Values family (1 memories)" in r.output and "0%" in r.output
    r = run(env, ["persona", "--review"], input="y\ne\nShort and dry\n")
    assert r.exit_code == 0
    persona = json.loads((vault_path / "persona.json").read_text())
    assert persona["values"][0]["status"] == "confirmed"
    assert persona["speech_style"][0] == {
        **persona["speech_style"][0],
        "status": "edited",
        "text": "Short and dry",
    }

    # Chat: grounded answer with source shown; correction updates the vault.
    r = run(
        env, ["chat"], input="Who is your father?\nNo, my father is called Murtaza Zaidi.\nexit\n"
    )
    assert r.exit_code == 0
    assert "Murtaza, my father" in r.output and "interview:q" in r.output
    assert "Corrected memory" in r.output
    father = [m for m in vault.all_memories() if "father" in m.text]
    assert len(father) == 1 and father[0].text == "My father is called Murtaza Zaidi."
    assert father[0].source_kind == "correction"

    # Status reflects everything.
    r = run(env, ["status"])
    assert "family" in r.output and "100% confirmed" in r.output

    # Fidelity test: one round, judge picks A; owner answer saved.
    r = run(env, ["test", "-n", "1"], input="Something honest.\nA\n")
    assert r.exit_code == 0 and "1 rounds" in r.output and "saved as memories" in r.output
    assert list((vault_path / "sessions").glob("fidelity-*.json"))

    # Scenario: labelled prediction; Enter accepts without adding a memory.
    before = vault.count()
    r = run(env, ["scenario", "Someone insults your family"], input="\n")
    assert r.exit_code == 0 and "PREDICTION" in r.output and "60%" in r.output
    assert vault.count() == before
    r = run(env, ["scenario", "You win the lottery"], input="I'd keep working.\n")
    assert "+ remembered: I'd keep working." in r.output

    # Legacy protocol.
    assert run(env, ["legacy", "executor", "sister@example.com"]).exit_code == 0
    assert run(env, ["legacy", "posthumous", "false"]).exit_code == 0
    r = run(env, ["chat", "--visitor"], input="hi\n")
    assert r.exit_code == 1 and "did not allow others" in r.output
    run(env, ["legacy", "posthumous", "true"])
    r = run(env, ["chat", "--visitor"], input="Who was his father?\nexit\n")
    assert r.exit_code == 0 and "Murtaza" in r.output
    r = run(env, ["legacy", "off-switch", "true"], input="YES\n")
    assert "Off switch engaged" in r.output
    r = run(env, ["chat"], input="hi\n")
    assert r.exit_code == 1 and "switched this twin off" in r.output
    run(env, ["legacy", "off-switch", "false"])
    r = run(env, ["legacy", "show"])
    assert "sister@example.com" in r.output and "Off switch: off" in r.output

    # Export and import into a fresh vault; memories and persona survive, search works.
    r = run(env, ["export"])
    assert r.exit_code == 0 and "Exported to" in r.output
    export_path = next((vault_path / "exports").glob("yousi-export-*.json"))
    copy_env = {**env, "YOUSI_VAULT": env["YOUSI_VAULT"] + "-copy"}
    r = run(copy_env, ["import", str(export_path)])
    assert r.exit_code == 0
    copy = Vault.open(Path(copy_env["YOUSI_VAULT"]))
    assert copy.count() == vault.count()
    assert copy.load_meta().executors == ["sister@example.com"]
    r = run(copy_env, ["chat"], input="Who is your father?\nexit\n")
    assert "indexing memories" in r.output and "Murtaza" in r.output


def test_commands_fail_cleanly_without_vault(env):
    for args in (["status"], ["chat"], ["interview"], ["export"]):
        r = run(env, args)
        assert r.exit_code == 1 and "yousi init" in r.output


def test_ingest_from_inbox(env, tmp_path, monkeypatch):
    def extract_step(msgs):
        if "extract what they reveal" in msgs[0].content:
            ref = re.search(r"\[(journal\.md#p\d+)\]", msgs[-1].content).group(1)
            return json.dumps(
                {
                    "memories": [
                        {
                            "text": "I usually sleep around midnight.",
                            "kind": "habit",
                            "domain": "routines",
                            "excerpt": "I usually sleep around midnight",
                            "unit_ref": ref,
                        }
                    ]
                }
            )
        return fake_model_step(msgs)

    monkeypatch.setattr(cli, "build_model", lambda s: ScriptedChat(script=[extract_step]))
    run(env, ["init", "--name", "Daniyal"])
    inbox = Path(env["YOUSI_VAULT"]) / "inbox"
    (inbox / "journal.md").write_text("# Notes\n\nI usually sleep around midnight. Coffee first.\n")
    r = run(env, ["ingest"])
    assert r.exit_code == 0 and "+1" in r.output
    r = run(env, ["ingest"])  # idempotent
    assert "+0" in r.output and "1 skipped" in r.output
    r = run(env, ["chat"], input="When do you sleep?\nexit\n")
    assert "file:journal.md" in r.output


def test_reindex_recovers_from_deleted_database(env):
    run(env, ["init", "--name", "D"])
    run(env, ["interview", "-n", "1"], input="I live in Seattle.\n")
    vault_path = Path(env["YOUSI_VAULT"])
    (vault_path / "memories.sqlite").unlink()
    r = run(env, ["reindex"])
    assert r.exit_code == 0 and "Rebuilt index from 1 memory files; embedded 1" in r.output
    assert Vault.open(vault_path).count() == 1
