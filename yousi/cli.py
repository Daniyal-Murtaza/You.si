"""Command line interface.

yousi init --name "Daniyal" [--vault PATH]
yousi ingest [PATH]            drop files in <vault>/inbox or pass a path
yousi interview                the twin asks you questions
yousi persona [--review]       infer / review who the twin thinks you are
yousi chat [--visitor]         talk to your twin
yousi test                     fidelity test: can a judge tell you apart?
yousi scenario "..."           how would I react if...
yousi status                   what the twin knows, by life domain
yousi legacy ...               executors, off switch, posthumous access
yousi export / import          move the whole vault, vendor-free
"""

from __future__ import annotations

import random
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from yousi import __version__
from yousi.config import MissingCredentialError, Settings, load_settings
from yousi.ingest.extract import ingest_path
from yousi.ingest.retrieval import Embedder, LocalEmbedder, embed_missing
from yousi.logging_setup import attach_file, configure_logging
from yousi.model import ChatModel, ModelError, build_model
from yousi.vault import DOMAINS, Vault, VaultNotFoundError

app = typer.Typer(
    add_completion=False, no_args_is_help=True, help="You.si: a personal twin you own."
)
legacy_app = typer.Typer(help="What happens to the twin after you.")
app.add_typer(legacy_app, name="legacy")
console = Console()
err = Console(stderr=True)

VaultOpt = typer.Option(None, "--vault", help="Vault folder (default from YOUSI_VAULT / .env)")
EXIT_WORDS = {"exit", "quit", "q", ":q"}


# --------------------------------------------------------------------------
# wiring
# --------------------------------------------------------------------------


class _Ctx:
    def __init__(self, vault_path: Path | None, *, need_model: bool = True) -> None:
        self.settings: Settings = load_settings(
            **({"YOUSI_VAULT": vault_path} if vault_path else {})
        )
        configure_logging(self.settings.log_level, quiet_console=True)
        try:
            self.vault = Vault.open(self.settings.vault)
        except VaultNotFoundError as exc:
            _fail(str(exc))
        attach_file(self.vault.log_path)
        self._model: ChatModel | None = None
        self._embedder: Embedder | None = None
        if need_model:
            try:
                self._model = build_model(self.settings)
            except MissingCredentialError as exc:
                _fail(str(exc))

    @property
    def model(self) -> ChatModel:
        assert self._model is not None
        return self._model

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = LocalEmbedder(self.settings.embed_model)
            # Imported or rebuilt vaults have memories without vectors; fix quietly.
            if self.vault.missing_embeddings():
                console.print("[dim]indexing memories ...[/dim]")
                embed_missing(self.vault, self._embedder)
        return self._embedder


def _fail(message: str, code: int = 1) -> None:
    err.print(f"[red]error:[/red] {message}")
    raise typer.Exit(code)


def _ask(prompt: str = "> ") -> str:
    try:
        return typer.prompt(prompt, prompt_suffix="", default="", show_default=False)
    except (KeyboardInterrupt, EOFError):
        return "exit"


# --------------------------------------------------------------------------
# init / status / ingest
# --------------------------------------------------------------------------


@app.callback(invoke_without_command=True)
def _main(
    ctx: typer.Context, version: bool = typer.Option(False, "--version", help="Show version")
) -> None:
    if version:
        console.print(f"yousi {__version__}")
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())
        raise typer.Exit()


@app.command()
def init(
    name: str = typer.Option(..., "--name", help="Your name"),
    birth_year: int | None = typer.Option(None, "--birth-year"),
    vault: Path | None = VaultOpt,
) -> None:
    """Create your vault: a folder you own. Put it somewhere you back up."""
    settings = load_settings(**({"YOUSI_VAULT": vault} if vault else {}))
    configure_logging(settings.log_level, quiet_console=True)
    try:
        v = Vault.create(settings.vault, name, birth_year=birth_year)
    except FileExistsError as exc:
        _fail(str(exc))
    console.print(
        Panel(
            f"Vault created at [bold]{v.root.resolve()}[/bold]\n\n"
            f"Drop notes, chat exports, emails into [bold]{v.root / 'inbox'}[/bold] and run "
            f"[bold]yousi ingest[/bold], or start with [bold]yousi interview[/bold].\n"
            f"Everything the twin learns is written as plain files in this folder. "
            f"It is yours; sync it to any cloud you trust.",
            title=f"You.si - {name}",
            border_style="green",
        )
    )


@app.command()
def status(vault: Path | None = VaultOpt) -> None:
    """What the twin knows so far, by life domain."""
    c = _Ctx(vault, need_model=False)
    meta = c.vault.load_meta()
    counts = c.vault.domain_counts()
    table = Table(title=f"{meta.owner_name}'s twin: {c.vault.count()} memories", box=None)
    table.add_column("domain")
    table.add_column("memories", justify="right")
    table.add_column("")
    for d in DOMAINS:
        n = counts.get(d, 0)
        table.add_row(d, str(n), "[green]" + "#" * min(n, 30) + "[/green]" if n else "[dim]-[/dim]")
    console.print(table)
    from yousi.persona import load_persona

    persona = load_persona(c.vault)
    if persona:
        console.print(
            f"Persona: {len(persona.all_traits())} traits, "
            f"{persona.confirmation_rate():.0%} confirmed by you. "
            f"Run [bold]yousi persona --review[/bold] to confirm more."
        )
    else:
        console.print("[dim]No persona yet. Run `yousi persona` once you have some memories.[/dim]")
    flags = []
    if meta.off_switch_engaged:
        flags.append("[red]OFF SWITCH ENGAGED[/red]")
    if not meta.allow_posthumous:
        flags.append("posthumous access: disabled")
    if meta.executors:
        flags.append(f"executors: {', '.join(meta.executors)}")
    if flags:
        console.print("Legacy: " + " | ".join(flags))


@app.command()
def ingest(
    path: Path | None = typer.Argument(None, help="File or folder (default: inbox)"),
    vault: Path | None = VaultOpt,
) -> None:
    """Read your files and turn them into memories (every one quotes its source)."""
    c = _Ctx(vault)
    target = path or (c.vault.root / "inbox")
    if not target.exists():
        _fail(f"{target} does not exist")
    console.print(f"Reading {target} ...")
    try:
        report = ingest_path(c.vault, c.model, c.embedder, target)
    except ModelError as exc:
        _fail(str(exc))
    console.print(
        f"files: {report.files_seen} seen, {report.files_skipped} skipped | "
        f"units: {report.units} | memories: [green]+{report.memories_added}[/green] "
        f"({report.memories_rejected} rejected as ungrounded)"
    )
    for e in report.errors[:10]:
        console.print(f"  [yellow]- {e}[/yellow]")
    console.print(f"Total memories now: {c.vault.count()}")


# --------------------------------------------------------------------------
# interview
# --------------------------------------------------------------------------


@app.command()
def interview(
    questions: int | None = typer.Option(None, "--questions", "-n"), vault: Path | None = VaultOpt
) -> None:
    """The twin asks about what it knows least. Answer, 'skip', 'I don't know', or 'exit'."""
    from yousi.interview import InterviewEngine

    c = _Ctx(vault)
    eng = InterviewEngine(
        c.vault,
        c.model,
        c.embedder,
        max_questions=questions or c.settings.max_questions_per_session,
    )
    session = eng.unfinished()
    if session:
        console.print(f"[dim]Resuming session {session.session_id}[/dim]")
    else:
        session = eng.start()
    console.print(
        Panel(
            "Answer in your own words. Type [bold]skip[/bold], "
            "[bold]I don't know[/bold], or [bold]exit[/bold] to stop and resume later.",
            title="Interview",
            border_style="cyan",
        )
    )
    added = 0
    while (q := eng.ask(session)) is not None:
        console.print(f"\n[bold cyan]{q.text}[/bold cyan]")
        if q.reason:
            console.print(f"[dim]{q.reason}[/dim]")
        raw = _ask()
        try:
            status, memories = eng.answer(session, raw)
        except ModelError as exc:
            _fail(str(exc))
        if status == "exit":
            console.print("\nPaused. Run [bold]yousi interview[/bold] again to continue.")
            raise typer.Exit(3)
        if memories:
            added += len(memories)
            for m in memories:
                console.print(f"  [green]+ remembered:[/green] {m.text} [dim][{m.memory_id}][/dim]")
    eng.finish(session)
    console.print(
        f"\nSession complete: {session.asked} questions, {added} new memories. "
        f"Total: {c.vault.count()}."
    )


# --------------------------------------------------------------------------
# persona
# --------------------------------------------------------------------------


@app.command()
def persona(
    review: bool = typer.Option(False, "--review", help="Confirm/edit/reject traits"),
    refresh: bool = typer.Option(False, "--refresh", help="Re-infer from current memories"),
    vault: Path | None = VaultOpt,
) -> None:
    """Who the twin thinks you are, with evidence. Confirm it so the twin can trust it."""
    from yousi.persona import apply_verdict, infer_persona, load_persona, save_persona

    c = _Ctx(vault)
    current = load_persona(c.vault)
    if current is None or refresh:
        if c.vault.count() == 0:
            _fail("no memories yet; run `yousi interview` or `yousi ingest` first")
        console.print("Inferring who you are from your memories ...")
        try:
            current = infer_persona(c.model, c.vault, previous=current)
        except ModelError as exc:
            _fail(str(exc))
        save_persona(c.vault, current)

    console.print(
        Panel(
            current.summary or "(no summary yet)",
            title=f"{current.owner_name}",
            border_style="magenta",
        )
    )
    for section in current.SECTIONS:
        items = getattr(current, section)
        if not items:
            continue
        console.print(f"\n[bold]{section.replace('_', ' ').title()}[/bold]")
        for i, t in enumerate(items):
            mark = {
                "confirmed": "[green]v[/green]",
                "edited": "[green]e[/green]",
                "rejected": "[red]x[/red]",
                "proposed": "[yellow]?[/yellow]",
            }[t.status]
            console.print(f"  {mark} {t.text} [dim]({len(t.evidence)} memories)[/dim]")
            if review and t.status == "proposed":
                ans = _ask("    [y]es / [n]o / [e]dit / [s]kip / exit: ").strip().lower()
                if ans in EXIT_WORDS:
                    save_persona(c.vault, current)
                    raise typer.Exit(3)
                if ans.startswith("y"):
                    apply_verdict(current, section, i, "confirmed")
                elif ans.startswith("n"):
                    apply_verdict(current, section, i, "rejected", note=_ask("    why? "))
                elif ans.startswith("e"):
                    apply_verdict(current, section, i, "edited", edited_text=_ask("    new text: "))
                save_persona(c.vault, current)
    bf = current.big_five
    console.print(
        f"\n[dim]Personality estimate (uncertain): O{bf.openness} C{bf.conscientiousness} "
        f"E{bf.extraversion} A{bf.agreeableness} N{bf.neuroticism}[/dim]"
    )
    console.print(
        f"Confirmed: {current.confirmation_rate():.0%}"
        + ("" if review else "  (run with --review to confirm)")
    )


# --------------------------------------------------------------------------
# chat / test / scenario
# --------------------------------------------------------------------------


@app.command()
def chat(
    visitor: bool = typer.Option(
        False, "--visitor", help="Talk as someone else (family). Twin never learns."
    ),
    show_sources: bool = typer.Option(True, "--sources/--no-sources"),
    vault: Path | None = VaultOpt,
) -> None:
    """Talk to your twin. It cites a memory for every claim or says it doesn't know yet."""
    from yousi.twin import Twin, TwinDisabledError

    c = _Ctx(vault)
    twin = Twin(
        c.vault,
        c.model,
        c.embedder,
        mode="visitor" if visitor else "owner",
        top_k=c.settings.retrieval_top_k,
    )
    meta = c.vault.load_meta()
    console.print(
        Panel(
            meta.disclosure_text()
            + (
                "\n[dim]Owner mode: corrections you type become memories.[/dim]"
                if not visitor
                else ""
            ),
            title="Twin",
            border_style="blue",
        )
    )
    while True:
        msg = _ask("you > ")
        if msg.strip().lower() in EXIT_WORDS:
            break
        try:
            r = twin.reply(msg)
        except TwinDisabledError as exc:
            _fail(str(exc))
        except ModelError as exc:
            _fail(str(exc))
        tag = "" if r.grounded else " [yellow](no memory for this)[/yellow]"
        console.print(f"[bold blue]{meta.owner_name}'s twin >[/bold blue] {r.text}{tag}")
        if show_sources and r.cited:
            for m in r.cited:
                console.print(
                    f"   [dim]{m.memory_id}  {m.source_kind}:{m.source_ref}  "
                    f'"{m.excerpt[:90]}"[/dim]'
                )


@app.command()
def test(rounds: int = typer.Option(5, "--rounds", "-n"), vault: Path | None = VaultOpt) -> None:
    """Fidelity test: you and the twin answer the same questions; a judge picks the real one."""
    from yousi.fidelity import FidelityTest

    c = _Ctx(vault)
    ft = FidelityTest(c.vault, c.model, c.embedder, rng=random.Random())
    console.print(
        Panel(
            "Answer each question yourself first. Then you (or a friend who knows "
            "you) will see two answers and guess which is really you.",
            title="Fidelity test",
            border_style="yellow",
        )
    )
    asked: set[str] = set()
    for i in range(rounds):
        q = ft.pick_question(exclude=asked)
        asked.add(q)
        console.print(f"\n[bold]{i + 1}/{rounds}: {q}[/bold]")
        owner = _ask("you > ")
        if owner.strip().lower() in EXIT_WORDS:
            break
        try:
            rnd = ft.new_round(q, owner)
        except ModelError as exc:
            _fail(str(exc))
        a, b = FidelityTest.pair(rnd)
        console.print(f"\n  [bold]A:[/bold] {a}\n  [bold]B:[/bold] {b}")
        pick = _ask("judge: which one is the real person? [A/B] > ")
        if pick.strip().lower() in EXIT_WORDS:
            break
        try:
            ft.judge(rnd, pick)
        except ValueError:
            console.print("[yellow]skipped (answer A or B)[/yellow]")
            continue
        console.print(
            "  [dim]twin was "
            + rnd.twin_shown_as
            + (" - fooled you" if rnd.fooled else " - you spotted it")
            + "[/dim]"
        )
    ft.save()
    kept = ft.owner_answers_as_memories()
    console.print(
        Panel(
            ft.report.summary() + f"\nYour {kept} real answers were saved as memories.",
            title="Result",
            border_style="yellow",
        )
    )


@app.command()
def scenario(
    text: str = typer.Argument(..., help="e.g. 'I get offered a job abroad'"),
    vault: Path | None = VaultOpt,
) -> None:
    """How would I react if ... (a labelled prediction, with the memories behind it)."""
    from yousi.fidelity import Sandbox

    c = _Ctx(vault)
    sb = Sandbox(c.vault, c.model, c.embedder)
    try:
        r = sb.predict(text)
    except ModelError as exc:
        _fail(str(exc))
    p = r.prediction
    console.print(
        Panel(
            f"{p.reaction}\n\n[dim]Why: {p.reasoning}[/dim]\n"
            f"[dim]Confidence: {p.confidence:.0%}  |  would change it: "
            f"{p.what_would_change_it or '-'}[/dim]",
            title=r.label,
            border_style="yellow",
        )
    )
    for m in r.cited:
        console.print(f"   [dim]{m.memory_id}  {m.text}[/dim]")
    ans = _ask("Is that right? Type what you'd really do, or press Enter to accept: ")
    if ans.strip() and ans.strip().lower() not in EXIT_WORDS:
        m = sb.confirm(r, ans)
        console.print(f"[green]+ remembered:[/green] {m.text} [dim][{m.memory_id}][/dim]")


# --------------------------------------------------------------------------
# legacy protocol
# --------------------------------------------------------------------------


@legacy_app.command("show")
def legacy_show(vault: Path | None = VaultOpt) -> None:
    """Current legacy settings."""
    c = _Ctx(vault, need_model=False)
    m = c.vault.load_meta()
    execs = ", ".join(m.executors) or "[dim]none[/dim]"
    visitors = "yes" if m.allow_posthumous else "no"
    switch = "[red]ENGAGED - twin will not speak[/red]" if m.off_switch_engaged else "off"
    console.print(
        Panel(
            f"Executors (may run the twin after you): {execs}\n"
            f"Others may talk to the twin (visitor mode): {visitors}\n"
            f"Off switch: {switch}\n"
            f'Disclosure every reply starts with: "{m.disclosure}"',
            title="Legacy protocol",
            border_style="white",
        )
    )


@legacy_app.command("executor")
def legacy_executor(
    name: str = typer.Argument(..., help="Name or email"),
    remove: bool = typer.Option(False, "--remove"),
    vault: Path | None = VaultOpt,
) -> None:
    """Add or remove someone allowed to run your twin after you."""
    c = _Ctx(vault, need_model=False)
    m = c.vault.load_meta()
    execs = [e for e in m.executors if e != name] if remove else [*m.executors, name]
    c.vault.update_meta(executors=list(dict.fromkeys(execs)))
    console.print(f"Executors: {', '.join(execs) or '(none)'}")


@legacy_app.command("posthumous")
def legacy_posthumous(
    allow: bool = typer.Argument(..., help="true / false"), vault: Path | None = VaultOpt
) -> None:
    """Allow or forbid others talking to the twin (visitor mode)."""
    c = _Ctx(vault, need_model=False)
    c.vault.update_meta(allow_posthumous=allow)
    console.print(f"Visitor access: {'allowed' if allow else 'disabled'}")


@legacy_app.command("off-switch")
def legacy_off_switch(
    engage: bool = typer.Argument(..., help="true to silence the twin for good"),
    vault: Path | None = VaultOpt,
) -> None:
    """Silence the twin entirely. Your memories stay yours; nothing speaks for you."""
    c = _Ctx(vault, need_model=False)
    if engage:
        sure = _ask("This stops the twin answering anyone, including you. Type YES to confirm: ")
        if sure.strip() != "YES":
            console.print("Cancelled.")
            raise typer.Exit()
    c.vault.update_meta(off_switch_engaged=engage)
    console.print("[red]Off switch engaged.[/red]" if engage else "Off switch released.")


@legacy_app.command("disclosure")
def legacy_disclosure(
    text: str = typer.Argument(..., help="Use {owner} for your name"), vault: Path | None = VaultOpt
) -> None:
    """Change the sentence the twin uses to say what it is."""
    c = _Ctx(vault, need_model=False)
    c.vault.update_meta(disclosure=text)
    console.print("Disclosure updated.")


# --------------------------------------------------------------------------
# export / import
# --------------------------------------------------------------------------


@app.command()
def export(vault: Path | None = VaultOpt) -> None:
    """One JSON file with everything. Readable by anyone, forever, with no You.si."""
    c = _Ctx(vault, need_model=False)
    out = c.vault.export_all()
    console.print(f"Exported to [bold]{out}[/bold] ({out.stat().st_size // 1024} KB)")


@app.command("import")
def import_(
    export_file: Path = typer.Argument(..., exists=True), vault: Path | None = VaultOpt
) -> None:
    """Recreate a vault from an export file."""
    settings = load_settings(**({"YOUSI_VAULT": vault} if vault else {}))
    configure_logging(settings.log_level, quiet_console=True)
    try:
        v = Vault.import_from(export_file, settings.vault)
    except (FileExistsError, ValueError) as exc:
        _fail(str(exc))
    console.print(
        f"Imported {v.count()} memories into {v.root}. "
        "Memories are indexed for search the first time you chat or ingest."
    )


@app.command()
def reindex(vault: Path | None = VaultOpt) -> None:
    """Rebuild the search index from the Markdown files alone (recovery tool)."""
    c = _Ctx(vault, need_model=False)
    n = c.vault.rebuild_index()
    embedded = embed_missing(c.vault, LocalEmbedder(c.settings.embed_model))
    console.print(f"Rebuilt index from {n} memory files; embedded {embedded}.")


def main() -> None:
    try:
        app()
    except KeyboardInterrupt:
        err.print("\ninterrupted")
        raise SystemExit(3) from None


if __name__ == "__main__":
    main()
