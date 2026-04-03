import spacy
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from spacy_summarize_core import summarize_text

DEMO_TEXT = """
The Federal Reserve raised its benchmark interest rate by a quarter point on Wednesday,
the tenth increase since March 2022. Policymakers said inflation remains elevated and
further rate increases may be necessary to bring it back to the 2 percent target. The
labor market has remained resilient despite tighter financial conditions, with
unemployment holding near historic lows. Chair Jerome Powell said the committee would
proceed carefully and remains data-dependent. Inflation has slowed from its 2022 peak
but continues to run above levels the Fed considers acceptable. Rate-sensitive sectors
like housing have already felt the impact, with mortgage rates doubling over the past
two years.
"""

console = Console()


def main() -> None:
    console.print("\n[bold cyan]Loading spaCy model...[/bold cyan]")
    nlp = spacy.load("en_core_web_sm")

    result = summarize_text(
        DEMO_TEXT, nlp, summary_sentence_count=2, top_keyword_count=5
    )

    if result.error:
        console.print(f"[red]{result.error}[/red]")
        return

    console.print(f"[green]✓[/green] Processed {result.num_sentences} sentences")

    table = Table(title="Top Keywords", show_header=True, header_style="bold magenta")
    table.add_column("Rank", style="dim", width=6)
    table.add_column("Keyword", style="cyan")
    table.add_column("Frequency", justify="right", style="green")
    for idx, (word, count) in enumerate(result.top_keywords, 1):
        table.add_row(str(idx), word, str(count))
    console.print(table)

    console.print("\n[bold cyan]Sentence importance scores...[/bold cyan]")
    scores_table = Table(
        title="Sentence Importance Scores",
        show_header=True,
        header_style="bold magenta",
    )
    scores_table.add_column("#", style="dim", width=4)
    scores_table.add_column("Score", justify="right", style="yellow", width=8)
    scores_table.add_column("Sentence Preview", style="white")
    for idx, (sent, score) in enumerate(result.sentence_scores, 1):
        preview = f"{sent[:80]}..." if len(sent) > 80 else sent
        scores_table.add_row(str(idx), f"{score:.2f}", preview)
    console.print(scores_table)

    console.print(
        Panel.fit(
            result.summary,
            title="[bold green]Summary[/bold green]",
            border_style="green",
            padding=(1, 2),
        )
    )


if __name__ == "__main__":
    main()
