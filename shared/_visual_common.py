"""
Shared visual helpers used by every module's _visual.py.

Each module extends this with module-specific display functions.
Falls back to plain-text helpers from shared/utils.py when Rich is unavailable.
"""

try:
    from rich.panel import Panel

    from shared._visual_base import console, safe

    RICH = True
except ImportError:
    RICH = False
    from shared.utils import print_section, print_step


def show_section(title: str) -> None:
    """Bordered section header."""
    if RICH:
        console.print()
        console.print(Panel(safe(title), border_style="section"))
    else:
        print_section(title)


def show_step(n: int, text: str) -> None:
    """Numbered step label."""
    if RICH:
        console.print(f"  [step]Step {n}[/] — [dim]{safe(text)}[/]")
    else:
        print_step(n, text)


def show_agent_response(text: str, max_len: int = 500) -> None:
    """Agent's final text response."""
    preview = safe(str(text))[:max_len]
    if RICH:
        console.print(Panel(preview, border_style="agent_text", title="Agent", padding=(1, 2)))
    else:
        print(f"\n  [AGENT] {preview}")
