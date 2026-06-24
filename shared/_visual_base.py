"""
Shared Rich Console and color palette for all visualizers.

Each module's _visual.py imports from here to avoid duplicating
console setup, color themes, and encoding handling.
"""

from rich.console import Console
from rich.theme import Theme

from shared.text import safe

# ---- Color palette ----
THEME = Theme({
    "tool_name": "bold yellow",
    "tool_args": "dim yellow",
    "tool_result": "cyan",
    "agent_text": "green",
    "human_text": "white",
    "system_text": "dim",
    "step": "bold blue",
    "section": "bold magenta",
    "interrupt": "bold red",
    "resume": "bold green",
    "error": "bold red",
    "success": "bold green",
    "progress": "yellow",
    "pending": "dim",
    "in_progress": "bold yellow",
    "completed": "green",
    "graph_edge": "dim cyan",
    "graph_node": "bold cyan",
    "code": "bright_black",
    "highlight": "bold white",
})

# Force narrow width and ASCII-safe rendering for Windows terminal compatibility.
console = Console(
    theme=THEME,
    width=100,
    force_terminal=True,
    legacy_windows=False,
)
