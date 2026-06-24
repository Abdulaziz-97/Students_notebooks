"""Reusable travel-assistant fixtures for bootcamp notebooks."""

from __future__ import annotations

from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.tools import tool
from pydantic import BaseModel, Field

# ── Knowledge base (used in LangGraph workflow demos) ─────────────────────

TRAVEL_KB: dict[str, str] = {
    "schengen": "Schengen visa: short stays up to 90 days in any 180-day period.",
    "tsa_liquids": "TSA carry-on: liquids must be 3.4oz/100ml in one quart-size bag.",
    "eu261": "EU261: delays over 3 hours on EU carriers may entitle compensation.",
    "lisbon": "Lisbon: mild winters, busy in summer; book tram tickets early.",
}


def search_kb(query: str) -> list[str]:
    """Simple keyword lookup over TRAVEL_KB."""
    q = query.lower()
    hits = [
        text
        for key, text in TRAVEL_KB.items()
        if key in q or any(w in text.lower() for w in q.split())
    ]
    return hits or [f"No KB match for: {query}"]


# ── Agent tools (used in notebooks 03–07, 10–12) ───────────────────────────

class SearchArgs(BaseModel):
    query: str = Field(description="Trip type, e.g. beach or city")
    budget: int = Field(description="Max USD", ge=0)


class WeatherArgs(BaseModel):
    city: str = Field(description="City name")
    date: str = Field(description="Travel date YYYY-MM-DD")


@tool(args_schema=SearchArgs)
def search_destinations(query: str, budget: int) -> str:
    """Search destinations under budget."""
    if "beach" in query.lower():
        return f"Phuket — ${min(budget, 450)}"
    if "city" in query.lower() or "europe" in query.lower():
        return f"Lisbon — ${min(budget, 520)}"
    return "No match — try beach or city"


@tool(args_schema=WeatherArgs)
def check_weather(city: str, date: str) -> str:
    """Check weather for a city on a date."""
    return f"{city} on {date}: 22°C, partly cloudy"


TRAVEL_TOOLS = [search_destinations, check_weather]


# ── RAG corpus loader (notebooks 05–06) ───────────────────────────────────

def load_travel_corpus(root: Path | None = None):
    """Load plain-text travel docs from data/."""
    root = root or Path.cwd()
    if not (root / "shared").exists():
        root = root.parent
    data_dir = root / "data"
    loader = DirectoryLoader(
        str(data_dir),
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=False,
    )
    docs = loader.load()
    print(f"Loaded {len(docs)} documents from {data_dir.name}/")
    return docs


def print_messages(messages, limit: int = 200) -> None:
    """Debug helper: show message types and final content."""
    for i, msg in enumerate(messages):
        print(f"[{i}] {type(msg).__name__}")
    if messages:
        content = getattr(messages[-1], "content", str(messages[-1]))
        print("Final:", content[:limit])
