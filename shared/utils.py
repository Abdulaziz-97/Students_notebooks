"""Display helpers for clean terminal output during the bootcamp."""

from shared.text import safe as _safe


def print_section(title: str, width: int = 60):
    """Print a titled section divider."""
    print()
    print("=" * width)
    print(f"  {_safe(title)}")
    print("=" * width)
    print()


def print_step(step: int, description: str):
    """Print a numbered step."""
    print(f"\n--- Step {step}: {_safe(description)} ---")


def print_messages(messages: list, label: str = "Messages"):
    """Pretty-print a list of LangChain messages."""
    print(f"\n{label}:")
    for msg in messages:
        role = getattr(msg, "type", getattr(msg, "role", "unknown"))
        content = getattr(msg, "content", str(msg))
        if isinstance(content, list):
            content = content[0].get("text", str(content)) if content else ""
        content_preview = _safe(str(content)[:120])
        print(f"  [{role}] {content_preview}")
