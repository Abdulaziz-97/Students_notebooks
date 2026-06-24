"""ASCII-safe text for terminal output on Windows."""


def safe(text: str) -> str:
    """Strip characters that can't render in Windows CP1252 terminal."""
    return text.encode("ascii", errors="replace").decode("ascii")
