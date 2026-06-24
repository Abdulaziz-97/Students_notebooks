"""Check that pip installed everything the bootcamp needs.

Run from repo root after: pip install -r requirements.txt
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

THIRD_PARTY = [
    "langchain",
    "langchain_core",
    "langgraph",
    "deepagents",
    "langsmith",
    "pydantic",
    "dotenv",
    "langchain_openai",
    "langchain_tavily",
    "langchain_text_splitters",
    "langchain_huggingface",
    "langchain_community",
    "langchain_chroma",
    "langchain_experimental",
    "langchain_classic",
    "openai",
    "torch",
    "transformers",
    "numpy",
    "rank_bm25",
    "nltk",
    "adaptive_chunking",
    "rich",
    "IPython",
]

LOCAL = [
    "shared.llm",
    "shared.bootcamp_fixtures",
    "shared.dataflow",
    "shared.notebook_display",
    "shared.rag_adaptive",
    "medical_deep_agent.config",
    "medical_deep_agent.workflow",
]


def main() -> int:
    failed: list[str] = []
    for name in THIRD_PARTY + LOCAL:
        try:
            importlib.import_module(name)
            print(f"OK  {name}")
        except Exception as exc:
            print(f"FAIL {name}: {exc}")
            failed.append(name)
    if failed:
        print(f"\n{len(failed)} failure(s)")
        return 1
    print(f"\nAll {len(THIRD_PARTY) + len(LOCAL)} checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
