"""Readable DATAFLOW traces for agent and LangGraph notebook demos."""

from __future__ import annotations

import sys
from typing import Any

__all__ = [
    "preview",
    "print_dataflow",
    "print_agent_dataflow",
    "print_rag_dataflow",
    "print_final_state",
]


def _emit(line: str) -> None:
    """Print safely on Windows consoles that cannot encode all Unicode."""
    try:
        print(line)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        sys.stdout.buffer.write((line + "\n").encode(encoding, errors="replace"))


def preview(value: Any, limit: int = 180) -> str:
    """Single-line preview for long LLM or state values."""
    text = str(value).replace("\n", " ")
    return text[:limit] + ("..." if len(text) > limit else "")


def print_dataflow(graph, inputs: dict, **stream_kwargs) -> dict:
    """Stream node updates once — shows how state moves through a compiled graph."""
    _emit("DATAFLOW")
    _emit(f"0. input: {preview(inputs)}")
    step = 1
    last_state = dict(inputs)
    for mode, chunk in graph.stream(inputs, stream_mode=["updates", "values"], **stream_kwargs):
        if mode == "updates":
            for node, delta in chunk.items():
                if not delta:
                    _emit(f"{step}. {node}: (route/no-op)")
                    step += 1
                    continue
                brief = {k: preview(v) for k, v in delta.items()}
                _emit(f"{step}. {node}: {brief}")
                step += 1
        elif mode == "values":
            last_state = chunk
    _emit(f"{step}. final: {preview(last_state)}")
    return last_state


def print_agent_dataflow(messages) -> None:
    """Print message/tool trace for create_agent or ToolNode loops."""
    _emit("DATAFLOW")
    for i, m in enumerate(messages):
        role = type(m).__name__
        tool_calls = getattr(m, "tool_calls", None) or []
        if tool_calls:
            calls = []
            for tc in tool_calls:
                if isinstance(tc, dict):
                    calls.append(
                        f"{tc.get('name', '?')}({preview(tc.get('args', tc.get('arguments', '')))})"
                    )
                else:
                    calls.append(getattr(tc, "name", str(tc)))
            _emit(f"[{i}] {role} tool_calls: {calls}")
        else:
            name = getattr(m, "name", "")
            label = f"{role}({name})" if name else role
            _emit(f"[{i}] {label}: {preview(getattr(m, 'content', m))}")
    if messages:
        _emit(f"FINAL: {preview(getattr(messages[-1], 'content', messages[-1]))}")


def print_rag_dataflow(question: str, retrieved: str, answer: str | None = None) -> None:
    """Print retrieval-then-generate trace for RAG agent cells."""
    _emit("DATAFLOW")
    _emit(f"1. question: {preview(question)}")
    _emit(f"2. retrieve: {preview(retrieved)}")
    if answer is not None:
        _emit(f"3. generate: {preview(answer)}")


def print_final_state(state: dict, keys: list[str] | None = None) -> None:
    """Print selected final state fields after a workflow completes."""
    _emit("FINAL STATE")
    selected = keys or list(state.keys())
    for key in selected:
        if key.startswith("__"):
            continue
        _emit(f"  {key}: {preview(state.get(key))}")
