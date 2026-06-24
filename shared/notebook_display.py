"""Notebook-only helpers for graph visuals and interactive human-in-the-loop demos."""

from __future__ import annotations

import os
from typing import Any


def show_workflow(graph) -> None:
    """Render the compiled LangGraph workflow as a Mermaid PNG in Jupyter."""
    from IPython.display import Image, display

    # Show workflow
    display(Image(graph.get_graph().draw_mermaid_png()))


def ask_approval(payload: Any, *, prompt: str = "Approve? [y/n]: ") -> dict[str, bool]:
    """
    Pause the notebook for a human decision.

    Set BOOTCAMP_SMOKE=1 to auto-approve during automated smoke tests.
    """
    print("HUMAN REVIEW")
    if isinstance(payload, dict):
        for key, value in payload.items():
            print(f"  {key}: {value}")
    else:
        print(f"  payload: {payload}")

    if os.getenv("BOOTCAMP_SMOKE"):
        print("  smoke mode: auto-approved")
        return {"approved": True}

    answer = input(prompt).strip().lower()
    return {"approved": answer in {"y", "yes", "true", "1"}}


def ask_hitl_decision(
    payload: Any,
    *,
    prompt: str = "Approve tool call? [y/n]: ",
) -> dict[str, list[dict[str, str]]]:
    """
    Pause for HumanInTheLoopMiddleware review.

    Returns LangChain resume payload: {"decisions": [{"type": "approve"|"reject"|"edit", ...}]}.
    Set BOOTCAMP_SMOKE=1 to auto-approve during automated smoke tests.
    """
    print("HUMAN REVIEW (HITL middleware)")
    action_requests: list[dict[str, Any]] = []
    if isinstance(payload, dict):
        action_requests = payload.get("action_requests") or []

    if not action_requests:
        print(f"  payload: {payload}")
    for req in action_requests:
        print(f"  tool: {req.get('name')}")
        print(f"  args: {req.get('args')}")
        description = req.get("description")
        if description:
            print(f"  note: {str(description).splitlines()[0]}")

    if os.getenv("BOOTCAMP_SMOKE"):
        print("  smoke mode: auto-approved")
        return {"decisions": [{"type": "approve"}]}

    answer = input(prompt).strip().lower()
    if answer in {"y", "yes", "approve", "a"}:
        return {"decisions": [{"type": "approve"}]}
    if answer in {"e", "edit"} and action_requests:
        req = action_requests[0]
        tool_name = str(req.get("name", ""))
        args = dict(req.get("args") or {})
        if tool_name == "book_trip" and "destination" in args:
            edited = input(f"Edit destination [{args['destination']}]: ").strip()
            if edited:
                args["destination"] = edited
        return {
            "decisions": [{
                "type": "edit",
                "edited_action": {"name": tool_name, "args": args},
            }]
        }
    feedback = input("Rejection reason (optional): ").strip()
    decision: dict[str, str] = {"type": "reject"}
    if feedback:
        decision["feedback"] = feedback
    return {"decisions": [decision]}
