"""Reference implementation — notebook 07 builds the same graph inline in cells."""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from shared.bootcamp_fixtures import search_kb
from shared.llm import get_model

__all__ = ["TravelSupportState", "TravelClassification", "empty_state", "build_support_graph"]


class TravelSupportState(TypedDict):
    request: str
    traveler_profile: dict
    classification: dict
    search_results: Annotated[list[str], operator.add]
    draft_reply: str
    risk_flags: Annotated[list[str], operator.add]
    needs_human: bool
    final_reply: str


class TravelClassification(BaseModel):
    intent: str = Field(description="support, booking, or general")
    urgency: Literal["low", "normal", "high"] = "normal"
    topic: str = Field(description="docs, booking, or itinerary")
    needs_human: bool = False


def empty_state(**overrides) -> TravelSupportState:
    base: TravelSupportState = {
        "request": "",
        "traveler_profile": {},
        "classification": {},
        "search_results": [],
        "draft_reply": "",
        "risk_flags": [],
        "needs_human": False,
        "final_reply": "",
    }
    base.update(overrides)
    return base


def build_support_graph():
    model = get_model()
    classifier = model.with_structured_output(TravelClassification, method="function_calling")

    def classify_intent(state: TravelSupportState):
        data = classifier.invoke([
            SystemMessage(content="You classify travel support tickets."),
            HumanMessage(content=state["request"]),
        ]).model_dump()
        return {"classification": data, "needs_human": data["needs_human"]}

    def draft_reply(state: TravelSupportState):
        facts = "; ".join(state["search_results"][:3]) or "No KB facts."
        text = model.invoke([
            SystemMessage(content="Draft concise travel support replies using provided facts."),
            HumanMessage(content=f"Request: {state['request']}\nFacts: {facts}"),
        ]).content
        return {"draft_reply": str(text).strip()}

    g = StateGraph(TravelSupportState)
    g.add_node("read_request", lambda s: {"request": s["request"], "traveler_profile": s.get("traveler_profile") or {}})
    g.add_node("classify", classify_intent)
    g.add_node("doc_search", lambda s: {"search_results": search_kb(s["request"])})
    g.add_node("draft_reply", draft_reply)
    g.add_node("send_reply", lambda s: {"final_reply": s["draft_reply"]})
    g.add_edge(START, "read_request")
    g.add_edge("read_request", "classify")
    g.add_edge("classify", "doc_search")
    g.add_edge("doc_search", "draft_reply")
    g.add_edge("draft_reply", "send_reply")
    g.add_edge("send_reply", END)
    return g.compile()
