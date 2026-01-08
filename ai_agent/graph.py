from __future__ import annotations

from typing import List, Literal, TypedDict

from langgraph.graph import END, StateGraph

from .config import get_settings
from .llm import get_chat_model, get_embedding_model
from .rag import answer_with_rag, get_retriever
from .weather import answer_weather_question


class AgentState(TypedDict, total=False):
    """LangGraph state shared between nodes."""

    question: str
    route: Literal["weather", "rag"]
    answer: str
    context_docs: List[dict]


def _route_question(question: str) -> Literal["weather", "rag"]:
    """Very simple router heuristic.

    This could be replaced by an LLM router, but for the assignment a rules-based
    router keeps behavior deterministic and easy to test.
    """

    q = question.lower()
    weather_keywords = [
        "weather",
        "temperature",
        "rain",
        "forecast",
        "humidity",
        "wind",
        "snow",
        "sunny",
        "cloudy",
    ]

    if any(word in q for word in weather_keywords):
        return "weather"
    return "rag"


def build_agent_graph():
    """Build and compile the LangGraph agent.

    Nodes:
      - router: decides between weather vs RAG.
      - weather: calls OpenWeatherMap and summarizes via LLM.
      - rag: runs RAG over the configured PDF using Qdrant.
    """

    settings = get_settings()
    llm = get_chat_model()
    embeddings = get_embedding_model()

    workflow = StateGraph(AgentState)

    # --- Node definitions -------------------------------------------------

    def router_node(state: AgentState) -> AgentState:
        question = state["question"]

        # Prefer an LLM-based router to distinguish weather vs PDF questions,
        # but fall back to a simple keyword heuristic for robustness and tests.
        try:
            routing_prompt = (
                "You are a router deciding whether a user question is about real-time weather "
                "or about the contents of a static PDF document.\n\n"
                "Return exactly one word: 'weather' or 'rag'.\n\n"
                f"Question: {question}"
            )
            result = llm.invoke(routing_prompt)
            content = getattr(result, "content", str(result)).strip().lower()

            if "weather" in content and "rag" in content:
                # Ambiguous model output; use heuristic.
                route = _route_question(question)
            elif "weather" in content:
                route = "weather"
            elif "rag" in content:
                route = "rag"
            else:
                route = _route_question(question)
        except Exception:
            route = _route_question(question)

        return {"route": route}

    def weather_node(state: AgentState) -> AgentState:
        question = state["question"]
        answer, raw_weather = answer_weather_question(
            question,
            weather_settings=settings.weather,
            llm=llm,
        )
        # Store raw weather JSON as a single "doc" for transparency.
        return {
            "answer": answer,
            "route": "weather",
            "context_docs": [raw_weather],
        }

    def rag_node(state: AgentState) -> AgentState:
        question = state["question"]
        retriever = get_retriever(pdf_settings=settings.pdf, embeddings=embeddings)
        answer, docs = answer_with_rag(question, llm, retriever)

        # Convert Document objects to simple dicts for easier serialization/inspection.
        serialized_docs = [
            {"page_content": d.page_content, "metadata": d.metadata} for d in docs
        ]

        return {
            "answer": answer,
            "route": "rag",
            "context_docs": serialized_docs,
        }

    # --- Graph wiring -----------------------------------------------------

    workflow.add_node("router", router_node)
    workflow.add_node("weather", weather_node)
    workflow.add_node("rag", rag_node)

    workflow.set_entry_point("router")

    def route_decider(state: AgentState) -> str:
        return state.get("route", "rag")

    workflow.add_conditional_edges(
        "router",
        route_decider,
        {
            "weather": "weather",
            "rag": "rag",
        },
    )

    workflow.add_edge("weather", END)
    workflow.add_edge("rag", END)

    return workflow.compile()
