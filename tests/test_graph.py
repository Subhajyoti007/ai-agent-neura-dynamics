from __future__ import annotations

from ai_agent.graph import _route_question


def test_route_question_weather_keywords():
    assert _route_question("What's the weather in London?") == "weather"
    assert _route_question("Tell me the temperature in Paris") == "weather"
    assert _route_question("Will it rain tomorrow?") == "weather"


def test_route_question_default_to_rag():
    assert _route_question("Summarize the introduction of the PDF.") == "rag"
    assert _route_question("What is the main idea of the document?") == "rag"
