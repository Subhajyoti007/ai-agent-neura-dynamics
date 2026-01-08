from __future__ import annotations

import types

import pytest

from ai_agent.weather import (
    answer_weather_question,
    build_weather_prompt,
    extract_city_from_question,
    fetch_weather,
    summarize_weather_with_llm,
)
from ai_agent.config import WeatherSettings


def test_extract_city_from_question_basic():
    assert extract_city_from_question("What's the weather in Paris today?", "London") == "Paris"
    assert extract_city_from_question("Tell me the weather in New York.", "London") == "New York"


def test_extract_city_from_question_fallback_to_default():
    assert extract_city_from_question("How's the weather?", "London") == "London"


class DummyResponse:
    def __init__(self, status_code: int, json_data=None, text: str = ""):
        self.status_code = status_code
        self._json_data = json_data or {}
        self.text = text

    def json(self):  # pragma: no cover - trivial
        return self._json_data


def test_fetch_weather_success(monkeypatch):
    def fake_get(url, params=None, timeout=10):  # pragma: no cover - trivial
        return DummyResponse(200, {"weather": [{"description": "clear"}], "main": {"temp": 25}})

    monkeypatch.setattr("ai_agent.weather.requests.get", fake_get)

    ws = WeatherSettings(api_key="dummy", units="metric", default_city="London")
    data = fetch_weather("London", ws)
    assert data["main"]["temp"] == 25


def test_fetch_weather_error_raises(monkeypatch):
    def fake_get(url, params=None, timeout=10):  # pragma: no cover - trivial
        return DummyResponse(404, text="not found")

    monkeypatch.setattr("ai_agent.weather.requests.get", fake_get)

    ws = WeatherSettings(api_key="dummy", units="metric", default_city="London")
    with pytest.raises(RuntimeError):
        fetch_weather("Nowhere", ws)


def test_summarize_weather_with_llm_uses_llm():
    class FakeLLM:
        def __init__(self):
            self.last_prompt = None

        def invoke(self, prompt):
            self.last_prompt = prompt
            return types.SimpleNamespace(content="It is sunny and 25°C.")

    llm = FakeLLM()
    question = "What's the weather in Paris?"
    weather_json = {"weather": [{"description": "clear"}], "main": {"temp": 25}}
    answer = summarize_weather_with_llm(question, weather_json, llm)

    assert "sunny" in answer or "25" in answer or "°" in answer
    assert question in llm.last_prompt


def test_answer_weather_question_high_level(monkeypatch):
    # Avoid real HTTP and LLM calls

    def fake_fetch_weather(city, settings):  # pragma: no cover - trivial
        return {"weather": [{"description": "rain"}], "main": {"temp": 10}}

    class FakeLLM:
        def invoke(self, prompt):  # pragma: no cover - trivial
            return types.SimpleNamespace(content=f"Fake answer about {prompt}")

    monkeypatch.setattr("ai_agent.weather.fetch_weather", fake_fetch_weather)

    ws = WeatherSettings(api_key="dummy", units="metric", default_city="London")
    answer, raw = answer_weather_question("What's the weather in London?", weather_settings=ws, llm=FakeLLM())

    assert "Fake answer" in answer
    assert raw["main"]["temp"] == 10
