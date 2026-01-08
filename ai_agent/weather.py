from __future__ import annotations

import json
import re
from typing import Any, Dict, Tuple

import requests

from .config import WeatherSettings, get_settings


def extract_city_from_question(question: str, default_city: str) -> str:
    """Very simple heuristic to guess the city from the question.

    If no city is detected, fall back to the default city.
    """

    # Look for patterns like "weather in London" or "in New York today"
    # Stop at time words or punctuation.
    match = re.search(r"in ([A-Za-z ]+?)(?:\s+(?:today|tomorrow|now|right now)|\?|\.|,|!|$)", question, flags=re.IGNORECASE)
    if match:
        city = match.group(1).strip()
        if city:
            return city
    return default_city


def fetch_weather(city: str, settings: WeatherSettings) -> Dict[str, Any]:
    """Call the OpenWeatherMap API and return the JSON payload.

    Raises RuntimeError on non-200 responses.
    """

    if not settings.api_key:
        raise RuntimeError("OPENWEATHER_API_KEY is not set.")

    params = {
        "q": city,
        "appid": settings.api_key,
        "units": settings.units,
    }
    response = requests.get(settings.base_url, params=params, timeout=10)

    if response.status_code != 200:
        raise RuntimeError(
            f"Weather API error: {response.status_code} {response.text}"  # pragma: no cover - message text
        )

    return response.json()


def build_weather_prompt(question: str, weather_json: Dict[str, Any]) -> str:
    """Create a natural-language prompt that includes raw weather JSON."""

    return (
        "You are a helpful weather assistant. "
        "Use only the JSON weather data below to answer the user's question.\n\n"
        f"User question: {question}\n\n"
        "Weather data (OpenWeatherMap JSON):\n"
        f"{json.dumps(weather_json, indent=2)}\n\n"
        "Provide a concise, user-friendly answer, including temperature, conditions, and any other important details."
    )


def summarize_weather_with_llm(question: str, weather_json: Dict[str, Any], llm) -> str:
    """Use the provided LLM to summarize weather information.

    `llm` is expected to implement `.invoke(prompt)` and return an object with
    a `.content` attribute or a plain string.
    """

    prompt = build_weather_prompt(question, weather_json)
    result = llm.invoke(prompt)
    return getattr(result, "content", str(result))


def answer_weather_question(
    question: str,
    *,
    weather_settings: WeatherSettings | None = None,
    llm=None,
) -> Tuple[str, Dict[str, Any]]:
    """High-level helper used by the LangGraph node.

    Returns a pair of (answer_text, raw_weather_json).
    """

    from .llm import get_chat_model  # local import to avoid cycles

    settings = get_settings() if weather_settings is None else None
    ws = weather_settings or settings.weather

    city = extract_city_from_question(question, default_city=ws.default_city)
    print(f"Extracted city: {city}")
    weather_json = fetch_weather(city, ws)

    model = llm or get_chat_model()
    answer_text = summarize_weather_with_llm(question, weather_json, model)

    return answer_text, weather_json
