from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()


@dataclass
class OpenAISettings:
    api_key: str
    model_name: str = "gpt-4o-mini"


@dataclass
class WeatherSettings:
    api_key: str
    base_url: str = "https://api.openweathermap.org/data/2.5/weather"
    units: str = "metric"
    default_city: str = "London"


@dataclass
class PDFSettings:
    pdf_path: str
    collection_name: str = "pdf_documents"


@dataclass
class Settings:
    openai: OpenAISettings
    weather: WeatherSettings
    pdf: PDFSettings


@lru_cache
def get_settings() -> Settings:
    """Return global application settings loaded from env vars.

    Raises a RuntimeError if mandatory settings like OPENAI_API_KEY are missing.
    """

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Please configure it in your environment or .env file.")

    openai_model = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")

    weather_api_key = os.getenv("OPENWEATHER_API_KEY")
    weather_units = os.getenv("OPENWEATHER_UNITS", "metric")
    weather_default_city = os.getenv("OPENWEATHER_DEFAULT_CITY", "London")

    pdf_path = os.getenv("PDF_PATH")
    if not pdf_path:
        raise RuntimeError("PDF_PATH is not set. Provide a path to the PDF used for RAG.")
    pdf_collection_name = os.getenv("PDF_COLLECTION_NAME", "pdf_documents")

    return Settings(
        openai=OpenAISettings(api_key=openai_api_key, model_name=openai_model),
        weather=WeatherSettings(
            api_key=weather_api_key or "",
            units=weather_units,
            default_city=weather_default_city,
        ),
        pdf=PDFSettings(pdf_path=pdf_path, collection_name=pdf_collection_name),
    )
