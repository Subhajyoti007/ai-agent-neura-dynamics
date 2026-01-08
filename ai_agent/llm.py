from __future__ import annotations

from functools import lru_cache

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from .config import get_settings


@lru_cache
def get_chat_model() -> ChatOpenAI:
    """Return a cached ChatOpenAI instance configured from settings."""

    settings = get_settings()
    return ChatOpenAI(
        model=settings.openai.model_name,
        api_key=settings.openai.api_key,
        temperature=0.0,
    )


@lru_cache
def get_embedding_model() -> OpenAIEmbeddings:
    """Return a cached OpenAIEmbeddings instance used for vector storage."""

    settings = get_settings()
    # Use a modern embedding model; adjust if your account uses a different default.
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=settings.openai.api_key,
    )
