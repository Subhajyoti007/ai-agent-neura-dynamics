from __future__ import annotations

from typing import Any, Dict, Optional

from langsmith import Client


def log_response_for_evaluation(
    question: str,
    answer: str,
    *,
    route: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Create a LangSmith example for later evaluation.

    If LANGSMITH_API_KEY is not configured, this becomes a no-op.
    """

    try:
        client = Client()
    except Exception:
        # If LangSmith is not configured, fail silently for convenience.
        return

    meta: Dict[str, Any] = {"route": route}
    if metadata:
        meta.update(metadata)

    try:
        client.create_example(
            inputs={"question": question},
            outputs={"answer": answer},
            metadata=meta,
        )
    except Exception:
        # Do not break the main app if LangSmith logging fails.
        return
