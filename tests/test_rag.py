from __future__ import annotations

import types

from langchain_core.documents import Document

import ai_agent.rag as rag_module
from ai_agent.rag import answer_with_rag


class FakeChain:
    def __init__(self, to_return):
        self.to_return = to_return
        self.last_input = None

    def invoke(self, input_dict):  # pragma: no cover - trivial
        self.last_input = input_dict
        return self.to_return


def test_answer_with_rag_uses_build_rag_chain(monkeypatch):
    docs = [Document(page_content="Test content", metadata={"page": 1})]

    fake_result = "Test answer"
    fake_chain = FakeChain(fake_result)

    def fake_build_rag_chain(llm, retriever):  # pragma: no cover - trivial
        return fake_chain

    monkeypatch.setattr(rag_module, "build_rag_chain", fake_build_rag_chain)

    class FakeLLM:
        pass

    class FakeRetriever:
        def invoke(self, query):  # pragma: no cover - trivial
            return docs

    answer, context_docs = answer_with_rag("What is in the doc?", FakeLLM(), FakeRetriever())

    assert answer == "Test answer"
    assert len(context_docs) == 1
    assert isinstance(context_docs[0], Document)
