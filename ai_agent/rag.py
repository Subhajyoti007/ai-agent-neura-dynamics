from __future__ import annotations

from typing import List, Tuple

import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams
from langchain_qdrant import QdrantVectorStore as Qdrant

from .config import PDFSettings, get_settings


_VECTORSTORE: Qdrant | None = None
_RETRIEVER = None


def load_pdf_documents(pdf_path: str) -> List[Document]:
    """Load a PDF into a list of Documents using LangChain's PyPDFLoader."""

    loader = PyPDFLoader(pdf_path)
    return loader.load()


def create_vector_store(
    documents: List[Document],
    embeddings: Embeddings,
    collection_name: str,
) -> Qdrant:
    """Create a persistent Qdrant vector store from documents.

    Uses a local on-disk Qdrant instance by default. Override path via QDRANT_PATH.
    """

    # Use a local on-disk Qdrant instance by default for persistence.
    # You can override the path by setting the QDRANT_PATH environment variable.
    qdrant_path = os.getenv("QDRANT_PATH", "qdrant_local")
    client = QdrantClient(path=qdrant_path)
    
    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1536, distance="Cosine"),
        )
        print(f"Created collection: {collection_name}")
    except Exception as e:
        print(f"Error creating collection: {e}")
        print(f"Or the collection {collection_name} already exists")
    
    vector_store = Qdrant(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )
    vector_store.add_documents(documents)
    return vector_store


def ensure_vector_store(
    pdf_settings: PDFSettings | None = None,
    embeddings: Embeddings | None = None,
) -> Qdrant:
    """Create (or reuse) the global Qdrant vector store for the PDF."""

    from .llm import get_embedding_model  # local import to avoid cycles

    global _VECTORSTORE

    if _VECTORSTORE is not None:
        return _VECTORSTORE

    settings = get_settings() if pdf_settings is None else None
    ps = pdf_settings or settings.pdf

    docs = load_pdf_documents(ps.pdf_path)
    embs = embeddings or get_embedding_model()

    _VECTORSTORE = create_vector_store(docs, embs, ps.collection_name)
    return _VECTORSTORE


def get_retriever(pdf_settings: PDFSettings | None = None, embeddings: Embeddings | None = None):
    """Return a retriever backed by the global Qdrant vector store."""

    global _RETRIEVER

    if _RETRIEVER is not None:
        return _RETRIEVER

    store = ensure_vector_store(pdf_settings=pdf_settings, embeddings=embeddings)
    _RETRIEVER = store.as_retriever()
    return _RETRIEVER


def build_rag_chain(
    llm: BaseLanguageModel,
    retriever,
):
    """Create a standard RAG chain using LangChain LCEL: retrieve docs then stuff into the LLM."""

    prompt = ChatPromptTemplate.from_template(
        """You are a helpful assistant. Use the provided context to answer the question.
If the answer is not contained in the context, say you don't know.

Context:
{context}

Question: {question}
"""
    )

    rag_chain = (
        RunnableParallel({"context": retriever | _format_docs, "question": RunnablePassthrough()})
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain


def _format_docs(docs):
    """Helper to concatenate document page contents for the prompt."""
    return "\n\n".join(doc.page_content for doc in docs)


def answer_with_rag(
    question: str,
    llm: BaseLanguageModel,
    retriever,
) -> Tuple[str, List[Document]]:
    """Run a full RAG query and return (answer, context_docs)."""

    chain = build_rag_chain(llm, retriever)
    answer = chain.invoke(question)

    # Retrieve the docs used for context (run retriever separately)
    context_docs = retriever.invoke(question)

    return answer, context_docs
