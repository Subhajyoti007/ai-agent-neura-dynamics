from __future__ import annotations

import streamlit as st

from ai_agent import build_agent_graph
from ai_agent.evaluation import log_response_for_evaluation


st.set_page_config(page_title="AI Agent Assignment", page_icon="🤖", layout="centered")

st.title("AI Agent: Weather & PDF RAG")
st.write(
    "Ask questions about the **weather** or about the content of the configured **PDF document**.\n"
    "The agent will decide whether to call the weather API or use RAG over the PDF."
)


if "graph" not in st.session_state:
    st.session_state.graph = build_agent_graph()

if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


user_input = st.chat_input("Type your question here...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            graph = st.session_state.graph
            state = graph.invoke({"question": user_input})

            answer = state.get("answer", "I was unable to produce an answer.")
            route = state.get("route", "unknown")
            context_docs = state.get("context_docs", [])

            st.markdown(answer)
            st.caption(f"Route: **{route}**")

            # with st.expander("Debug: context / raw data"):
            #     st.json(context_docs)

            # Log to LangSmith (no-op if not configured)
            try:
                log_response_for_evaluation(
                    question=user_input,
                    answer=answer,
                    route=route,
                    metadata={"context_docs": context_docs},
                )
            except Exception:
                # Never break the UI due to evaluation logging issues.
                pass

        st.session_state.messages.append({"role": "assistant", "content": answer})
