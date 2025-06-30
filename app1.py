import streamlit as st
from Langgraph import rag_chain

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("LANGGRAPH QA System (Minimal)")

# Display chat history
for msg in st.session_state.chat_history:
    role = msg["role"]
    content = msg["content"]
    st.chat_message(role).write(content)

# Chat Input
question = st.chat_input("Ask a question...")

if question:
    # Add user message to history
    st.session_state.chat_history.append({"role": "user", "content": question})
    st.chat_message("user").write(question)

    with st.spinner("Processing your question..."):
        result = rag_chain.invoke({
            "question": question,
            "history": st.session_state.chat_history
        })

    # Add assistant message to history
    st.session_state.chat_history.append({"role": "assistant", "content": result["answer"]})
    st.chat_message("assistant").write(result["answer"])

    # Show structured insights or context if available
    if "structured_insights" in result:
        st.markdown("#### Inventory/Logistics Summary:")
        st.markdown(result["structured_insights"])
    elif "context_prompt" in result:
        st.markdown("#### Context:")
        st.markdown(result["context_prompt"])

    # Show sources if available
    if "top_docs" in result and result["top_docs"]:
        st.markdown("#### แหล่งข้อมูลที่เกี่ยวข้อง:")
        for doc in result["top_docs"]:
            title = doc.metadata.get("title", "ไม่ทราบชื่อเรื่อง")
            url = doc.metadata.get("source", "")
            if url:
                st.markdown(f"- [{title}]({url})")
            else:
                st.markdown(f"- {title}")