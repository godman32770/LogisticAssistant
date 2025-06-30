
from core import RAGState
from utils.vectorstores import model
from langchain_core.prompts import ChatPromptTemplate


template = """
You are an assistant for question-answering tasks. Use the following retrieved context to answer the question. If you don't know the answer, just say that you don't know.

Previous conversation: {history}
Question: {question}
Context: {context}

**Reasoning:**
Think step by step based on the retrieved context. Explain your thought process clearly.

**Answer:**
Provide the final answer based on the context you retrieved.

**Token Usage:**
Please provide the following information at the end of your response:
- Input tokens used: [count the tokens in the question, context, and history]
- Output tokens generated: [count the tokens in your response]
- Total tokens: [sum of input and output tokens]
- Estimated cost: [if applicable, based on your model's pricing]

Note: Include actual token counts in your response, not placeholder text.
"""


# ✅ Agent 3: Generate Answer
def generate_answer(state: RAGState) -> RAGState:
    max_context_chars = 6000  # adjust as needed
    context = ""
    pipeline = state.get("pipeline_used", "")
    # print(f"Pipeline used: {pipeline}")

    if pipeline == "inventory":
        # Prefer structured_insights or context_prompt
        context = state.get("structured_insights") or state.get("context_prompt", "")

    elif pipeline == "rag":
        # Use top_docs as before
        for doc in state.get("top_docs", []):
            if len(context) + len(doc.page_content) > max_context_chars:
                break
            context += doc.page_content + "\n\n"
    elif pipeline == "tool":
        # Optionally, summarize tool_results for context
        tool_results = state.get("tool_results", [])
        if tool_results:
            context = "\n".join(
                f"{tr.get('tool', '')}: {tr.get('result', '')}" for tr in tool_results
            )
    else:
        # Fallback: try context_prompt or top_docs
        context = state.get("context_prompt", "")
        if not context:
            for doc in state.get("top_docs", []):
                if len(context) + len(doc.page_content) > max_context_chars:
                    break
                context += doc.page_content + "\n\n"

    # Format history as a string
    history_str = ""
    if "history" in state and state["history"]:
        for msg in state["history"]:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            history_str += f"{role.capitalize()}: {content}\n"

    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    response = chain.invoke({
        "question": state["question"],
        "context": context,
        "history": history_str
    })
    return {**state, "answer": response.content.strip()}