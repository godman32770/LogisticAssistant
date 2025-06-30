from core import RAGState

def aggregate_contexts(state: RAGState) -> RAGState:
    """
    Aggregate all available context (tool results, knowledge search, structured insights)
    into a single context string for the LLM.
    """
    context_parts = []
    # Add structured insights from logistics/contextualize agent
    if "context_prompt" in state:
        context_parts.append(state["context_prompt"])
        print(f"Context prompt added: {state['context_prompt'][:30]}...")

    # Add tool results
    tool_results = state.get("tool_results", [])
    if tool_results:
        tool_context = "Tool Results:\n"
        for result in tool_results:
            subtask = result.get("subtask", "")
            tool = result.get("tool", "")
            output = result.get("result", "")
            if output:
                tool_context += f"- [{tool}] {subtask}: {output}\n"
        context_parts.append(tool_context)

    # Add knowledge search (RAG) results
    top_docs = state.get("top_docs", [])
    if top_docs:
        rag_context = "Knowledge Search Results:\n"
        for doc in top_docs:
            title = doc.metadata.get("title", "Untitled")
            content = doc.page_content[:500]  # Truncate for brevity
            rag_context += f"- {title}: {content}\n"
        context_parts.append(rag_context)

    # Combine all parts
    full_context = "\n\n".join(context_parts).strip()
    state["context"] = full_context
    print(f"Full context created: {full_context[:300]}...")  # Debugging output
    return state