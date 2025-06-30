from core import RAGState

def contextualize_agent(state: RAGState) -> RAGState:

    """
    Takes structured insights from logistics_data_agent and formats them as context for the LLM.
    """
    # Get the structured insights from state
    insights = state.get("structured_insights", "")
    # Optionally, add more formatting or additional context logic here
    context_prompt = (
        "Here is the latest inventory and logistics summary for your reference:\n\n"
        f"{insights}\n"
        "Use this information to help answer any relevant user questions."
    )
    print(f"Context prompt created: {context_prompt[:300]}...")  # Debugging output
    # Add the formatted context to the state
    state["context_prompt"] = context_prompt
    state["pipeline_used"] = "inventory"
    return state