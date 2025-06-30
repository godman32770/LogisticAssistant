from core import RAGState
from utils.vectorstores import model

def supervisor_agent(state: RAGState) -> RAGState:
    # Prepare history string
    history_str = ""
    if "history" in state and state["history"]:
        for msg in state["history"]:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            history_str += f"{role.capitalize()}: {content}\n"

    react_prompt = """
        You are a supervisor agent that helps an assistant solve user queries by breaking them into actionable subgoals.
        For now you can only use inventory/logistic pipelines

        For each subgoal, also decide which pipeline should be used:
        - inventory: for inventory/logistics data
        - tool: for tool/API calls (e.g., weather, calculator)
        - knowledge_search: for knowledge retrieval from documents or web

        Conversation history:
        {history}

        User question:
        {question}

        Your tasks:
        1. Analyze the user question and decompose it into clear, tool-oriented subtasks.
        2. For each subgoal, decide which pipeline is most appropriate.
        3. Output your plan as a JSON list of objects, each with "subgoal" and "pipeline" fields.

        Format:
        [
        {{"subgoal": "Check tomorrow's weather forecast", "pipeline": "tool"}},
        {{"subgoal": "Summarize the latest inventory for China warehouse", "pipeline": "inventory"}},
        {{"subgoal": "Find recent news about supply chain", "pipeline": "knowledge_search"}}
        ]
        """
    prompt = react_prompt.format(
        history=history_str,
        question=state["question"]
    )

    plan_response = model.invoke(prompt)
    import json
    try:
        plan = json.loads(plan_response.content)
        subtasks = plan.get("subtasks", [])
    except Exception:
        subtasks = []

    return {**state, "subtasks": subtasks}

