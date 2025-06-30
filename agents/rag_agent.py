from core import RAGState
from utils.vectorstores import model

def rag_agent(state: RAGState) -> RAGState:
    # List available databases (customize as needed)
    state["pipeline_used"] = "rag"
    databases = "- web_scraper_db (default)\n- another_db (example)\n"

    react_prompt = """
You are an assistant that receives a list of subgoals and decides for each subgoal which database(s) to query and what information to retrieve.

Available databases:
{databases}

Subgoals:
{subgoals}

Output your plan as a JSON object with a field "retrieval_plan" (a list of objects, each with "subgoal", "database", and "what_to_retrieve").

Format example:
{{
  "retrieval_plan": [
    {{"subgoal": "Find the definition of X", "database": "web_scraper_db", "what_to_retrieve": "definition of X"}},
    {{"subgoal": "Get recent news about X", "database": "another_db", "what_to_retrieve": "recent news about X"}}
  ]
}}
"""
    prompt = react_prompt.format(
        databases=databases,
        subgoals="\n".join(f"- {sg}" for sg in state.get("subtasks", []))
    )

    plan_response = model.invoke(prompt)
    import json
    try:
        plan = json.loads(plan_response.content)
        retrieval_plan = plan.get("retrieval_plan", [])
    except Exception:
        retrieval_plan = []

    return {**state, "retrieval_plan": retrieval_plan}


