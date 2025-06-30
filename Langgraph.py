
from dotenv import load_dotenv
from typing import List, Dict, TypedDict
from langchain_core.documents import Document
from agents.supervisor_agent import supervisor_agent
# from agents.mcp_agent import mcp_agent
# from agents.tool_agent import tool_executor_agent
from langgraph.graph import StateGraph, END
from agents.supervisor_agent import supervisor_agent
# from agents.rag_agent import rag_agent
# from agents.grade_docs import grade_docs
from agents.generate_answer import generate_answer
from utils.vectorstores import vector_store, model
from agents.contextualize_agent import contextualize_agent
from agents.aggregate_contexts import aggregate_contexts
from openai import OpenAI
from agents.data_loader import load_inventory_data
from agents.logistic_data import logistics_data_agent
# Load environment variables
load_dotenv()


class RAGState(TypedDict):
    question: str
    query: str
    retrieved_docs: List[Document]
    top_docs: List[Document]
    answer: str
    history: List[Dict[str, str]]
    subtasks: List[str]
    retrieval_plan: List[Dict]
    tool_results: List[Dict]
    tool_plan: List[Dict]
    context_prompt: str        # <-- Add this for contextualize_agent
    context: str               # <-- Add this for aggregate_contexts


# ✅ Agent 1: Generate Query
def generate_query(state: RAGState) -> RAGState:
    prompt = f"Rephrase this into a search query: {state['question']}"
    response = model.invoke(prompt)
    return {**state, "query": response.content.strip()}


client = OpenAI()

def classify_intent(query):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "Based on the user query, determine which tool needs to be used.\n"
                    "Respond with one of the following exact labels only:\n"
                    "- inventory → for queries about stock, materials, warehouse, logistics, etc.\n"
                    "- none → if no tool is required."
                )
            },
            {"role": "user", "content": query}
        ]
    )
    intent = response.choices[0].message.content.strip().lower()
    print(f"[Tool Decision] Query: '{query}' → Tool: {intent}")
    return intent


def should_load_inventory(state):
    return "load_inventory" if classify_intent(state["query"]) == "inventory" else "skip_inventory"
def should_use_tools(state):
    return "use_tools" if classify_intent(state["query"]) == "tool_use" else "skip_tools"
def should_use_rag(state):
    return "use_rag" if classify_intent(state["query"]) == "knowledge_search" else "skip_rag"



# Create the graph
graph = StateGraph(RAGState)

# Add all nodes
graph.add_node("generate_query", generate_query)
graph.add_node("supervisor_agent", supervisor_agent)
# graph.add_node("mcp_agent", mcp_agent)
# graph.add_node("tool_executor_agent", tool_executor_agent)
# graph.add_node("rag_agent", rag_agent)
# graph.add_node("grade_documents", grade_docs)
graph.add_node("load_inventory_data", load_inventory_data)
graph.add_node("logistics_data_agent", logistics_data_agent)
graph.add_node("contextualize_agent", contextualize_agent)
graph.add_node("aggregate_contexts", aggregate_contexts)  # New node to combine all contexts
graph.add_node("generate_answer", generate_answer)

# Set entry point with user query
graph.set_entry_point("generate_query")

# Initial flow: query -> supervisor
graph.add_edge("generate_query", "supervisor_agent")

# From supervisor, conditionally branch to three parallel paths
graph.add_conditional_edges(
    "supervisor_agent",
    should_load_inventory,
    {
        "load_inventory": "load_inventory_data",
        "skip_inventory": "aggregate_contexts"
        
    }
)

# graph.add_conditional_edges(
#     "supervisor_agent", 
#     should_use_tools,
#     {
#         "use_tools": "mcp_agent",
#         "skip_tools": "aggregate_contexts"
#     }
# )

# graph.add_conditional_edges(
#     "supervisor_agent",
#     should_use_rag, 
#     {
#         "use_rag": "rag_agent",
#         "skip_rag": "aggregate_contexts"
#     }
# )

# Path 1: Inventory/Logistics Context Pipeline
graph.add_edge("load_inventory_data", "logistics_data_agent")
graph.add_edge("logistics_data_agent", "contextualize_agent")
graph.add_edge("contextualize_agent", "aggregate_contexts")

# Path 2: Tool Execution Context Pipeline  
# graph.add_edge("mcp_agent", "tool_executor_agent")
# graph.add_edge("tool_executor_agent", "aggregate_contexts")

# Path 3: RAG Context Pipeline
# graph.add_edge("rag_agent", "grade_documents") 
# graph.add_edge("grade_documents", "aggregate_contexts")

# Final step: aggregate all contexts and generate answer
graph.add_edge("aggregate_contexts", "generate_answer")
graph.add_edge("generate_answer", END)

# Compile the graph
rag_chain = graph.compile()


__all__ = [
    "vector_store",
    "process_documents",
    "load_json",
    "index_docs",
    "rag_chain",
    "Document",
    "contextualize_agent",      # <-- Add if you want to expose it
    "aggregate_contexts"        # <-- Add if you want to expose it
]

