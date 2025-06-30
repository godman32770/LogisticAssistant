
from typing import List, Dict, TypedDict, Any
from langchain_core.documents import Document

class RAGState(TypedDict):
    question: str
    query: str
    # retrieved_docs: List[Document]
    # top_docs: List[Document]
    # retrieval_plan: List[Dict]
    answer: str
    history: List[Dict[str, str]]
    subtasks: List[str]
    # tool_results: List[Dict]
    # tool_plan: List[Dict]
    data: Dict[str, Any]
    structured_insights: str
    context_prompt: str
    context: str

# If you use model globally, you can also move model here:
# from utils.vectorstores import model