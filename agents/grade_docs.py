from core import RAGState
from utils.vectorstores import model , vector_store
from utils.helpers import filter_docs_by_domain, cohere_rerank

# ✅ Agent 2: Grade and Select Top Documents
def grade_docs(state: RAGState) -> RAGState:
    # Example: you have multiple vector stores
    vector_stores = {
        "web_scraper_db": vector_store,
        # "another_db": another_vector_store,  # Add more if needed
    }

    all_retrieved = []
    retrieval_plan = state.get("retrieval_plan", [])
    for plan in retrieval_plan:
        db_name = plan.get("database", "web_scraper_db")
        query = plan.get("what_to_retrieve", state["query"])
        store = vector_stores.get(db_name, vector_store)
        retrieved = store.similarity_search(query)
        # Optionally, add subgoal info to metadata
        for doc in retrieved:
            doc.metadata["subgoal"] = plan.get("subgoal", "")
            doc.metadata["used_db"] = db_name
        all_retrieved.extend(retrieved)

    # Structured filtering by domain (optional)
    allowed_domains = ["example.com", "anotherdomain.com"]  # <-- Edit as needed
    filtered = filter_docs_by_domain(all_retrieved, allowed_domains)
    if not filtered:
        filtered = all_retrieved  # fallback if nothing matches

    # Cohere rerank (on all filtered docs)
    top_docs = cohere_rerank(state["query"], filtered, top_n=2)

    return {**state, "retrieved_docs": filtered, "top_docs": top_docs}