from urllib.parse import urlparse

def filter_docs_by_domain(docs, allowed_domains):
    filtered = []
    for doc in docs:
        url = doc.metadata.get("source", "")
        domain = urlparse(url).netloc
        if any(domain.endswith(allowed) for allowed in allowed_domains):
            filtered.append(doc)
    return filtered

def cohere_rerank(query, docs, co, top_n=2):
    passages = [doc.page_content for doc in docs]
    results = co.rerank(
        query=query,
        documents=passages,
        top_n=top_n,
        model="rerank-english-v2.0"
    )
    reranked_docs = [docs[r['index']] for r in results]
    return reranked_docs