import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import Chroma
import cohere
from langchain_core.documents import Document
import json

load_dotenv()

# Embeddings and LLM
embedding = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
model = ChatOpenAI(model_name="gpt-4o-mini", openai_api_key=os.getenv("OPENAI_API_KEY"))

# Vector stores
vector_store = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding,
    collection_name="web_scraper_db"
)
# Add more vector stores as needed:
# another_vector_store = Chroma(...)

# Cohere client
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
co = cohere.Client(COHERE_API_KEY)



# ✅ Load JSON Data
def load_json(file_path="scraped_output_openaiv3.json"):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

# ✅ Convert Chunked JSON to Documents
def process_documents(data):
    documents = []
    for item in data:
        for chunk in item.get("chunks", []):
            documents.append(Document(
                page_content=chunk,
                metadata={
                    "source": item.get("url", ""),
                    "title": item.get("title", ""),
                    "headings": ", ".join(item.get("headings", []))
                }
            ))
    return documents

# ✅ Index Documents (only once)
def index_docs(documents):
    if vector_store._collection.count() == 0:
        vector_store.add_documents(documents)

