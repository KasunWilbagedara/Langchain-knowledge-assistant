import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

from app.documents import load_pdf, split_documents

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "chroma_db"

def search_with_scores(query, k=3):
    load_dotenv()

    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=os.getenv("GEMINI_API_KEY")
    )

    vector_store = Chroma(
        collection_name="knowledge_assistant",
        embedding_function=embeddings,
        persist_directory=str(DB_PATH)
    )

    results = vector_store.similarity_search_with_score(
        query,
        k=k
    )

    return results

def create_vector_store():
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("GEMINI_API_KEY is missing")

    print("STEP 1: Loading PDF...", flush=True)
    documents = load_pdf()
    print(f"Loaded {len(documents)} pages", flush=True)

    print("STEP 2: Splitting documents...", flush=True)
    chunks = split_documents(documents)
    print(f"Created {len(chunks)} chunks", flush=True)

    if not chunks:
        raise ValueError("No text chunks found")

    print("STEP 3: Initializing embedding model...", flush=True)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=api_key
    )

    print("STEP 4: Testing one embedding...", flush=True)
    test_vector = embeddings.embed_query("Hello world")
    print(f"Embedding successful: {len(test_vector)} dimensions", flush=True)

    print("STEP 5: Creating Chroma database...", flush=True)

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="knowledge_assistant",
        persist_directory=str(DB_PATH)
    )

    print("STEP 6: Database created successfully!", flush=True)

    return vector_store

if __name__ == "__main__":
    print("Starting retrieval...", flush=True)

    vector_store = create_vector_store()

    query = input("Ask a question about your PDF: ")

    results = vector_store.similarity_search(query, k=3)

    for index, document in enumerate(results, start=1):
        print(f"\n--- Result {index} ---")
        print(document.page_content)


def get_retriever():
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("GEMINI_API_KEY is missing")

    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Vector database not found: {DB_PATH}"
        )

    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=api_key
    )

    vector_store = Chroma(
        collection_name="knowledge_assistant",
        embedding_function=embeddings,
        persist_directory=str(DB_PATH)
    )

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    return retriever