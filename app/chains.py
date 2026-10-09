
import os

from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser

from app.prompts import knowledge_assistant_prompt

from langchain_core.runnables import RunnablePassthrough
from app.prompts import rag_prompt
from app.retrieval import get_retriever
from pathlib import Path

from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableLambda,
)

from app.prompts import citation_prompt



def format_documents_with_sources(documents):
    formatted_chunks = []

    for index, document in enumerate(documents, start=1):
        source = Path(
            document.metadata.get("source", "Unknown")
        ).name

        page_index = document.metadata.get("page")
        page = (
            page_index + 1
            if isinstance(page_index, int)
            else "Unknown"
        )

        formatted_chunks.append(
            f"[Source {index}]\n"
            f"File: {source}\n"
            f"Page: {page}\n"
            f"Content:\n{document.page_content}"
        )

    return "\n\n".join(formatted_chunks)

def create_citation_rag_chain():
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("GEMINI_API_KEY is missing")

    model = init_chat_model(
        "google_genai:gemini-3.8-flash",
        api_key=api_key,
        temperature=0
    )

    retriever = get_retriever()

    # Step 1: Retrieve documents
    retrieve_chain = RunnablePassthrough.assign(
        sources=lambda inputs: retriever.invoke(
            inputs["question"]
        )
    )

    # Step 2: Generate answer
    answer_chain = RunnablePassthrough.assign(
        answer=(
            {
                "context": (
                    lambda inputs:
                    format_documents_with_sources(
                        inputs["sources"]
                    )
                ),
                "question": lambda inputs: inputs["question"]
            }
            | citation_prompt
            | model
            | StrOutputParser()
        )
    )

    return retrieve_chain | answer_chain




def format_documents(documents):
    return "\n\n".join(
        document.page_content
        for document in documents
    )

def create_rag_chain():
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY is missing")
    model= init_chat_model (
    
        "google_genai:gemini-3.8-flash",
        api_key=api_key,
        temperature=0,
     
     )

    retriever = get_retriever()

    rag_chain = (
        {
            "context": retriever | format_documents,
            "question": RunnablePassthrough()
        }
        | rag_prompt
        | model
        | StrOutputParser()
    )

    return rag_chain

def create_knowledge_chain():
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

    placeholder_values = (
        "your_google_api_key_here",
        "your_gemini_api_key_here",
        "your_api_key_here",
        "example",
        "demo",
    )

    if not api_key or api_key.lower() in {value.lower() for value in placeholder_values}:
        raise ValueError(
            "No valid Google Gemini API key found. Set GOOGLE_API_KEY or GEMINI_API_KEY to a real key in your environment or .env file."
        )

    model = init_chat_model(
        "google_genai:gemini-3.8-flash",
        api_key=api_key,
        temperature=0,
    )

    parser = StrOutputParser()

    chain = (
        knowledge_assistant_prompt
        | model
        | parser
    )

    return chain
