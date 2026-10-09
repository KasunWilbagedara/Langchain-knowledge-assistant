from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Correct PDF path
PDF_PATH = BASE_DIR / "data" / "documents" / "sample.pdf"


def load_pdf():
    if not PDF_PATH.is_file():
        raise FileNotFoundError(
            f"PDF file not found: {PDF_PATH}"
        )

    loader = PyPDFLoader(str(PDF_PATH))
    documents = loader.load()

    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    return chunks


if __name__ == "__main__":
    documents = load_pdf()
    chunks = split_documents(documents)

    print("Total pages:", len(documents))
    print("Total chunks:", len(chunks))

    if chunks:
        print("\nFirst chunk:\n")
        print(chunks[0].page_content)

        print("\nMetadata:\n")
        print(chunks[0].metadata)