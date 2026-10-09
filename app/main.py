from pathlib import Path
from dotenv import load_dotenv

from app.chains import create_citation_rag_chain


def main():
    load_dotenv()

    print("Starting Citation-Aware RAG Assistant...")

    chain = create_citation_rag_chain()

    print("Assistant ready!")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()

        if question.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        if not question:
            continue

        try:
            result = chain.invoke({
                "question": question
            })

            print("\nAI Answer:\n")
            print(result["answer"])

            print("\nRetrieved Sources:")

            for index, document in enumerate(
                result["sources"],
                start=1
            ):
                source = Path(
                    document.metadata.get(
                        "source", "Unknown"
                    )
                ).name

                page_index = document.metadata.get("page")

                page = (
                    page_index + 1
                    if isinstance(page_index, int)
                    else "Unknown"
                )

                print(
                    f"[Source {index}] "
                    f"{source} — Page {page}"
                )

            print()

        except Exception as error:
            print(f"\nError: {error}\n")


if __name__ == "__main__":
    main()