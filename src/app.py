"""
Income Tax RAG Assistant — Terminal Entry Point.
"""

from vectorstore import get_vectorstore
from rag_chain import build_chain


def main():
    print("\n" + "=" * 55)
    print("  💰  Income Tax RAG Assistant  💰")
    print("=" * 55)
    print("  Ask questions about Indian Income Tax.")
    print("  Type 'exit' or 'quit' to stop.\n")

    # ── Initialise pipeline ──
    vectorstore = get_vectorstore()
    chain = build_chain(vectorstore)
    print("\n✅ Ready! Ask your questions.\n")

    # ── Interactive loop ──
    while True:
        try:
            query = input("❓ Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue
        if query.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        result = chain.invoke({"query": query})

        print("\n📝 Answer:")
        print(result["result"])

        # Show source pages
        if result.get("source_documents"):
            pages = {
                doc.metadata.get("page", "?")
                for doc in result["source_documents"]
            }
            print(f"\n📄 Sources: pages {sorted(pages)}")

        print("-" * 55 + "\n")


if __name__ == "__main__":
    main()
