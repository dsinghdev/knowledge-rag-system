"""
Income Tax RAG Assistant — Terminal Entry Point.
"""

from backend_service import ask


def main():
    print("\n" + "=" * 55)
    print("  💰  Income Tax RAG Assistant  💰")
    print("=" * 55)
    print("  Ask questions about Indian Income Tax.")
    print("  Type 'exit' or 'quit' to stop.\n")

    print("✅ Ready! Ask your questions.\n")

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

        result = ask(query)

        print("\n📝 Answer:")
        print(result["answer"])

        if result["sources"]:
            print(f"\n📄 Sources: pages {result['sources']}")

        print("-" * 55 + "\n")


if __name__ == "__main__":
    main()
