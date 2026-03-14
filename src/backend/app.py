"""
Knowledge RAG Assistant — Terminal Entry Point.
"""

import sys
import os

# Support running from project root or inside src/backend
if os.path.dirname(__file__) not in sys.path:
    sys.path.append(os.path.dirname(__file__))

from rag_engine import ask_stream


def main():
    print("\n" + "=" * 55)
    print("  📈  Knowledge RAG Assistant  📈")
    print("=" * 55)
    print("  Ask questions based on your provided documents.")
    print("  Type 'exit' or 'quit' to stop.\n")

    print("✅ Ready! Ask your questions.\n")

    chat_history = []

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

        print("\n📝 Answer: ", end="", flush=True)
        
        sources_str = ""
        full_answer = ""
        
        # We only want to keep the last few messages to avoid huge context windows over time.
        # But for now passing the whole history.
        for chunk in ask_stream(query, chat_history=chat_history):
            if chunk.startswith("\n\n[SOURCES_METADATA:"):
                # Clean up the sources metadata chunk for display
                raw_sources = chunk.replace("\n\n[SOURCES_METADATA:", "").replace("]", "")
                sources_str = raw_sources
                continue
                
            print(chunk, end="", flush=True)
            full_answer += chunk

        print() # Newline after answer finishes

        if sources_str:
            print(f"\n📄 Sources: [{sources_str}]")

        print("-" * 55 + "\n")
        
        # Add to history
        chat_history.append({"role": "user", "content": query})
        chat_history.append({"role": "assistant", "content": full_answer})


if __name__ == "__main__":
    main()
