# 📈 Knowledge RAG Assistant

A robust, conversational Retrieval-Augmented Generation (RAG) system built with **LangChain**, **Gemini 2.6 Flash**, and **FAISS**. This assistant allows you to chat with any collection of documents, maintaining context across multiple turns with high-speed streaming results.

## 🏛️ System Architecture

This project follows a professional RAG pipeline, ensuring high accuracy and conversational awareness.

```mermaid
graph TD
    Start([<b>START</b>]) --> UI[<b>User Interface</b><br/>Question Input]
    UI --> History{<b>Any Chat History?</b>}
    
    History -- "YES" --> Brain1[<b>BRAIN #1: REWRITE</b><br/>Condense Query using Memory]
    History -- "NO" --> Embed[<b>QUERY EMBEDDING</b><br/>MiniLM-L12-v2 Model]
    
    Brain1 --> Embed
    
    subgraph Knowledge_Retrieval [KNOWLEDGE RETRIEVAL]
        Embed --> DB[(<b>Vector Database</b><br/>FAISS Index)]
        DB -- "Search" --> Docs[<b>Relevant Context</b><br/>Top-4 Snippets]
    end
    
    Docs --> Brain2[<b>BRAIN #2: AI GENERATOR</b><br/>Gemini 2.6 Flash]
    Brain2 --> Response([<b>RESPONSE</b>])
    
    style Start fill:#f8cecc,stroke:#b85450
    style UI fill:#dae8fc,stroke:#6c8ebf
    style History fill:#e1d5e7,stroke:#9673a6
    style Brain1 fill:#d5e8d4,stroke:#82b366
    style Embed fill:#fff2cc,stroke:#d6b656
    style Knowledge_Retrieval fill:#f5f5f5,stroke:#666666,stroke-dasharray: 5 5
    style Brain2 fill:#f8cecc,stroke:#b85450
    style Response fill:#dae8fc,stroke:#6c8ebf
```

## 🚀 Key Features

- **Brain #1 (The Condenser)**: Automatically rewrites multi-turn questions into standalone search queries.
- **Smart Knowledge Search**: Uses `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` for precise semantic retrieval.
- **Gemini 2.6 Flash**: Powered by the latest high-speed generative model for accurate, context-grounded answers.
- **Real-time Streaming**: Watch the response get typed out character-by-character for zero perceived latency.

## 🛠️ How to View/Edit the Design

The professional architectural design is available as a **Draw.io** XML file. 

1. Go to [app.diagrams.net](https://app.diagrams.net/).
2. Drag and drop the file `generic_rag_design.drawio` located in the artifacts directory.

---

### Getting Started

1. **Add Documents**: Place your `.pdf`, `.txt`, or `.md` files in the `/data` folder.
2. **Setup Env**: Add your `GOOGLE_API_KEY` to the `.env` file.
3. **Run UI**:
   ```bash
   streamlit run src/frontend/streamlit_app.py
   ```
