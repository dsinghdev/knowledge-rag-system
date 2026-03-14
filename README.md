# 📈 Knowledge RAG Assistant

A robust, conversational Retrieval-Augmented Generation (RAG) system built with **LangChain**, **Gemini 2.5 Flash**, and **FAISS**. This assistant allows you to chat with any collection of documents, maintaining context across multiple turns with high-speed streaming results.

## 🏛️ System Architecture

This project follows a professional RAG pipeline, ensuring high accuracy and conversational awareness.

```mermaid
graph TD
    Start([<b>START</b>]) --> UI[<b>User Interface</b><br/>Question Input]
    UI --> History{<b>Any Chat History?</b>}
    
    History -- "YES" --> Condenser[<b>Query Condenser</b><br/>Rewrites query using Memory]
    History -- "NO" --> Embed[<b>Query Embedding</b><br/>MiniLM-L12-v2 Model]
    
    Condenser --> Embed
    
    subgraph Knowledge_Retrieval [Knowledge Retrieval]
        Embed --> DB[(<b>Vector Database</b><br/>FAISS Index)]
        DB -- "Search" --> Docs[<b>Relevant Context</b><br/>Top Snippets]
    end
    
    Docs --> Generator[<b>Answer Generator</b><br/>Gemini 2.5 Flash]
    Generator --> Response([<b>RESPONSE</b>])
    
    %% Style definitions for high-contrast visibility (Light & Dark modes)
    classDef logic_node fill:#1a73e8,stroke:#174ea6,stroke-width:2px,color:#ffffff;
    classDef data_node fill:#f9ab00,stroke:#ea8600,stroke-width:2px,color:#202124;
    classDef ai_node fill:#00d1b2,stroke:#009e86,stroke-width:2px,color:#202124;
    classDef start_node fill:#ea4335,stroke:#c5221f,stroke-width:2px,color:#ffffff;
    
    class Start,Response,Generator start_node;
    class UI,History,Condenser logic_node;
    class Embed,DB,Docs data_node;
    class Generator ai_node;
```

## 🚀 Key Features

- **Context-Aware Retrieval**: Automatically rewrites multi-turn questions to resolve pronouns and context.
- **Semantic Search**: Uses `paraphrase-multilingual-MiniLM-L12-v2` for high-precision local document matching.
- **Generative Intelligence**: Powered by `Gemini 2.5 Flash` for factual, grounded, and concise answers.
- **Real-time Streaming**: Instant feedback with character-by-character output generation.

## 🛠️ Getting Started

1. **Add Documents**: Place your `.pdf`, `.txt`, or `.md` files in the `/data` folder.
2. **Setup Env**: Add your `GOOGLE_API_KEY` to the `.env` file.
3. **Run UI**:
   ```bash
   streamlit run src/frontend/streamlit_app.py
   ```
