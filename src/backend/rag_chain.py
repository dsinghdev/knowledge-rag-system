"""
RAG chain: builds a RetrievalQA chain with a grounding prompt.
"""

import logging

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

from config import GOOGLE_API_KEY, LLM_MODEL, LLM_TEMPERATURE, factory

logger = factory.initialize()


from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])


def build_chain(vectorstore):
    """Build and return an LCEL conversational RAG chain with streaming support."""
    logger.info("Building LCEL Conversational RAG chain with model=%s", LLM_MODEL)
    
    # Initialize the LLM with streaming=True
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        google_api_key=GOOGLE_API_KEY,
        streaming=True,
    )

    # Initialize the retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )

    # 1. Contextualize the question based on chat history
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    # Pre-chain to resolve the standalone question if history exists
    # We use a simple lambda to decide whether to invoke the condense chain
    condense_chain = contextualize_q_prompt | llm | StrOutputParser()
    
    def get_standalone_question(inputs):
        chat_history = inputs.get("chat_history", [])
        if chat_history:
            return condense_chain.invoke(inputs)
        return inputs["input"]

    # 2. Answer the question using the retrieved documents
    qa_system_prompt = """You are a helpful and expert AI assistant.
Use the following context retrieved from the provided documents to answer the question.

CONCISENESS IS CRITICAL: Provide brief, high-level summaries. Use short bullet points where appropriate. Avoid large walls of text.

Context:
{context}

Instructions:
- Be clear, brief, and objective.
- Focus only on the most important facts from the retrieved context.
- If the answer cannot be found in the context, politely state that you don't know based on the provided documents.
- Do NOT provide personalized advice or make up information outside of the context.
"""
    
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # 3. Combine into a single LCEL chain
    rag_chain = (
        RunnablePassthrough.assign(
            standalone_question=RunnablePassthrough() | get_standalone_question
        )
        .assign(
            context_docs=lambda x: retriever.invoke(x["standalone_question"])
        )
        .assign(
            context=lambda x: format_docs(x["context_docs"])
        )
        .assign(
            answer=qa_prompt | llm | StrOutputParser()
        )
    )
    
    # We map "context_docs" back to "context" for the engine to grab the metadata
    return rag_chain | (lambda x: {"answer": x["answer"], "context": x["context_docs"]})

