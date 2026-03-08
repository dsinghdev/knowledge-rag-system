"""
RAG chain: builds a RetrievalQA chain with a grounding prompt.
"""

import logging

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

from config import GOOGLE_API_KEY, LLM_MODEL, LLM_TEMPERATURE

logger = logging.getLogger(__name__)


PROMPT_TEMPLATE = """You are an expert in Indian Small Savings and Investment schemes.
Use ONLY the following context retrieved from official government documents (India Post, Ministry of Finance, AMFI, etc.) to answer the question.
If the answer is not in the context, say "I don't have enough information from the documents to answer this."

Context:
{context}

Question: {question}

Instructions:
- Be clear, educational, and objective about Indian financial schemes (PPF, NPS, SSY, etc.).
- Highlight key facts like Lock-in periods, Eligibility, and Tax Benefits.
- Mention that interest rates for small savings schemes (like PPF, NSC) are reviewed quarterly by the Government of India.
- Use bullet points for comparisons.
- Do NOT provide personalized financial advice (e.g., "You should invest in X").
- Do NOT make up information.

Answer:"""

QA_PROMPT = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["context", "question"],
)


def build_chain(vectorstore):
    """Build and return a RetrievalQA chain."""
    logger.info("Building RetrievalQA chain with model=%s", LLM_MODEL)
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        google_api_key=GOOGLE_API_KEY,
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_PROMPT},
    )

    return chain
