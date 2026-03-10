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
Use the following context retrieved from official government documents to answer the question.

CONCISENESS IS CRITICAL: Provide brief, high-level summaries. Use short bullet points. Avoid large walls of text.

GREETING RULE: If the user input is ONLY a greeting (e.g., "Hi", "Hellp", "Hello there") and does NOT ask about a scheme, do NOT use the context. Respond ONLY with a friendly greeting.

Context:
{context}

Question: {question}

Instructions:
- Be clear, brief, and objective about schemes like PPF, NPS, SSY, KVP, NSC.
- Use a maximum of 3-4 bullet points for each scheme unless more detail is requested.
- Focus only on the most important facts (e.g., Interest Rate, Lock-in, Tax Benefit).
- Mention quarterly review of rates briefly.
- Do NOT provide personalized financial advice or make up information.

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
