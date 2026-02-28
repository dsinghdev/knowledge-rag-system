"""
RAG chain: builds a RetrievalQA chain with a grounding prompt.
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

from config import GOOGLE_API_KEY, LLM_MODEL, LLM_TEMPERATURE


PROMPT_TEMPLATE = """You are an expert Indian Income Tax assistant.
Use ONLY the following context retrieved from official tax documents to answer the question.
If the answer is not in the context, say "I don't have enough information from the documents to answer this."

Context:
{context}

Question: {question}

Instructions:
- Be precise and cite section numbers when available.
- Use bullet points for clarity.
- Do NOT make up information.

Answer:"""

QA_PROMPT = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["context", "question"],
)


def build_chain(vectorstore):
    """Build and return a RetrievalQA chain."""
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
