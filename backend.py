from __future__ import annotations

import os
import streamlit as st
from typing import List, Literal, TypedDict

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

load_dotenv()

try:
    for key, value in st.secrets.items():
        if isinstance(value, str):
            os.environ.setdefault(key, value)
except Exception:
    pass

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DOCUMENTS_DIR     = "./documents"
CHUNK_SIZE        = 600
CHUNK_OVERLAP     = 150
RETRIEVER_K       = 4
MAX_RETRIES       = 10
MAX_REWRITE_TRIES = 3

# ---------------------------------------------------------------------------
# LLM + Embeddings
# ---------------------------------------------------------------------------

_llm = ChatGroq(
    model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
)

_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ---------------------------------------------------------------------------
# Build vector store (done once at import time)
# ---------------------------------------------------------------------------

def _load_vector_store() -> FAISS:
    pdf_files = [
        "Company_Policies.pdf",
        "Company_Profile.pdf",
        "Product_and_pricing.pdf",
    ]
    docs: List[Document] = []
    for fname in pdf_files:
        path = os.path.join(DOCUMENTS_DIR, fname)
        if os.path.exists(path):
            docs.extend(PyPDFLoader(path).load())

    if not docs:
        raise FileNotFoundError(
            f"No PDF files found in '{DOCUMENTS_DIR}'. "
            "Place Company_Policies.pdf, Company_Profile.pdf, "
            "and Product_and_pricing.pdf there."
        )

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    ).split_documents(docs)

    return FAISS.from_documents(chunks, _embeddings)


_vector_store = _load_vector_store()
_retriever = _vector_store.as_retriever(search_kwargs={"k": RETRIEVER_K})


# ---------------------------------------------------------------------------
# Graph State
# ---------------------------------------------------------------------------

class State(TypedDict):
    question:        str
    retrieval_query: str
    rewrite_tries:   int
    need_retrieval:  bool
    docs:            List[Document]
    relevant_docs:   List[Document]
    context:         str
    answer:          str
    issup:           Literal["fully_supported", "partially_supported", "no_support", ""]
    evidence:        List[str]
    retries:         int
    isuse:           Literal["useful", "not_useful", ""]
    use_reason:      str
    trace:           List[str]


# ---------------------------------------------------------------------------
# 1. Decide retrieval
# ---------------------------------------------------------------------------

_ALWAYS_RETRIEVE_KEYWORDS = [
    "nexaai", "nexa ai", "nexa-ai",
    "company", "culture", "ceo", "founder", "team", "employee",
    "policy", "policies", "leave", "probation", "notice", "terminate",
    "product", "pricing", "plan", "refund", "trial", "subscription",
    "feature", "integration", "api", "support", "billing",
    "who is", "what is", "how does", "describe", "tell me about",
    "document", "documents",
]

_META_KEYWORDS = [
    "hello", "hi ", "hey", "what can you do", "how are you",
    "who are you", "what are you",
]


def _force_retrieve(question: str) -> bool | None:
    q = question.lower()
    if any(kw in q for kw in _META_KEYWORDS):
        return False
    if any(kw in q for kw in _ALWAYS_RETRIEVE_KEYWORDS):
        return True
    return None


class RetrieveDecision(BaseModel):
    should_retrieve: bool = Field(
        description="True if external documents are needed to answer reliably."
    )


_decide_retrieval_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You decide whether to retrieve documents from a company knowledge base.\n"
        "Return JSON with exactly one key: should_retrieve (boolean).\n\n"
        "RULES — follow in order:\n"
        "1. should_retrieve=TRUE for ANY question about:\n"
        "   - A specific company, product, plan, policy, person, or organisation.\n"
        "   - Prices, features, refunds, trials, APIs, integrations.\n"
        "   - Company culture, leadership, team, employees, history.\n"
        "2. should_retrieve=FALSE ONLY for pure general knowledge with no company angle:\n"
        "   - Basic definitions (e.g. 'what is machine learning').\n"
        "   - Math, grammar, science facts.\n"
        "   - Greetings or small talk.\n"
        "3. When in doubt → should_retrieve=TRUE.\n\n"
        "Example TRUE:  'Who is the CEO?'  'What is NexaAI?'  'Describe the company culture.'\n"
        "Example FALSE: 'What is 2+2?'  'Define entropy.'  'Hi, how are you?'",
    ),
    ("human", "Question: {question}"),
])

_should_retrieve_llm = _llm.with_structured_output(RetrieveDecision)


def decide_retrieval(state: State) -> dict:
    trace = state.get("trace", [])
    question = state["question"]

    forced = _force_retrieve(question)
    if forced is True:
        trace.append("🔍 Decide retrieval → **retrieve** _(keyword match)_")
        return {"need_retrieval": True, "trace": trace}
    if forced is False:
        trace.append("🔍 Decide retrieval → **direct answer** _(keyword match)_")
        return {"need_retrieval": False, "trace": trace}

    try:
        decision: RetrieveDecision = _should_retrieve_llm.invoke(
            _decide_retrieval_prompt.format_messages(question=question)
        )
        should = decision.should_retrieve
    except Exception:
        should = True

    trace.append(f"🔍 Decide retrieval → **{'retrieve' if should else 'direct answer'}** _(LLM decision)_")
    return {"need_retrieval": should, "trace": trace}


def _route_after_decide(state: State) -> Literal["generate_direct", "retrieve"]:
    return "retrieve" if state["need_retrieval"] else "generate_direct"


# ---------------------------------------------------------------------------
# 2. Direct answer
# ---------------------------------------------------------------------------

_direct_generation_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful assistant. Answer using your general knowledge.\n\n"
        "RULES:\n"
        "- For greetings or small talk, respond naturally and warmly.\n"
        "- For general knowledge questions (math, science, definitions), answer directly.\n"
        "- If the question is about a specific company, person, product, or policy "
        "that you don't have reliable information about, say: "
        "'I don't have specific information about that. Try asking something else!'\n"
        "- Keep answers concise.",
    ),
    ("human", "{question}"),
])


def generate_direct(state: State) -> dict:
    out = _llm.invoke(
        _direct_generation_prompt.format_messages(question=state["question"])
    )
    trace = state.get("trace", [])
    trace.append("💬 Generated **direct answer** (no retrieval needed)")
    return {"answer": out.content, "trace": trace}


# ---------------------------------------------------------------------------
# 3. Retrieve
# ---------------------------------------------------------------------------

def retrieve(state: State) -> dict:
    q = state.get("retrieval_query") or state["question"]
    docs = _retriever.invoke(q)
    trace = state.get("trace", [])
    trace.append(f"📄 Retrieved **{len(docs)}** chunks  (query: _{q}_)")
    return {"docs": docs, "trace": trace}


# ---------------------------------------------------------------------------
# 4. Relevance filter
# ---------------------------------------------------------------------------

class RelevanceDecision(BaseModel):
    is_relevant: bool = Field(
        description="True if the document discusses the same topic area as the question."
    )


_is_relevant_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are judging document relevance at a TOPIC level.\n"
        "Return JSON matching the schema.\n\n"
        "A document is relevant if it discusses the same entity or topic area as the question.\n"
        "It does NOT need to contain the exact answer.\n\n"
        "Examples:\n"
        "- HR policies are relevant to questions about notice period, probation, termination, benefits.\n"
        "- Pricing documents are relevant to questions about refunds, trials, billing terms.\n"
        "- Company profile is relevant to questions about leadership, culture, size, or strategy.\n\n"
        "Do NOT decide whether the document fully answers the question.\n"
        "When unsure, return is_relevant=true.",
    ),
    ("human", "Question:\n{question}\n\nDocument:\n{document}"),
])

_relevance_llm = _llm.with_structured_output(RelevanceDecision)


def is_relevant(state: State) -> dict:
    all_docs = state.get("docs", [])
    relevant_docs: List[Document] = []
    failed = 0
    for doc in all_docs:
        try:
            decision: RelevanceDecision = _relevance_llm.invoke(
                _is_relevant_prompt.format_messages(
                    question=state["question"],
                    document=doc.page_content,
                )
            )
            if decision.is_relevant:
                relevant_docs.append(doc)
        except Exception:
            relevant_docs.append(doc)
            failed += 1

    if not relevant_docs and all_docs:
        relevant_docs = all_docs
        trace = state.get("trace", [])
        trace.append(
            f"⚠️ Relevance filter dropped all chunks — using all **{len(all_docs)}** as fallback"
        )
        return {"relevant_docs": relevant_docs, "trace": trace}

    trace = state.get("trace", [])
    suffix = f" _(+{failed} parse errors accepted)_" if failed else ""
    trace.append(
        f"✅ Relevance filter → **{len(relevant_docs)}/{len(all_docs)}** chunks kept{suffix}"
    )
    return {"relevant_docs": relevant_docs, "trace": trace}


def _route_after_relevance(
    state: State,
) -> Literal["generate_from_context", "no_answer_found"]:
    if state.get("relevant_docs"):
        return "generate_from_context"
    return "no_answer_found"


# ---------------------------------------------------------------------------
# 5. Generate from context
# ---------------------------------------------------------------------------

_rag_generation_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a business RAG chatbot.\n"
        "Answer the question based solely on the provided context.\n"
        "Do not mention that you are using a context block.",
    ),
    ("human", "Question:\n{question}\n\nContext:\n{context}"),
])


def generate_from_context(state: State) -> dict:
    context = "\n\n---\n\n".join(
        [d.page_content for d in state.get("relevant_docs", [])]
    ).strip()
    if not context:
        return {"answer": "No answer found.", "context": ""}
    out = _llm.invoke(
        _rag_generation_prompt.format_messages(
            question=state["question"], context=context
        )
    )
    trace = state.get("trace", [])
    trace.append("🤖 Generated answer from context")
    return {"answer": out.content, "context": context, "trace": trace}


def no_answer_found(state: State) -> dict:
    trace = state.get("trace", [])
    trace.append("❌ No relevant documents found — returning fallback")
    return {
        "answer": "I couldn't find a specific answer in the company documents for that question. Try rephrasing or ask something else!",
        "context": "",
        "trace": trace,
    }


# ---------------------------------------------------------------------------
# 6. IsSUP verify + revise loop
# ---------------------------------------------------------------------------

class IsSUPDecision(BaseModel):
    issup: Literal["fully_supported", "partially_supported", "no_support"]
    evidence: List[str] = Field(default_factory=list)


_issup_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are verifying whether the ANSWER is supported by the CONTEXT.\n"
        "Return JSON with keys: issup, evidence.\n"
        "issup must be one of: fully_supported, partially_supported, no_support.\n\n"
        "- fully_supported: every meaningful claim is explicitly supported.\n"
        "- partially_supported: core facts are supported BUT answer includes abstraction/interpretation not in CONTEXT.\n"
        "- no_support: key claims are not supported by CONTEXT.\n\n"
        "evidence: up to 3 short direct quotes from CONTEXT supporting the supported parts.",
    ),
    ("human", "Question:\n{question}\n\nAnswer:\n{answer}\n\nContext:\n{context}\n"),
])

_issup_llm = _llm.with_structured_output(IsSUPDecision)


def is_sup(state: State) -> dict:
    decision: IsSUPDecision = _issup_llm.invoke(
        _issup_prompt.format_messages(
            question=state["question"],
            answer=state.get("answer", ""),
            context=state.get("context", ""),
        )
    )
    trace = state.get("trace", [])
    trace.append(f"🧪 IsSUP → **{decision.issup}**")
    return {"issup": decision.issup, "evidence": decision.evidence, "trace": trace}


def _route_after_issup(
    state: State,
) -> Literal["accept_answer", "revise_answer"]:
    if state.get("issup") == "fully_supported":
        return "accept_answer"
    if state.get("retries", 0) >= MAX_RETRIES:
        return "accept_answer"
    return "revise_answer"


def accept_answer(state: State) -> dict:
    return {}


_revise_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a STRICT reviser.\n\n"
        "FORMAT (quote-only answer):\n"
        "- <direct quote from the CONTEXT>\n"
        "- <direct quote from the CONTEXT>\n\n"
        "Rules:\n"
        "- Use ONLY the CONTEXT.\n"
        "- Do NOT add any words besides bullet dashes and the quotes.\n"
        "- Do NOT explain or say 'context', 'not mentioned', etc.",
    ),
    ("human", "Question:\n{question}\n\nCurrent Answer:\n{answer}\n\nCONTEXT:\n{context}"),
])


def revise_answer(state: State) -> dict:
    out = _llm.invoke(
        _revise_prompt.format_messages(
            question=state["question"],
            answer=state.get("answer", ""),
            context=state.get("context", ""),
        )
    )
    retries = state.get("retries", 0) + 1
    trace = state.get("trace", [])
    trace.append(f"✏️ Revised answer (attempt {retries})")
    return {"answer": out.content, "retries": retries, "trace": trace}


# ---------------------------------------------------------------------------
# 7. IsUSE
# ---------------------------------------------------------------------------

class IsUSEDecision(BaseModel):
    isuse: Literal["useful", "not_useful"]
    reason: str = Field(description="Short reason in 1 line.")


_isuse_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You judge USEFULNESS of the ANSWER for the QUESTION.\n"
        "Return JSON with keys: isuse, reason.\n"
        "- useful: answer directly answers the question or provides the requested specific info.\n"
        "- not_useful: answer is generic, off-topic, or only gives background without answering.\n"
        "Do NOT re-check grounding. Only check: 'Did we answer the question?'\n"
        "Keep reason to 1 short line.",
    ),
    ("human", "Question:\n{question}\n\nAnswer:\n{answer}"),
])

_isuse_llm = _llm.with_structured_output(IsUSEDecision)


def is_use(state: State) -> dict:
    decision: IsUSEDecision = _isuse_llm.invoke(
        _isuse_prompt.format_messages(
            question=state["question"],
            answer=state.get("answer", ""),
        )
    )
    trace = state.get("trace", [])
    trace.append(f"💡 IsUSE → **{decision.isuse}** — _{decision.reason}_")
    return {"isuse": decision.isuse, "use_reason": decision.reason, "trace": trace}


def _route_after_isuse(
    state: State,
) -> Literal["END", "rewrite_question", "no_answer_found"]:
    if state.get("isuse") == "useful":
        return "END"
    if state.get("rewrite_tries", 0) >= MAX_REWRITE_TRIES:
        return "no_answer_found"
    return "rewrite_question"


# ---------------------------------------------------------------------------
# 8. Rewrite query
# ---------------------------------------------------------------------------

class RewriteDecision(BaseModel):
    retrieval_query: str = Field(
        description="Rewritten query optimised for vector retrieval over internal company PDFs."
    )


_rewrite_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Rewrite the user's QUESTION into a query optimised for vector retrieval over INTERNAL company PDFs.\n\n"
        "Rules:\n"
        "- Keep it short (6-16 words).\n"
        "- Add 2-5 high-signal keywords that likely appear in policy/pricing docs.\n"
        "- Remove filler words.\n"
        "- Do NOT answer the question.\n"
        "- Output JSON with key: retrieval_query",
    ),
    (
        "human",
        "QUESTION:\n{question}\n\n"
        "Previous retrieval query:\n{retrieval_query}\n\n"
        "Answer (if any):\n{answer}",
    ),
])

_rewrite_llm = _llm.with_structured_output(RewriteDecision)


def rewrite_question(state: State) -> dict:
    decision: RewriteDecision = _rewrite_llm.invoke(
        _rewrite_prompt.format_messages(
            question=state["question"],
            retrieval_query=state.get("retrieval_query", ""),
            answer=state.get("answer", ""),
        )
    )
    rewrite_tries = state.get("rewrite_tries", 0) + 1
    trace = state.get("trace", [])
    trace.append(
        f"🔄 Rewrote query (attempt {rewrite_tries}): _{decision.retrieval_query}_"
    )
    return {
        "retrieval_query": decision.retrieval_query,
        "rewrite_tries":   rewrite_tries,
        "docs":            [],
        "relevant_docs":   [],
        "context":         "",
        "trace":           trace,
    }


# ---------------------------------------------------------------------------
# Build graph
# ---------------------------------------------------------------------------

def _build_graph():
    g = StateGraph(State)

    g.add_node("decide_retrieval",      decide_retrieval)
    g.add_node("generate_direct",       generate_direct)
    g.add_node("retrieve",              retrieve)
    g.add_node("is_relevant",           is_relevant)
    g.add_node("generate_from_context", generate_from_context)
    g.add_node("no_answer_found",       no_answer_found)
    g.add_node("is_sup",                is_sup)
    g.add_node("accept_answer",         accept_answer)
    g.add_node("revise_answer",         revise_answer)
    g.add_node("is_use",                is_use)
    g.add_node("rewrite_question",      rewrite_question)

    g.add_edge(START, "decide_retrieval")
    g.add_conditional_edges(
        "decide_retrieval", _route_after_decide,
        {"generate_direct": "generate_direct", "retrieve": "retrieve"},
    )
    g.add_edge("generate_direct", END)
    g.add_edge("retrieve", "is_relevant")
    g.add_conditional_edges(
        "is_relevant", _route_after_relevance,
        {"generate_from_context": "generate_from_context", "no_answer_found": "no_answer_found"},
    )
    g.add_edge("no_answer_found", END)
    g.add_edge("generate_from_context", "is_sup")
    g.add_conditional_edges(
        "is_sup", _route_after_issup,
        {"accept_answer": "accept_answer", "revise_answer": "revise_answer"},
    )
    g.add_edge("revise_answer", "is_sup")
    g.add_edge("accept_answer", "is_use")
    g.add_conditional_edges(
        "is_use", _route_after_isuse,
        {"END": END, "rewrite_question": "rewrite_question", "no_answer_found": "no_answer_found"},
    )
    g.add_edge("rewrite_question", "retrieve")

    return g.compile()


_app = _build_graph()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_rag(question: str) -> dict:
    initial_state: State = {
        "question":        question,
        "retrieval_query": "",
        "rewrite_tries":   0,
        "need_retrieval":  False,
        "docs":            [],
        "relevant_docs":   [],
        "context":         "",
        "answer":          "",
        "issup":           "",
        "evidence":        [],
        "retries":         0,
        "isuse":           "",
        "use_reason":      "",
        "trace":           [],
    }
    result = _app.invoke(initial_state, config={"recursion_limit": 80})
    return result