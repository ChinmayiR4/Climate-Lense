"""
Node functions for the ClimateLens LangGraph pipeline.

Flow:
  retrieve -> grade_relevance -> [loop back to retrieve if weak, up to 2x]
           -> compare_to_benchmark (only if target_doc_type is set)
           -> generate_answer
           -> log_interaction

The "agentic" part isn't the LLM call - it's that grade_relevance can
send control flow back to retrieve with a rewritten query, and
compare_to_benchmark makes an actual judgment call (alignment score +
flags) rather than just returning retrieved text.
"""

import os
import time

from dotenv import load_dotenv

load_dotenv()

from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from db.mongo_client import ClimateLensMongo
from db.sql_models import ClimateLensSQL
from graph.state import ClimateLensState

CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_store")
MAX_RETRIEVAL_ATTEMPTS = 2

# Open-source model via Groq's free tier (no cost, no credit card, generous rate limits)
_llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)
# Local embeddings - no API calls, runs on-device via sentence-transformers
_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
_store = Chroma(persist_directory=CHROMA_DIR, embedding_function=_embeddings)

_mongo = ClimateLensMongo()
_sql = ClimateLensSQL()


def retrieve(state: ClimateLensState) -> ClimateLensState:
    query = state.get("rewritten_query") or state["query"]
    filter_dict = None
    if state.get("target_doc_type"):
        filter_dict = {"source_type": state["target_doc_type"]}

    results = _store.similarity_search_with_score(query, k=5, filter=filter_dict)
    chunks = [{
        "text": doc.page_content,
        "source_type": doc.metadata.get("source_type"),
        "title": doc.metadata.get("title"),
        "entity_name": doc.metadata.get("entity_name"),
        "score": score,
    } for doc, score in results]

    state["retrieved_chunks"] = chunks
    state["retrieval_attempts"] = state.get("retrieval_attempts", 0) + 1
    return state


def grade_relevance(state: ClimateLensState) -> ClimateLensState:
    """Cheap heuristic + LLM check: is retrieved evidence actually
    relevant to the question, or do we need to re-query?"""
    chunks = state.get("retrieved_chunks", [])
    if not chunks:
        state["relevance_ok"] = False
        return state

    combined = "\n---\n".join(c["text"][:300] for c in chunks[:3])
    prompt = (
        f"Question: {state['query']}\n\n"
        f"Retrieved excerpts:\n{combined}\n\n"
        "Do these excerpts contain enough information to answer the question? "
        "Reply with exactly one word: YES or NO."
    )
    resp = _llm.invoke(prompt).content.strip().upper()
    state["relevance_ok"] = resp.startswith("YES")

    if not state["relevance_ok"] and state.get("retrieval_attempts", 0) < MAX_RETRIEVAL_ATTEMPTS:
        # Rewrite the query for a second retrieval pass instead of giving up
        rewrite_prompt = (
            f"Original question: {state['query']}\n"
            "This question returned weak search results. Rewrite it as a more "
            "specific search query using climate/policy terminology likely to "
            "appear in official reports. Reply with only the rewritten query."
        )
        state["rewritten_query"] = _llm.invoke(rewrite_prompt).content.strip()

    return state


def should_retry_retrieval(state: ClimateLensState) -> str:
    """Conditional edge: loop back to retrieve, or move on."""
    if not state.get("relevance_ok") and state.get("retrieval_attempts", 0) < MAX_RETRIEVAL_ATTEMPTS:
        return "retrieve"
    return "next"


def compare_to_benchmark(state: ClimateLensState) -> ClimateLensState:
    """Only runs when target_doc_type is set (i.e. the user is asking
    'does X's pledge/report align with IPCC targets'). Pulls IPCC chunks
    as the benchmark and asks the LLM to score alignment + flag gaps."""
    if not state.get("target_doc_type"):
        state["alignment_score"] = None
        state["flags"] = []
        return state

    benchmark_results = _store.similarity_search(
        state["query"], k=4, filter={"source_type": "ipcc"}
    )
    benchmark_chunks = [{
        "text": d.page_content,
        "title": d.metadata.get("title"),
    } for d in benchmark_results]
    state["benchmark_chunks"] = benchmark_chunks

    claim_text = "\n---\n".join(c["text"] for c in state["retrieved_chunks"])
    benchmark_text = "\n---\n".join(c["text"] for c in benchmark_chunks)

    prompt = (
        "You are evaluating whether a claim from a national climate pledge or "
        "corporate ESG report aligns with IPCC benchmarks.\n\n"
        f"CLAIM (from {state['target_doc_type']}):\n{claim_text}\n\n"
        f"IPCC BENCHMARK EXCERPTS:\n{benchmark_text}\n\n"
        "Respond in this exact format:\n"
        "SCORE: <a number from 0.0 to 1.0, where 1.0 means fully aligned/consistent>\n"
        "FLAGS: <comma-separated list of specific gaps or unsupported claims, "
        "or NONE if there are no issues>"
    )
    resp = _llm.invoke(prompt).content.strip()

    score = 0.5
    flags: list[str] = []
    for line in resp.splitlines():
        if line.upper().startswith("SCORE:"):
            try:
                score = float(line.split(":", 1)[1].strip())
            except ValueError:
                pass
        elif line.upper().startswith("FLAGS:"):
            raw = line.split(":", 1)[1].strip()
            if raw.upper() != "NONE":
                flags = [f.strip() for f in raw.split(",") if f.strip()]

    state["alignment_score"] = score
    state["flags"] = flags
    return state


def generate_answer(state: ClimateLensState) -> ClimateLensState:
    chunks_text = "\n---\n".join(
        f"[{c['title']}] {c['text']}" for c in state.get("retrieved_chunks", [])
    )

    alignment_note = ""
    if state.get("alignment_score") is not None:
        alignment_note = (
            f"\n\nAlignment score vs. IPCC benchmark: {state['alignment_score']:.2f}\n"
            f"Flags: {', '.join(state.get('flags', [])) or 'None'}"
        )

    prompt = (
        f"Question: {state['query']}\n\n"
        f"Evidence:\n{chunks_text}{alignment_note}\n\n"
        "Answer the question clearly, citing which document(s) support your answer. "
        "If an alignment score is present, explain what's driving it."
    )
    state["answer"] = _llm.invoke(prompt).content.strip()
    return state


def log_interaction(state: ClimateLensState) -> ClimateLensState:
    """Writes to both MongoDB (full chat turn + retrieved chunks) and
    SQL (structured analytics row) - this is the node that demonstrates
    why both databases exist in this system."""
    start = state.get("start_time", time.time())
    latency_ms = int((time.time() - start) * 1000)

    _mongo.append_turn(
        session_id=state["session_id"],
        role="assistant",
        content=state["answer"],
        retrieved_chunks=state.get("retrieved_chunks", []),
        alignment_score=state.get("alignment_score"),
    )

    entity = None
    if state.get("retrieved_chunks"):
        entity = state["retrieved_chunks"][0].get("entity_name")

    _sql.log_query(
        session_id=state["session_id"],
        query_text=state["query"],
        entity_queried=entity,
        target_doc_type=state.get("target_doc_type"),
        alignment_score=state.get("alignment_score"),
        latency_ms=latency_ms,
    )
    return state
