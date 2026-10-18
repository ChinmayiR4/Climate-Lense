"""
Shared state passed between LangGraph nodes.

Using a TypedDict (not just a plain LangChain chain) because the pipeline
needs to branch and loop: if retrieval quality is weak, we re-retrieve with
a rewritten query instead of answering on thin evidence. That control flow
is the actual reason to reach for LangGraph over a linear LangChain chain.
"""

from typing import Optional, TypedDict


class ClimateLensState(TypedDict, total=False):
    session_id: str
    query: str                       # original user question
    rewritten_query: Optional[str]   # set if we loop back to retrieve again
    retrieval_attempts: int
    _start_time: float               # wall-clock start, used to compute latency_ms
    start_time: float                # wall-clock time.time() when the run began

    retrieved_chunks: list[dict]     # [{text, source_type, title, entity_name, score}]
    relevance_ok: bool               # set by grade_relevance node

    target_doc_type: Optional[str]   # 'ndc_pledge' or 'esg_report', if comparing
    benchmark_chunks: list[dict]     # IPCC chunks used as the benchmark, if comparing

    answer: str
    alignment_score: Optional[float] # 0.0-1.0, how well the claim matches benchmark
    flags: list[str]                 # e.g. ["No quantified emissions target found"]
