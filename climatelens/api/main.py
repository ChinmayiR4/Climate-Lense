"""
FastAPI backend for ClimateLens.

This is the app-facing surface (used by a React/Next.js frontend or Postman);
the MCP server (mcp_server/server.py) is a separate, parallel surface for
agent-to-agent / tool-calling use. Both call into the same LangGraph pipeline
and SQL/Mongo layers, which is the point - the logic isn't duplicated per
interface.
"""

import uuid

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from graph.build_graph import run_query
from db.sql_models import ClimateLensSQL
from db.mongo_client import ClimateLensMongo

app = FastAPI(title="ClimateLens API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten before shipping
    allow_methods=["*"],
    allow_headers=["*"],
)

sql = ClimateLensSQL()
mongo = ClimateLensMongo()


class QueryRequest(BaseModel):
    session_id: str | None = None
    query: str
    target_doc_type: str | None = None  # "ndc_pledge" or "esg_report"


class QueryResponse(BaseModel):
    session_id: str
    answer: str
    alignment_score: float | None
    flags: list[str]
    retrieved_titles: list[str]


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    session_id = req.session_id or str(uuid.uuid4())
    result = run_query(session_id, req.query, req.target_doc_type)
    return QueryResponse(
        session_id=session_id,
        answer=result["answer"],
        alignment_score=result.get("alignment_score"),
        flags=result.get("flags", []),
        retrieved_titles=[c["title"] for c in result.get("retrieved_chunks", [])],
    )


@app.get("/session/{session_id}")
def get_session(session_id: str):
    return mongo.get_session(session_id)


@app.get("/analytics/stats")
def analytics_stats(entity: str | None = None):
    return sql.get_query_stats(entity_queried=entity)


@app.get("/analytics/top-queried")
def top_queried(limit: int = 5):
    return sql.most_queried_documents(limit=limit)


@app.get("/documents")
def list_documents(source_type: str | None = None):
    return mongo.list_documents(source_type=source_type)
