"""
MCP server for ClimateLens.

Exposes two tools over the Model Context Protocol so ANY MCP-compatible
client (Claude Desktop, other agents) can use ClimateLens's retrieval and
alignment-scoring capabilities without going through the FastAPI app.

Tools:
  - search_climate_docs(query, source_type?) -> retrieved chunks
  - check_pledge_alignment(entity, target_doc_type) -> full alignment
    analysis by running the LangGraph pipeline
  - get_query_stats(entity?) -> structured analytics from SQL

Run with: python -m mcp_server.server
"""

import asyncio
import os

from dotenv import load_dotenv

load_dotenv()

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from db.sql_models import ClimateLensSQL
from graph.build_graph import run_query

CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_store")

app = Server("climatelens")

_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
_store = Chroma(persist_directory=CHROMA_DIR, embedding_function=_embeddings)
_sql = ClimateLensSQL()


@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="search_climate_docs",
            description=(
                "Semantic search over climate documents (IPCC reports, NDC "
                "pledges, ESG reports). Returns the most relevant excerpts."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural language search query"},
                    "source_type": {
                        "type": "string",
                        "enum": ["ipcc", "ndc_pledge", "esg_report"],
                        "description": "Optional filter to a single document type",
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="check_pledge_alignment",
            description=(
                "Runs the full ClimateLens agentic pipeline to check whether "
                "a country's NDC pledge or a company's ESG report aligns with "
                "IPCC benchmarks, returning an alignment score and flagged gaps."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The specific claim or question to check"},
                    "target_doc_type": {
                        "type": "string",
                        "enum": ["ndc_pledge", "esg_report"],
                    },
                },
                "required": ["query", "target_doc_type"],
            },
        ),
        Tool(
            name="get_query_stats",
            description="Returns structured analytics: query counts, average alignment score, and latency, optionally filtered by entity.",
            inputSchema={
                "type": "object",
                "properties": {
                    "entity": {"type": "string", "description": "Optional country/company name filter"},
                },
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "search_climate_docs":
        filter_dict = {"source_type": arguments["source_type"]} if arguments.get("source_type") else None
        results = _store.similarity_search(arguments["query"], k=5, filter=filter_dict)
        text = "\n---\n".join(
            f"[{d.metadata.get('title')}] {d.page_content[:500]}" for d in results
        )
        return [TextContent(type="text", text=text or "No results found.")]

    if name == "check_pledge_alignment":
        result = run_query(
            session_id="mcp-session",
            query=arguments["query"],
            target_doc_type=arguments["target_doc_type"],
        )
        text = (
            f"Answer: {result['answer']}\n\n"
            f"Alignment score: {result.get('alignment_score')}\n"
            f"Flags: {', '.join(result.get('flags', [])) or 'None'}"
        )
        return [TextContent(type="text", text=text)]

    if name == "get_query_stats":
        stats = _sql.get_query_stats(entity_queried=arguments.get("entity"))
        return [TextContent(type="text", text=str(stats))]

    raise ValueError(f"Unknown tool: {name}")


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
