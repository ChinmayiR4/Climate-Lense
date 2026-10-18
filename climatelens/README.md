# ClimateLens

Agentic RAG system that checks whether a country's climate pledge (NDC) or
a company's ESG report aligns with IPCC benchmarks — not just document
Q&A, but cross-document comparison with a computed alignment score and
flagged gaps.

## Why this exists
Built to demonstrate an end-to-end agentic AI stack: LangGraph orchestration,
RAG retrieval, MCP tool exposure, MongoDB + SQL working together for
different data shapes, and a FastAPI backend.

## Architecture

```
                     ┌─────────────────┐
   User query  ─────▶│   FastAPI app    │◀───── React/Next.js frontend
                     └────────┬─────────┘
                              │
                     ┌────────▼─────────┐
                     │  LangGraph        │
                     │  ┌─────────────┐  │
                     │  │  retrieve   │◀─┼──┐ (loop back on weak retrieval,
                     │  └──────┬──────┘  │  │  up to 2 attempts)
                     │  ┌──────▼──────┐  │  │
                     │  │grade_       │──┼──┘
                     │  │relevance    │  │
                     │  └──────┬──────┘  │
                     │  ┌──────▼──────┐  │
                     │  │compare_to_  │  │  ← queries IPCC chunks as benchmark
                     │  │benchmark    │  │
                     │  └──────┬──────┘  │
                     │  ┌──────▼──────┐  │
                     │  │generate_    │  │
                     │  │answer       │  │
                     │  └──────┬──────┘  │
                     │  ┌──────▼──────┐  │
                     │  │log_         │  │
                     │  │interaction  │  │
                     │  └─────────────┘  │
                     └────────┬──────────┘
                              │
              ┌───────────────┴────────────────┐
              ▼                                 ▼
        ┌───────────┐                    ┌─────────────┐
        │  MongoDB   │                    │   SQLite    │
        │ (docs,     │                    │ (structured │
        │  chat      │                    │  analytics: │
        │  history)  │                    │  query      │
        └───────────┘                    │  stats,     │
                                          │  alignment) │
                                          └─────────────┘

        Chroma vector store (embeddings for IPCC / NDC / ESG chunks)
                              ▲
                              │
                     ┌────────┴─────────┐
                     │   MCP server      │  ← exposes search_climate_docs,
                     │  (mcp_server/)    │    check_pledge_alignment,
                     └───────────────────┘    get_query_stats to any MCP client
```

## Why each database
- **MongoDB**: raw documents and chat/session history are naturally nested
  and schema-varies per source (IPCC report ≠ NDC pledge ≠ ESG report
  structure). Document store fits.
- **SQLite/SQL**: query analytics (counts, average alignment score per
  entity, latency) are inherently relational — GROUP BY / aggregate
  queries, not document lookups.
- **Chroma**: vector similarity search over chunked, embedded text.

## Why LangGraph, not just a LangChain chain
The pipeline needs conditional control flow: if the first retrieval pass
returns weak evidence, `grade_relevance` rewrites the query and loops back
to `retrieve` (up to 2 attempts) instead of answering on thin evidence.
A linear chain can't branch or loop like this.

## Setup

```bash
cp .env.example .env   # fill in OPENAI_API_KEY
docker compose up -d mongo

pip install -r requirements.txt

# Drop a few PDFs into data/sample_docs/{ipcc,ndc_pledge,esg_report}/
python -m ingest.chunk_and_embed

# Run the API
uvicorn api.main:app --reload

# Or run a query directly
python -m graph.build_graph

# Or run the MCP server (for use with an MCP client)
python -m mcp_server.server
```

## Sample data sources (all public, no auth)
- IPCC AR6 Synthesis Report: https://www.ipcc.ch/report/ar6/syr/
- UNFCCC NDC Registry: https://unfccc.int/NDCREG
- Any public company sustainability report (investor relations PDF)

## Interview talking points
- **RAG chunking decisions**: 800-char chunks with 150 overlap, chosen to
  keep a full policy claim + its qualifier together without pulling whole
  pages (see `ingest/chunk_and_embed.py`).
- **Why LangGraph over LangChain alone**: conditional retry loop in the
  graph, not a fixed pipeline.
- **MCP tool design**: `check_pledge_alignment` isn't a thin wrapper — it
  runs the full graph and returns a computed judgment (score + flags),
  which is what makes it a genuine "tool" rather than just retrieval.
- **Mongo vs SQL split**: be ready to explain why each data shape landed
  where it did (see "Why each database" above) — this is the most common
  follow-up question.
- **Known limitation to be upfront about**: the alignment score is an LLM
  judgment call, not a verified fact — worth saying this out loud before
  an interviewer asks, it shows maturity rather than overselling accuracy.
