"""
SQL layer for ClimateLens structured analytics.

Why SQL here and not Mongo: this data IS relational and benefits from it.
We want fast aggregate queries like "average alignment score per entity"
or "which documents get queried most" - that's a GROUP BY + JOIN problem,
not a document-store problem. This is what the JD's "SQL databases
(joins, indexing, query optimization)" requirement is actually testing for.
"""

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


DDL = """
CREATE TABLE IF NOT EXISTS queries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    query_text TEXT NOT NULL,
    entity_queried TEXT,          -- e.g. 'Country: India' or 'Company: Acme Corp'
    target_doc_type TEXT,         -- 'ndc_pledge' or 'esg_report'
    alignment_score REAL,         -- 0.0 to 1.0, how well claim matches IPCC benchmark
    latency_ms INTEGER,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_entity ON queries(entity_queried);
CREATE INDEX IF NOT EXISTS idx_doc_type ON queries(target_doc_type);

CREATE TABLE IF NOT EXISTS documents_meta (
    doc_id TEXT PRIMARY KEY,
    source_type TEXT NOT NULL,
    entity_name TEXT,             -- country or company name
    chunk_count INTEGER,
    indexed_at TEXT NOT NULL
);
"""


class ClimateLensSQL:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or os.getenv("SQLITE_PATH", "./data/analytics.db")
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        with self._connect() as conn:
            conn.executescript(DDL)

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def log_query(self, session_id: str, query_text: str, entity_queried: Optional[str],
                  target_doc_type: Optional[str], alignment_score: Optional[float],
                  latency_ms: int) -> None:
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO queries
                   (session_id, query_text, entity_queried, target_doc_type,
                    alignment_score, latency_ms, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (session_id, query_text, entity_queried, target_doc_type,
                 alignment_score, latency_ms, datetime.now(timezone.utc).isoformat()),
            )

    def register_document(self, doc_id: str, source_type: str,
                           entity_name: Optional[str], chunk_count: int) -> None:
        with self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO documents_meta
                   (doc_id, source_type, entity_name, chunk_count, indexed_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (doc_id, source_type, entity_name, chunk_count,
                 datetime.now(timezone.utc).isoformat()),
            )

    def get_query_stats(self, entity_queried: Optional[str] = None) -> list[dict]:
        """Used by the MCP tool `get_query_stats`."""
        with self._connect() as conn:
            if entity_queried:
                rows = conn.execute(
                    """SELECT entity_queried, target_doc_type,
                              COUNT(*) as query_count,
                              AVG(alignment_score) as avg_alignment,
                              AVG(latency_ms) as avg_latency_ms
                       FROM queries WHERE entity_queried = ?
                       GROUP BY entity_queried, target_doc_type""",
                    (entity_queried,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT entity_queried, target_doc_type,
                              COUNT(*) as query_count,
                              AVG(alignment_score) as avg_alignment,
                              AVG(latency_ms) as avg_latency_ms
                       FROM queries
                       GROUP BY entity_queried, target_doc_type
                       ORDER BY query_count DESC"""
                ).fetchall()
            return [dict(r) for r in rows]

    def most_queried_documents(self, limit: int = 5) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT entity_queried, COUNT(*) as hits
                   FROM queries
                   WHERE entity_queried IS NOT NULL
                   GROUP BY entity_queried
                   ORDER BY hits DESC LIMIT ?""",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]
