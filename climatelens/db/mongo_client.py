"""
MongoDB layer for ClimateLens.

Why MongoDB here and not just SQL: the documents we ingest (IPCC reports,
NDC pledges, ESG disclosures) have wildly different structures per source,
and chat/session history is naturally nested (a session has many turns,
each turn has retrieved chunks + metadata attached). Schema flexibility
matters more here than relational integrity, so Mongo is the right tool
for THIS data, while structured analytics goes to SQL (see sql_models.py).
"""

import os
from datetime import datetime, timezone
from typing import Any, Optional

from dotenv import load_dotenv

load_dotenv()

from pymongo import MongoClient
from pymongo.collection import Collection


class ClimateLensMongo:
    def __init__(self, uri: Optional[str] = None, db_name: Optional[str] = None):
        uri = uri or os.getenv("MONGO_URI", "mongodb://localhost:27017")
        db_name = db_name or os.getenv("MONGO_DB_NAME", "climatelens")
        self.client = MongoClient(uri)
        self.db = self.client[db_name]

        self.documents: Collection = self.db["documents"]
        self.chat_sessions: Collection = self.db["chat_sessions"]

        # Helpful indexes
        self.documents.create_index("source_type")
        self.documents.create_index("doc_id", unique=True)
        self.chat_sessions.create_index("session_id")

    # ---------- Documents ----------
    def insert_document(self, doc_id: str, source_type: str, title: str,
                         raw_text: str, metadata: dict[str, Any]) -> None:
        """source_type is one of: 'ipcc', 'ndc_pledge', 'esg_report'."""
        self.documents.update_one(
            {"doc_id": doc_id},
            {"$set": {
                "doc_id": doc_id,
                "source_type": source_type,
                "title": title,
                "raw_text": raw_text,
                "metadata": metadata,
                "ingested_at": datetime.now(timezone.utc),
            }},
            upsert=True,
        )

    def get_document(self, doc_id: str) -> Optional[dict]:
        return self.documents.find_one({"doc_id": doc_id})

    def list_documents(self, source_type: Optional[str] = None) -> list[dict]:
        query = {"source_type": source_type} if source_type else {}
        return list(self.documents.find(query, {"raw_text": 0}))

    # ---------- Chat / session history ----------
    def append_turn(self, session_id: str, role: str, content: str,
                     retrieved_chunks: Optional[list[dict]] = None,
                     alignment_score: Optional[float] = None) -> None:
        turn = {
            "role": role,
            "content": content,
            "retrieved_chunks": retrieved_chunks or [],
            "alignment_score": alignment_score,
            "timestamp": datetime.now(timezone.utc),
        }
        self.chat_sessions.update_one(
            {"session_id": session_id},
            {"$push": {"turns": turn},
             "$setOnInsert": {"session_id": session_id, "created_at": datetime.now(timezone.utc)}},
            upsert=True,
        )

    def get_session(self, session_id: str) -> Optional[dict]:
        return self.chat_sessions.find_one({"session_id": session_id})
