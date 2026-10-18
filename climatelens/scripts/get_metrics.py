"""
Prints real metrics from your own SQL and MongoDB logs - no invented numbers.
Run this after you've done a handful of test queries through build_graph.py
or the FastAPI endpoint, so there's real data to report on.

Usage: python scripts/get_metrics.py
"""

import os
import sys
import sqlite3

# Ensure the project root (parent of scripts/) is importable, regardless of
# how this script is invoked - fixes "ModuleNotFoundError: No module named 'db'"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.mongo_client import ClimateLensMongo


def print_sql_metrics():
    conn = sqlite3.connect("data/analytics.db")

    total_queries = conn.execute("SELECT COUNT(*) FROM queries").fetchone()[0]
    doc_count = conn.execute("SELECT COUNT(*) FROM documents_meta").fetchone()[0]
    total_chunks = conn.execute("SELECT SUM(chunk_count) FROM documents_meta").fetchone()[0]

    print("=== Document / chunk counts ===")
    print(f"Documents indexed: {doc_count}")
    print(f"Total chunks embedded: {total_chunks}")
    print()
    print("By source type:")
    for source_type, count, chunks in conn.execute(
        "SELECT source_type, COUNT(*), SUM(chunk_count) FROM documents_meta GROUP BY source_type"
    ):
        print(f"  {source_type}: {count} document(s), {chunks} chunks")

    print()
    print("=== Query metrics ===")
    print(f"Total queries logged: {total_queries}")

    if total_queries > 0:
        avg_latency = conn.execute(
            "SELECT AVG(latency_ms) FROM queries WHERE latency_ms > 0"
        ).fetchone()[0]
        avg_score = conn.execute(
            "SELECT AVG(alignment_score) FROM queries WHERE alignment_score IS NOT NULL"
        ).fetchone()[0]
        min_latency, max_latency = conn.execute(
            "SELECT MIN(latency_ms), MAX(latency_ms) FROM queries WHERE latency_ms > 0"
        ).fetchone()

        print(f"Average query latency: {avg_latency:.0f} ms" if avg_latency else "Average latency: no valid latency data yet")
        print(f"Latency range: {min_latency}-{max_latency} ms" if min_latency else "")
        print(f"Average alignment score: {avg_score:.2f}" if avg_score is not None else "No alignment scores logged yet")

    conn.close()


def print_mongo_metrics():
    mongo = ClimateLensMongo()
    docs = mongo.list_documents()
    sessions = list(mongo.chat_sessions.find({}, {"turns": 1}))
    total_turns = sum(len(s.get("turns", [])) for s in sessions)

    print()
    print("=== MongoDB ===")
    print(f"Documents stored: {len(docs)}")
    print(f"Chat sessions: {len(sessions)}")
    print(f"Total conversation turns logged: {total_turns}")


if __name__ == "__main__":
    print_sql_metrics()
    print_mongo_metrics()