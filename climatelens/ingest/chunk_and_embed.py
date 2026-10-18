"""
Chunks loaded documents and embeds them into a Chroma vector store,
partitioned by source_type as metadata so retrieval can be filtered
(e.g. "only search IPCC benchmarks" vs "only search this company's ESG report").
"""

import os

from dotenv import load_dotenv

load_dotenv()

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from db.mongo_client import ClimateLensMongo
from db.sql_models import ClimateLensSQL
from ingest.load_documents import load_all

CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_store")

# Chunk size/overlap chosen for dense policy-language PDFs: large enough to
# keep a full claim + its qualifying clause together, small enough that
# retrieval stays precise rather than pulling whole pages.
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " "],
)


def build_vector_store() -> Chroma:
    mongo = ClimateLensMongo()
    sql = ClimateLensSQL()
    docs = load_all(mongo)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    store = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

    for doc in docs:
        chunks = splitter.split_text(doc["raw_text"])
        metadatas = [{
            "doc_id": doc["doc_id"],
            "source_type": doc["source_type"],
            "title": doc["title"],
            "entity_name": doc["metadata"].get("entity_name"),
            "chunk_index": i,
        } for i in range(len(chunks))]

        ids = [f"{doc['doc_id']}_{i}" for i in range(len(chunks))]
        store.add_texts(texts=chunks, metadatas=metadatas, ids=ids)

        sql.register_document(
            doc_id=doc["doc_id"],
            source_type=doc["source_type"],
            entity_name=doc["metadata"].get("entity_name"),
            chunk_count=len(chunks),
        )
        print(f"Embedded {len(chunks)} chunks for [{doc['source_type']}] {doc['title']}")

    store.persist()
    return store


if __name__ == "__main__":
    build_vector_store()
