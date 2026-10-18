"""
Loads raw documents from data/sample_docs and pushes them into MongoDB.

Suggested sources (all public, no auth):
  - IPCC AR6 Synthesis Report:
      https://www.ipcc.ch/report/ar6/syr/
  - UNFCCC NDC Registry (national pledges):
      https://unfccc.int/NDCREG
  - A handful of public corporate sustainability/ESG reports (PDF, from
    company investor-relations pages)

Drop the PDFs into data/sample_docs/{ipcc,ndc_pledge,esg_report}/ and run
this script. Keep it to 1-2 IPCC docs, 2-3 NDC pledges, 2-3 ESG reports —
enough to demonstrate cross-document comparison without needing a huge corpus.
"""

import os
import uuid

from pypdf import PdfReader

from db.mongo_client import ClimateLensMongo

SOURCE_DIRS = {
    "ipcc": "data/sample_docs/ipcc",
    "ndc_pledge": "data/sample_docs/ndc_pledge",
    "esg_report": "data/sample_docs/esg_report",
}


def extract_text_from_pdf(path: str) -> str:
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def load_all(mongo: ClimateLensMongo) -> list[dict]:
    """Returns a list of {doc_id, source_type, title, raw_text, metadata} for
    downstream chunking/embedding."""
    loaded = []
    for source_type, dir_path in SOURCE_DIRS.items():
        if not os.path.isdir(dir_path):
            continue
        for fname in os.listdir(dir_path):
            if not fname.lower().endswith(".pdf"):
                continue
            full_path = os.path.join(dir_path, fname)
            text = extract_text_from_pdf(full_path)
            doc_id = str(uuid.uuid5(uuid.NAMESPACE_URL, full_path))
            title = os.path.splitext(fname)[0]

            # entity_name is a light heuristic - in production you'd parse
            # this from the document metadata/filename convention, e.g.
            # "India_NDC_2025.pdf" -> entity_name = "India"
            entity_name = title.split("_")[0] if "_" in title else title

            metadata = {"entity_name": entity_name, "filename": fname}

            mongo.insert_document(doc_id, source_type, title, text, metadata)
            loaded.append({
                "doc_id": doc_id,
                "source_type": source_type,
                "title": title,
                "raw_text": text,
                "metadata": metadata,
            })
    return loaded


if __name__ == "__main__":
    mongo = ClimateLensMongo()
    docs = load_all(mongo)
    print(f"Loaded {len(docs)} documents into MongoDB.")
    for d in docs:
        print(f"  [{d['source_type']}] {d['title']}")
