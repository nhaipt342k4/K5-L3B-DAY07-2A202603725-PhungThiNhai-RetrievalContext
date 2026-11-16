"""Export the Day 6 Chroma index (strategy D) to the Day 7 chunk contract.

Reads the persisted Chroma collection directly (no re-chunking, no re-embedding), so
dense search and the lexical retrievers see exactly the same 148 chunks and IDs.

    python -m src.day7.export_day6
"""
import chromadb

from .common import CHROMA_PATH, CHUNKS_PATH, DAY6_COLLECTION, write_json


def main() -> None:
    if not CHROMA_PATH.exists():
        raise SystemExit(f"Day 6 Chroma store not found at {CHROMA_PATH}; set DAY6_STARTER.")
    col = chromadb.PersistentClient(path=str(CHROMA_PATH)).get_collection(DAY6_COLLECTION)
    got = col.get(include=["documents", "metadatas"])
    chunks = []
    for cid, doc, meta in zip(got["ids"], got["documents"], got["metadatas"]):
        meta = dict(meta)
        chunks.append({"id": cid, "text": doc, "source_id": meta.pop("document_id"),
                       "source": meta.pop("source"), "page": meta.pop("page"), "metadata": meta})
    chunks.sort(key=lambda c: c["id"])
    if len({c["id"] for c in chunks}) != len(chunks):
        raise SystemExit("duplicate chunk IDs in Day 6 collection")
    write_json(CHUNKS_PATH, chunks)
    pages = sorted({c["page"] for c in chunks})
    print(f"Exported {len(chunks)} chunks from {DAY6_COLLECTION} ({len(pages)} pages {pages[0]}-{pages[-1]}) -> {CHUNKS_PATH}")


if __name__ == "__main__":
    main()
