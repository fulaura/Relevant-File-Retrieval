from __future__ import annotations

import logging
from typing import Dict, List

from database import database_search, database_setup
from models import encoder


class RetrievalEngine:
    def __init__(self, top_k: int, chunks_per_doc: int, dsn: str | None = None) -> None:
        conn, cur = database_setup.db_connect()
        if dsn:
            logging.warning("Custom DSN is not yet supported; using default connection settings.")
        if not (conn and cur):
            raise RuntimeError("Failed to connect to the vector database.")
        self.conn = conn
        self.cur = cur
        self.top_k = top_k
        self.chunks_per_doc = chunks_per_doc

    def close(self) -> None:
        self.cur.close()
        self.conn.close()

    def search(self, query: str) -> List[Dict]:
        embedded = encoder.encode_text_transformer(query)
        raw = database_search.similar_n(
            self.conn,
            self.cur,
            embedded,
            n=self.top_k * self.chunks_per_doc,
            similarity_threshold=None,
        )
        grouped: Dict[str, Dict] = {}
        for similarity, file_name, content, metadata, path in raw:
            doc = grouped.setdefault(path, {
                "file_name": file_name,
                "path": path,
                "metadata": metadata or {},
                "chunks": []
            })
            doc["chunks"].append({"similarity": similarity, "content": content})

        def best_score(doc: Dict) -> float:
            chunks = doc.get("chunks", [])
            return min(chunk["similarity"] for chunk in chunks) if chunks else float("inf")

        ordered = sorted(grouped.values(), key=best_score)
        for doc in ordered:
            doc["chunks"] = doc["chunks"][: self.chunks_per_doc]
            doc["content"] = "\n\n".join(chunk["content"] for chunk in doc["chunks"])
        return ordered[: self.top_k]
