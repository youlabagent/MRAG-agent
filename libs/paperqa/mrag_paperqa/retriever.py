"""Dense retrieval from Qdrant collection using BGE-M3."""
from __future__ import annotations

import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Sequence

from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, MatchAny
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore", message="Local mode is not recommended.*")

DEFAULT_QDRANT = "/BIO1/MRAG/Agent/libs/paperqa/qdrant"
DEFAULT_COLLECTION = "mrag_papers"
DEFAULT_MODEL = "BAAI/bge-m3"


@dataclass
class Passage:
    score: float
    pmid: Optional[str]
    doc_id: str
    title: Optional[str]
    cancer: Optional[str]
    year: Optional[int]
    source: Optional[str]
    section: Optional[str]
    text: str

    def cite(self) -> str:
        if self.pmid:
            return f"PMID:{self.pmid}"
        return f"doc:{self.doc_id}"

    def to_dict(self) -> dict:
        return asdict(self)


class Retriever:
    def __init__(
        self,
        qdrant_path: str = DEFAULT_QDRANT,
        collection: str = DEFAULT_COLLECTION,
        model_name: str = DEFAULT_MODEL,
        device: str = "cuda",
    ):
        self.client = QdrantClient(path=qdrant_path)
        self.collection = collection
        self.model = SentenceTransformer(model_name, device=device)

    def embed(self, query: str):
        return self.model.encode(query, normalize_embeddings=True).tolist()

    def search(
        self,
        query: str,
        top_k: int = 10,
        cancer: Optional[Sequence[str] | str] = None,
        year_min: Optional[int] = None,
        source: Optional[str] = None,
    ) -> List[Passage]:
        must = []
        if cancer:
            if isinstance(cancer, str):
                must.append(FieldCondition(key="cancer", match=MatchValue(value=cancer)))
            else:
                must.append(FieldCondition(key="cancer", match=MatchAny(any=list(cancer))))
        if source:
            must.append(FieldCondition(key="source", match=MatchValue(value=source)))
        qfilter = Filter(must=must) if must else None

        vec = self.embed(query)
        result = self.client.query_points(
            collection_name=self.collection,
            query=vec,
            limit=top_k,
            query_filter=qfilter,
            with_payload=True,
        )
        hits = result.points

        out: List[Passage] = []
        for h in hits:
            p = h.payload or {}
            year = p.get("year")
            if year_min is not None and (year is None or year < year_min):
                continue
            out.append(
                Passage(
                    score=float(h.score),
                    pmid=p.get("pmid"),
                    doc_id=str(p.get("doc_id", "")),
                    title=p.get("title"),
                    cancer=p.get("cancer"),
                    year=year,
                    source=p.get("source"),
                    section=p.get("section"),
                    text=p.get("text", ""),
                )
            )
        return out
