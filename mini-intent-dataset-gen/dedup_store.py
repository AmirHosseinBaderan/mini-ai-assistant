import uuid
from typing import List, Optional, Tuple

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


class DedupStore:
    """Stores accepted (text, label, embedding) rows in Qdrant and checks new
    candidates against nearby neighbors before they're accepted:
      - if a near neighbor has a DIFFERENT label -> reject as a contradiction
      - if a near neighbor has the SAME label and is nearly identical -> reject as a duplicate
    """

    def __init__(self, client: QdrantClient, collection: str, vector_size: int,
                 duplicate_threshold: float = 0.97, contradiction_threshold: float = 0.90):
        self.client = client
        self.collection = collection
        self.duplicate_threshold = duplicate_threshold
        self.contradiction_threshold = contradiction_threshold

        existing = [c.name for c in self.client.get_collections().collections]
        if collection not in existing:
            self.client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )

    def check_and_maybe_reject(self, embedding: List[float], label: str) -> Tuple[bool, Optional[str]]:
        """Returns (accepted, reason_if_rejected)."""
        hits = self.client.query_points(
            collection_name=self.collection,
            query_vector=embedding,
            limit=5,
            score_threshold=self.contradiction_threshold,
        )
        for hit in hits:
            hit_label = hit.payload.get("label")
            if hit_label != label:
                return False, f"contradiction (similarity={hit.score:.3f} vs existing label={hit_label})"
            if hit.score >= self.duplicate_threshold:
                return False, f"duplicate (similarity={hit.score:.3f})"
        return True, None

    def add(self, text: str, label: str, lang: str, embedding: List[float]):
        self.client.upsert(
            collection_name=self.collection,
            points=[
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={"text": text, "label": label, "lang": lang},
                )
            ],
        )
