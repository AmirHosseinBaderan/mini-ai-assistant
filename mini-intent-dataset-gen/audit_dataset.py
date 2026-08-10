"""
Scans an existing JSONL dataset (e.g. your 200k-item file) and reports rows
whose embedding is very close to another row that has a DIFFERENT label.
These are the contradictions that confuse the classifier.

Usage:
    python audit_dataset.py --input big_dataset.jsonl --out contradictions.jsonl \
        --sample 5000

--sample lets you audit a random subset first (recommended for 200k rows,
since this makes one embedding call per row).
"""
import argparse
import json
import random

from qdrant_client import QdrantClient
from tqdm import tqdm

from config import load_config
from ollama_client import OllamaClient


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="contradictions.jsonl")
    ap.add_argument("--sample", type=int, default=5000,
                     help="Randomly sample this many rows to audit (0 = all rows)")
    ap.add_argument("--contradiction-threshold", type=float, default=0.90)
    args = ap.parse_args()

    cfg = load_config()
    ollama = OllamaClient(cfg.ollama_host, cfg.ollama_model, cfg.ollama_embedding_model)
    qdrant = QdrantClient(host=cfg.qdrant_host, port=cfg.qdrant_port)

    rows = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    if args.sample and args.sample < len(rows):
        random.shuffle(rows)
        rows = rows[: args.sample]

    print(f"Auditing {len(rows)} rows...")

    from qdrant_client.models import Distance, VectorParams, PointStruct
    import uuid

    collection = "audit_tmp"
    probe = ollama.embed(rows[0]["text"])
    vector_size = len(probe)
    existing = [c.name for c in qdrant.get_collections().collections]
    if collection in existing:
        qdrant.delete_collection(collection)
    qdrant.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )

    contradictions = []
    for row in tqdm(rows):
        text, label = row["text"], row["label"]
        emb = ollama.embed(text)
        hits = qdrant.search(
            collection_name=collection,
            query_vector=emb,
            limit=3,
            score_threshold=args.contradiction_threshold,
        )
        for hit in hits:
            if hit.payload.get("label") != label:
                contradictions.append({
                    "text_a": text, "label_a": label,
                    "text_b": hit.payload["text"], "label_b": hit.payload["label"],
                    "similarity": hit.score,
                })
        qdrant.upsert(
            collection_name=collection,
            points=[PointStruct(id=str(uuid.uuid4()), vector=emb,
                                 payload={"text": text, "label": label})],
        )

    with open(args.out, "w", encoding="utf-8") as f:
        for c in contradictions:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"Found {len(contradictions)} likely contradictions. Written to {args.out}")


if __name__ == "__main__":
    main()
