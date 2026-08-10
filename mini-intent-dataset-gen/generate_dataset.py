"""
Grows the RAG/CHAT intent dataset using a local Ollama model for paraphrasing
and Qdrant to reject anything that contradicts or duplicates existing data.

Usage:
    python generate_dataset.py --seed seed.jsonl --target-total 3000

Env vars (see config.py for defaults):
    OLLAMA_HOST, OLLAMA_MODEL, OLLAMA_EMBEDDING_MODEL,
    QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION
"""
import argparse
import json
import random
import re
import sys

from qdrant_client import QdrantClient
from tqdm import tqdm

from config import load_config
from ollama_client import OllamaClient
from dedup_store import DedupStore
from dotenv import load_dotenv,find_dotenv

load_dotenv(find_dotenv(),verbose=True)

PROMPT_TEMPLATE = {
    "rag": {
        "en": (
            "Rewrite the following sentence in {n} different ways, keeping it in English. "
            "The sentence expresses a request that needs looking something up in documents, "
            "a knowledge base, or a database (a RAG / retrieval intent). "
            "Keep each rewrite clearly a lookup/retrieval request, vary the wording and length. "
            "Output exactly {n} lines, one rewrite per line, no numbering, no quotes, no extra text.\n\n"
            "Sentence: {text}"
        ),
        "fa": (
            "جمله‌ی زیر را به {n} شکل متفاوت اما هم‌معنی بازنویسی کن، به فارسی. "
            "این جمله یک درخواست است که نیاز به جست‌وجو در اسناد، نالج‌بیس یا دیتابیس دارد (نیت RAG). "
            "هر بازنویسی باید همچنان به‌وضوح یک درخواست جست‌وجو/بازیابی باشد، طول و کلمات را متنوع کن. "
            "دقیقاً {n} خط خروجی بده، هر بازنویسی در یک خط، بدون شماره‌گذاری، بدون گیومه، بدون متن اضافه.\n\n"
            "جمله: {text}"
        ),
    },
    "chat": {
        "en": (
            "Rewrite the following sentence in {n} different ways, keeping it in English. "
            "The sentence is casual conversation, a greeting, small talk, or a personal statement "
            "that does NOT need looking anything up (a plain CHAT intent, no retrieval). "
            "Keep each rewrite clearly casual conversation, vary the wording and length. "
            "Output exactly {n} lines, one rewrite per line, no numbering, no quotes, no extra text.\n\n"
            "Sentence: {text}"
        ),
        "fa": (
            "جمله‌ی زیر را به {n} شکل متفاوت اما هم‌معنی بازنویسی کن، به فارسی. "
            "این جمله یک مکالمه‌ی معمولی، سلام و احوال‌پرسی یا یک جمله‌ی شخصی است که نیاز به "
            "هیچ جست‌وجویی ندارد (نیت CHAT ساده، بدون بازیابی). "
            "هر بازنویسی باید همچنان به‌وضوح مکالمه‌ی معمولی باشد، طول و کلمات را متنوع کن. "
            "دقیقاً {n} خط خروجی بده، هر بازنویسی در یک خط، بدون شماره‌گذاری، بدون گیومه، بدون متن اضافه.\n\n"
            "جمله: {text}"
        ),
    },
}

NUMBERING_RE = re.compile(r"^\s*[\d]+[\.\)\-]\s*")
QUOTE_RE = re.compile(r'^["\'"\u00ab\u00bb]|["\'"\u00ab\u00bb]$')


def guess_lang(text: str) -> str:
    return "fa" if re.search(r"[\u0600-\u06FF]", text) else "en"


def parse_generated_lines(raw: str, n: int):
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    cleaned = []
    for l in lines:
        l = NUMBERING_RE.sub("", l)
        l = QUOTE_RE.sub("", l).strip()
        if l:
            cleaned.append(l)
    return cleaned[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", default="seed.jsonl")
    ap.add_argument("--target-total", type=int, default=3000,
                     help="Stop once this many accepted rows exist (seed + generated)")
    ap.add_argument("--per-seed-batch", type=int, default=6,
                     help="How many paraphrases to request per generation call")
    ap.add_argument("--train-ratio", type=float, default=0.85)
    ap.add_argument("--out-dir", default="output")
    args = ap.parse_args()

    cfg = load_config()
    ollama = OllamaClient(cfg.ollama_host, cfg.ollama_model, cfg.ollama_embedding_model)
    qdrant = QdrantClient(host=cfg.qdrant_host, port=cfg.qdrant_port)

    seeds = []
    with open(args.seed, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["lang"] = row.get("lang") or guess_lang(row["text"])
            seeds.append(row)

    if not seeds:
        print("No seed rows found.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(seeds)} seed rows.")
    print("Probing embedding model to get vector size...")
    probe_vec = ollama.embed(seeds[0]["text"])
    vector_size = len(probe_vec)
    print(f"Embedding size: {vector_size}")

    store = DedupStore(qdrant, cfg.qdrant_collection, vector_size)

    accepted = []
    rejected_count = 0

    # 1) seed the store with the original (already contradiction-free) seed rows
    print("Seeding dedup store with original seed rows...")
    for row in tqdm(seeds):
        emb = ollama.embed(row["text"])
        ok, reason = store.check_and_maybe_reject(emb, row["label"])
        if ok:
            store.add(row["text"], row["label"], row["lang"], emb)
            accepted.append(row)
        # if a seed row itself collides, just skip re-adding it (already effectively present)

    print(f"Seed rows accepted into store: {len(accepted)}")

    # 2) grow the dataset by generating paraphrases of seed rows, round-robin,
    #    until target-total is reached or we run out of useful attempts
    pbar = tqdm(total=args.target_total, initial=len(accepted))
    max_rounds = 50
    round_i = 0
    while len(accepted) < args.target_total and round_i < max_rounds:
        round_i += 1
        random.shuffle(seeds)
        for row in seeds:
            if len(accepted) >= args.target_total:
                break
            label, lang, text = row["label"], row["lang"], row["text"]
            prompt = PROMPT_TEMPLATE[label][lang].format(n=args.per_seed_batch, text=text)
            try:
                raw = ollama.generate(prompt)
            except Exception as e:
                print(f"generation error, skipping: {e}", file=sys.stderr)
                continue
            candidates = parse_generated_lines(raw, args.per_seed_batch)
            for cand in candidates:
                if len(cand) < 3:
                    continue
                try:
                    emb = ollama.embed(cand)
                except Exception as e:
                    print(f"embedding error, skipping: {e}", file=sys.stderr)
                    continue
                ok, reason = store.check_and_maybe_reject(emb, label)
                if ok:
                    store.add(cand, label, lang, emb)
                    accepted.append({"text": cand, "label": label, "lang": lang})
                    pbar.update(1)
                else:
                    rejected_count += 1
    pbar.close()

    print(f"Total accepted: {len(accepted)} | rejected (dupe/contradiction): {rejected_count}")

    # 3) balance classes, then split train/val
    rag_rows = [r for r in accepted if r["label"] == "rag"]
    chat_rows = [r for r in accepted if r["label"] == "chat"]
    m = min(len(rag_rows), len(chat_rows))
    random.shuffle(rag_rows)
    random.shuffle(chat_rows)
    balanced = rag_rows[:m] + chat_rows[:m]
    random.shuffle(balanced)

    split = int(len(balanced) * args.train_ratio)
    train_rows = balanced[:split]
    val_rows = balanced[split:]

    import os
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "train.jsonl"), "w", encoding="utf-8") as f:
        for r in train_rows:
            f.write(json.dumps({"text": r["text"], "label": r["label"]}, ensure_ascii=False) + "\n")
    with open(os.path.join(args.out_dir, "val.jsonl"), "w", encoding="utf-8") as f:
        for r in val_rows:
            f.write(json.dumps({"text": r["text"], "label": r["label"]}, ensure_ascii=False) + "\n")

    print(f"Wrote {len(train_rows)} train rows and {len(val_rows)} val rows to {args.out_dir}/")
    print(f"Balanced classes: rag={m}, chat={m}")


if __name__ == "__main__":
    main()
