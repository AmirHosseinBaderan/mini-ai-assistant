# Intent Dataset Generator (RAG vs CHAT)

Grows a small, clean seed dataset into a much larger one using your local
Ollama model for paraphrasing, and Qdrant to reject anything that
contradicts or duplicates existing rows — which is what caused your
classifier's weird behavior in the first place (near-identical sentences
with different labels).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set your environment (matches what you gave me):

```bash
export OLLAMA_HOST=http://ollama.cloudito.local
export OLLAMA_MODEL=gemma3:1b
export OLLAMA_EMBEDDING_MODEL=qwen3-embedding:latest
export QDRANT_HOST=192.168.0.247
```

## 1. Grow the dataset from the seed file

`seed.jsonl` (included, 262 rows, FA+EN, RAG/CHAT, already contradiction-free)
is the starting point.

```bash
python generate_dataset.py --seed seed.jsonl --target-total 3000
```

This will:
1. Embed and load all seed rows into a fresh Qdrant collection (`intent_dataset`).
2. Repeatedly ask `gemma3:1b` to paraphrase seed sentences (in the same
   language, keeping the same intent).
3. Embed every candidate with `qwen3-embedding` and check it against Qdrant:
   - if it's near a sentence with a **different** label -> rejected (contradiction)
   - if it's near-identical to a sentence with the **same** label -> rejected (duplicate)
   - otherwise -> accepted and added to the store
4. Balances the two classes and writes `output/train.jsonl` and `output/val.jsonl`.

Useful flags:
- `--target-total 3000` — stop once this many accepted rows exist (raise this
  for a bigger dataset; each round makes real API calls, so start smaller to
  sanity-check quality before going for tens of thousands)
- `--per-seed-batch 6` — how many paraphrases to request per call

## 2. (Optional) Audit your existing 200k-item dataset

If you want to find contradictions in your *current* dataset rather than
starting over:

```bash
python audit_dataset.py --input your_200k_dataset.jsonl --sample 5000
```

This samples rows, embeds them, and reports pairs that are semantically very
close but labeled differently — the exact pattern that made your classifier
latch onto proper nouns instead of actual intent. Start with `--sample 5000`
since it's one embedding call per row; increase or set `--sample 0` for a
full pass once you trust the pipeline.

## Notes / limitations

- `gemma3:1b` is a small model — spot-check a few hundred generated rows by
  hand before training on the full output. It will occasionally drift off
  the requested label; the Qdrant contradiction check catches the worst
  cases but isn't perfect.
- The contradiction/duplicate thresholds (`0.90` / `0.97` cosine similarity)
  are reasonable defaults but depend on your embedding model — if you see
  too many false rejections or too many contradictions slipping through,
  adjust them in `dedup_store.py`.
- This only generates **paraphrases of your existing seed patterns** — it
  won't invent genuinely new topics/scenarios. For real coverage growth,
  periodically add new hand-written seed rows for cases your assistant
  actually encounters (e.g. mined from production logs) and re-run.
