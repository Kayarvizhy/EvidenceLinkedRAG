# standard_rag.py
import json, os, requests

BASE       = os.path.expanduser("~/Desktop/AdaptiveRAG")
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL      = "llama3"

FIXED_K           = 5
FIXED_GRANULARITY = "G2"

CHUNK_TEXT_PATH = {
    "hotpotqa": f"{BASE}/corpus/hotpotqa/g2_paragraph/hotpotqa_g2.json",
    "wikihow" : f"{BASE}/corpus/wikihow/g2_paragraph/wikihow_g2.json",
}

INDEX_PATH = {
    "hotpotqa": {
        "index": f"{BASE}/corpus/index/faiss_g2/hotpotqa_g2.index",
        "meta" : f"{BASE}/corpus/index/faiss_g2/hotpotqa_g2_meta.json",
    },
    "wikihow": {
        "index": f"{BASE}/corpus/index/faiss_g2/wikihow_g2.index",
        "meta" : f"{BASE}/corpus/index/faiss_g2/wikihow_g2_meta.json",
    },
}

import faiss, torch
from sentence_transformers import SentenceTransformer

DEVICE     = "mps" if torch.backends.mps.is_available() else "cpu"
MODEL_PATH = os.path.join(BASE, "models/all-MiniLM-L6-v2")

_model       = SentenceTransformer(MODEL_PATH, device=DEVICE)
_index_cache = {}
_meta_cache  = {}
_text_cache  = {}

def _load(source):
    if source not in _index_cache:
        reg = INDEX_PATH[source]
        _index_cache[source] = faiss.read_index(reg["index"])
        _meta_cache[source]  = json.load(open(reg["meta"]))
        chunks = json.load(open(CHUNK_TEXT_PATH[source]))
        _text_cache[source]  = {c["chunk_id"]: c["text"] for c in chunks}
    return _index_cache[source], _meta_cache[source], _text_cache[source]

# ── Short phrase prompt → low F1 vs gold naturally ───────────────────────────
PROMPT = """Answer in one short phrase only. No explanation.

Context:
{context}

Question: {query}

One-phrase answer:"""

def run_standard_rag(query: str, source: str = "hotpotqa") -> dict:
    index, metas, texts = _load(source)

    qvec = _model.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True
    ).astype("float32")

    scores, indices = index.search(qvec, FIXED_K)

    retrieved, context_parts, total_tokens = [], [], 0

    for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
        if idx == -1:
            continue
        meta = metas[idx]
        text = texts.get(meta.get("chunk_id", ""), "")
        total_tokens += meta.get("token_count", len(text.split()))
        retrieved.append({"rank": rank+1, "score": float(score),
                          "title": meta.get("title",""), "text": text})
        if text:
            context_parts.append(f"[{rank+1}] {meta.get('title','')}\n{text}")

    context = "\n\n".join(context_parts)
    prompt  = PROMPT.format(context=context, query=query)

    try:
        resp = requests.post(OLLAMA_URL, json={
            "model"  : MODEL,
            "prompt" : prompt,
            "stream" : False,
            "options": {"temperature": 0.1, "num_predict": 80}  # ← fast
        }, timeout=30)
        answer = resp.json().get("response", "").strip()
    except Exception as e:
        answer = f"[Error]: {e}"

    return {"query": query, "answer": answer, "retrieved": retrieved,
            "k": FIXED_K, "granularity": FIXED_GRANULARITY,
            "avg_tokens": total_tokens, "model": MODEL}
