# retriever.py
import json, os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

BASE       = os.path.expanduser("~/OneDrive/Desktop/AdaptiveRAG")
MODEL_PATH = os.path.join(BASE, "models/all-MiniLM-L6-v2")

import torch
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

print(f"[Retriever] Loading embedding model on {DEVICE}...")
_model = SentenceTransformer(MODEL_PATH, device=DEVICE)
print(f"[Retriever] Model ready ✅")

INDEX_REGISTRY = {
    "G1": {
        "hotpotqa": {
            "index": f"{BASE}/corpus/index/faiss_g1/hotpotqa_g1.index",
            "meta" : f"{BASE}/corpus/index/faiss_g1/hotpotqa_g1_meta.json",
        },
        "wikihow": {
            "index": f"{BASE}/corpus/index/faiss_g1/wikihow_g1.index",
            "meta" : f"{BASE}/corpus/index/faiss_g1/wikihow_g1_meta.json",
        },
    },
    "G2": {
        "hotpotqa": {
            "index": f"{BASE}/corpus/index/faiss_g2/hotpotqa_g2.index",
            "meta" : f"{BASE}/corpus/index/faiss_g2/hotpotqa_g2_meta.json",
        },
        "wikihow": {
            "index": f"{BASE}/corpus/index/faiss_g2/wikihow_g2.index",
            "meta" : f"{BASE}/corpus/index/faiss_g2/wikihow_g2_meta.json",
        },
    },
    "G3": {
        "hotpotqa": {
            "index": f"{BASE}/corpus/index/faiss_g3/hotpotqa_g3.index",
            "meta" : f"{BASE}/corpus/index/faiss_g3/hotpotqa_g3_meta.json",
        },
        "wikihow": {
            "index": f"{BASE}/corpus/index/faiss_g3/wikihow_g3.index",
            "meta" : f"{BASE}/corpus/index/faiss_g3/wikihow_g3_meta.json",
        },
    },
}

CHUNK_TEXT_PATH = {
    "G1": {
        "hotpotqa": f"{BASE}/corpus/hotpotqa/g1_sentence/hotpotqa_g1.json",
        "wikihow" : f"{BASE}/corpus/wikihow/g1_sentence/wikihow_g1.json",
    },
    "G2": {
        "hotpotqa": f"{BASE}/corpus/hotpotqa/g2_paragraph/hotpotqa_g2.json",
        "wikihow" : f"{BASE}/corpus/wikihow/g2_paragraph/wikihow_g2.json",
    },
    "G3": {
        "hotpotqa": f"{BASE}/corpus/hotpotqa/g3_section/hotpotqa_g3.json",
        "wikihow" : f"{BASE}/corpus/wikihow/g3_section/wikihow_g3.json",
    },
}

_index_cache = {}
_meta_cache  = {}
_text_cache  = {}

def _load_index(granularity: str, source: str):
    key = f"{granularity}_{source}"
    if key not in _index_cache:
        reg = INDEX_REGISTRY[granularity][source]
        print(f"[Retriever] Loading index {key}...")
        _index_cache[key] = faiss.read_index(reg["index"])
        _meta_cache[key]  = json.load(open(reg["meta"]))
        text_path = CHUNK_TEXT_PATH[granularity][source]
        chunks = json.load(open(text_path))
        _text_cache[key] = {c["chunk_id"]: c["text"] for c in chunks}
        print(f"[Retriever] Index {key} loaded ✅  ({_index_cache[key].ntotal} vectors)")
    return _index_cache[key], _meta_cache[key], _text_cache[key]

def _select_source(intent: str) -> str:
    if intent == "procedural":
        return "wikihow"
    return "hotpotqa"

def retrieve(query: str, intent: str, granularity: str, k: int) -> list:
    source = _select_source(intent)
    index, metas, texts = _load_index(granularity, source)

    qvec = _model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    scores, indices = index.search(qvec, k)

    results = []
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
        if idx == -1:
            continue
        meta = metas[idx]
        text = texts.get(meta.get("chunk_id", ""), "")
        results.append({
            "rank"        : rank + 1,
            "score"       : float(score),
            "chunk_id"    : meta.get("chunk_id", ""),
            "title"       : meta.get("title", ""),
            "source"      : meta.get("source", ""),
            "granularity" : meta.get("granularity", ""),
            "type"        : meta.get("type", ""),
            "doc_id"      : meta.get("doc_id", ""),
            "token_count" : meta.get("token_count", 0),
            "text"        : text,
        })

    # deduplicate by title
    seen = set()
    deduped = []
    for r in results:
        t = r.get('title','')
        if t not in seen:
            seen.add(t)
            deduped.append(r)
    # re-rank
    for i,r in enumerate(deduped):
        r['rank'] = i+1
    return deduped
