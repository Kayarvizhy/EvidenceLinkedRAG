# build_index.py
import json, os, gc
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch

# ── device ────────────────────────────────────────────────────────────────────
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"[INFO] Using device : {DEVICE}")

MODEL_NAME = r"C:\Users\bsshr\OneDrive\Desktop\AdaptiveRAG\models\all-MiniLM-L6-v2"

print(f"[INFO] Loading model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME, device=DEVICE)
DIM   = model.get_sentence_embedding_dimension()
print(f"[INFO] Embedding dim: {DIM}")

BASE = r"C:\Users\bsshr\OneDrive\Desktop\AdaptiveRAG"

# ── files to index ────────────────────────────────────────────────────────────
INDEX_JOBS = [
    # (chunk_json_path,                                    faiss_index_out,                          metadata_out,                              level)
   # (f"{BASE}/corpus/hotpotqa/g1_sentence/hotpotqa_g1.json",  f"{BASE}/corpus/index/faiss_g1/hotpotqa_g1.index",  f"{BASE}/corpus/index/faiss_g1/hotpotqa_g1_meta.json",  "G1"),
   # (f"{BASE}/corpus/hotpotqa/g2_paragraph/hotpotqa_g2.json", f"{BASE}/corpus/index/faiss_g2/hotpotqa_g2.index",  f"{BASE}/corpus/index/faiss_g2/hotpotqa_g2_meta.json",  "G2"),
   # (f"{BASE}/corpus/hotpotqa/g3_section/hotpotqa_g3.json",   f"{BASE}/corpus/index/faiss_g3/hotpotqa_g3.index",  f"{BASE}/corpus/index/faiss_g3/hotpotqa_g3_meta.json",  "G3"),
    (f"{BASE}/corpus/wikihow/g1_sentence/wikihow_g1.json",    f"{BASE}/corpus/index/faiss_g1/wikihow_g1.index",   f"{BASE}/corpus/index/faiss_g1/wikihow_g1_meta.json",   "G1"),
    (f"{BASE}/corpus/wikihow/g2_paragraph/wikihow_g2.json",   f"{BASE}/corpus/index/faiss_g2/wikihow_g2.index",  f"{BASE}/corpus/index/faiss_g2/wikihow_g2_meta.json",   "G2"),
    (f"{BASE}/corpus/wikihow/g3_section/wikihow_g3.json",     f"{BASE}/corpus/index/faiss_g3/wikihow_g3.index",  f"{BASE}/corpus/index/faiss_g3/wikihow_g3_meta.json",   "G3"),
]

BATCH_SIZE = 256

# ── build each index ──────────────────────────────────────────────────────────
for chunk_path, index_path, meta_path, level in INDEX_JOBS:
    print(f"\n{'='*60}")
    print(f"[INFO] Processing : {os.path.basename(chunk_path)}")

    chunks = json.load(open(chunk_path))
    texts  = [c["text"] for c in chunks]
    metas  = [{k: v for k, v in c.items() if k != "text"} for c in chunks]

    print(f"   Chunks         : {len(texts)}")
    print(f"   Granularity    : {level}")

    # embed in batches
    all_embeddings = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc=f"Embedding {level}"):
        batch = texts[i : i + BATCH_SIZE]
        embs  = model.encode(
            batch,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        all_embeddings.append(embs)

    embeddings = np.vstack(all_embeddings).astype("float32")
    print(f"   Embeddings     : {embeddings.shape}")

    # build FAISS flat IP index (cosine via normalized vectors)
    index = faiss.IndexFlatIP(DIM)
    index.add(embeddings)
    print(f"   FAISS vectors  : {index.ntotal}")

    # save index + metadata
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    faiss.write_index(index, index_path)
    json.dump(metas, open(meta_path, "w"), indent=2)

    print(f"✅ Saved index     → {index_path}")
    print(f"✅ Saved metadata  → {meta_path}")

    # free memory
    del embeddings, all_embeddings, chunks, texts, metas, index
    gc.collect()

print("\n" + "="*60)
print("✅ ALL FAISS INDEXES BUILT SUCCESSFULLY!")
print("="*60)
print(f"   Model used     : {MODEL_NAME}")
print(f"   Device used    : {DEVICE}")
print(f"   Granularities  : G1 / G2 / G3")
print(f"   Datasets       : HotpotQA + WikiHow")
