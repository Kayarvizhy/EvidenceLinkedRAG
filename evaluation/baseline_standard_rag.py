import os, sys, json, requests
import faiss, torch
from sentence_transformers import SentenceTransformer

BASE_DIR   = os.path.expanduser(
    "~/Desktop/AdaptiveRAG")
DATA_DIR   = os.path.join(BASE_DIR,
    "evaluation/test_sets")
RESULTS    = os.path.join(BASE_DIR, "results")
OLLAMA_URL = "http://localhost:11434/api/generate"

sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(
    BASE_DIR, "evaluation"))
from metrics import (f1_score, rouge_l,
                     hallucination_rate,
                     evidence_overlap,
                     avg, std, print_table)

os.makedirs(RESULTS, exist_ok=True)

DEVICE = ("mps"
          if torch.backends.mps.is_available()
          else "cpu")

# ── Load test sets ─────────────────────────────
hp_test = json.load(open(
    f"{DATA_DIR}/hotpotqa_test100.json"))[:50]
wh_test = json.load(open(
    f"{DATA_DIR}/wikihow_test100.json"))[:50]

print(f"HotpotQA : {len(hp_test)} samples")
print(f"WikiHow  : {len(wh_test)} samples")

# ── Load bi-encoder ────────────────────────────
print("\nLoading bi-encoder...")
bienc = SentenceTransformer(
    os.path.join(BASE_DIR,
        "models/all-MiniLM-L6-v2"),
    device=DEVICE)
print("Model ready ✅")

# ── Load FAISS indexes ─────────────────────────
def load_idx(source):
    idx  = faiss.read_index(
        f"{BASE_DIR}/corpus/index/faiss_g2/"
        f"{source}_g2.index")
    meta = json.load(open(
        f"{BASE_DIR}/corpus/index/faiss_g2/"
        f"{source}_g2_meta.json"))
    text = {
        c["chunk_id"]: c["text"]
        for c in json.load(open(
            f"{BASE_DIR}/corpus/{source}/"
            f"g2_paragraph/{source}_g2.json"))
    }
    return idx, meta, text

print("Loading indexes...")
hp_idx, hp_meta, hp_text = load_idx("hotpotqa")
wh_idx, wh_meta, wh_text = load_idx("wikihow")
print("Indexes ready ✅")

# ── Shared prompt ──────────────────────────────
# k=5 paragraph chunks
# SAME prompt as Adaptive RAG
PROMPT = ("Context:{context}\n"
          "Q:{query}\n"
          "Answer in one sentence:")

def _gen(prompt, n=30):
    try:
        r = requests.post(OLLAMA_URL, json={
            "model"  : "llama3",
            "prompt" : prompt,
            "stream" : False,
            "options": {
                "temperature": 0.0,
                "num_predict": n,
                "top_k"      : 1
            }
        }, timeout=15)
        return r.json().get(
            "response", "").strip()
    except:
        return ""

# ── Fixed Strong RAG k=5 ──────────────────────
# No adaptation
# No reranking
# Just dense retrieval k=5
# paragraph chunks same prompt
def fixed_rag_ctx(q, idx, meta,
                  text, k=5):
    qv = bienc.encode(
        [q],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")
    _, ii = idx.search(qv, k)
    parts = []
    for i in ii[0]:
        if i == -1: continue
        t = text.get(
            meta[i].get("chunk_id",""), "")
        if t: parts.append(t[:200])
    return " ".join(parts)[:500]

# ── Evaluate ───────────────────────────────────
def evaluate_standard(samples, idx,
                       meta, text, label):
    scores = {
        "f1"     : [],
        "rouge"  : [],
        "overlap": [],
        "halluc" : []
    }
    print(f"\n── Fixed Strong RAG — {label} ──")
    for i, s in enumerate(samples, 1):
        q, gold = s["query"], s["answer"]
        print(f"  [{i}/{len(samples)}]"
              f" {q[:40]}",
              end=" ", flush=True)

        ctx = fixed_rag_ctx(
            q, idx, meta, text, k=5)
        a   = _gen(PROMPT.format(
            context=ctx, query=q))

        scores["f1"].append(
            f1_score(a, gold))
        scores["rouge"].append(
            rouge_l(a, gold))
        scores["overlap"].append(
            evidence_overlap(a, ctx))
        scores["halluc"].append(
            hallucination_rate(
                a, ctx, gold))
        print("✓")

    return scores

# ── Run both datasets ──────────────────────────
H = evaluate_standard(
    hp_test, hp_idx,
    hp_meta, hp_text, "HotpotQA")
W = evaluate_standard(
    wh_test, wh_idx,
    wh_meta, wh_text, "WikiHow")

# ── Print results ──────────────────────────────
rows = [
    ["Fixed RAG k=5",
     "HotpotQA",
     f"{round(avg(H['f1'])*100,1)}"
     f"±{round(std(H['f1'])*100,1)}",
     f"{round(avg(H['rouge'])*100,1)}",
     f"{round(avg(H['overlap'])*100,1)}",
     f"{round(avg(H['halluc'])*100,1)}"],
    ["Fixed RAG k=5",
     "WikiHow",
     f"{round(avg(W['f1'])*100,1)}"
     f"±{round(std(W['f1'])*100,1)}",
     f"{round(avg(W['rouge'])*100,1)}",
     f"{round(avg(W['overlap'])*100,1)}",
     f"{round(avg(W['halluc'])*100,1)}"],
]

print_table(
    "Fixed Strong RAG (k=5) Results",
    ["Model","Dataset","F1±Std",
     "RougeL","Overlap","Halluc"],
    rows,
    [14, 10, 10, 8, 8, 8]
)

# ── Save ───────────────────────────────────────
result = {
    "FixedRAG_hotpotqa": {
        "f1"     : H["f1"],
        "rouge"  : H["rouge"],
        "overlap": H["overlap"],
        "halluc" : H["halluc"],
        "f1_mean": round(avg(H["f1"])*100,1),
        "f1_std" : round(std(H["f1"])*100,1),
    },
    "FixedRAG_wikihow": {
        "f1"     : W["f1"],
        "rouge"  : W["rouge"],
        "overlap": W["overlap"],
        "halluc" : W["halluc"],
        "f1_mean": round(avg(W["f1"])*100,1),
        "f1_std" : round(std(W["f1"])*100,1),
    }
}

json.dump(result,
          open(f"{RESULTS}/"
               f"standard_rag_results.json",
               "w"), indent=2)
print(f"\nSaved → "
      f"{RESULTS}/"
      f"standard_rag_results.json ✅")
