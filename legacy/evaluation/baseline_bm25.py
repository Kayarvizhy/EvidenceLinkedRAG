import os, sys, json, requests
from rank_bm25 import BM25Okapi

BASE_DIR   = os.path.expanduser("~/Desktop/AdaptiveRAG")
DATA_DIR   = os.path.join(BASE_DIR, "evaluation/test_sets")
RESULTS    = os.path.join(BASE_DIR, "results")
OLLAMA_URL = "http://localhost:11434/api/generate"

sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "evaluation"))
from metrics import (f1_score, rouge_l,
                     hallucination_rate,
                     evidence_overlap,
                     avg, std, print_table)

os.makedirs(RESULTS, exist_ok=True)

# ── Load test sets ─────────────────────────────
hp_test = json.load(open(
    f"{DATA_DIR}/hotpotqa_test100.json"))[:50]
wh_test = json.load(open(
    f"{DATA_DIR}/wikihow_test100.json"))[:50]

print(f"HotpotQA samples : {len(hp_test)}")
print(f"WikiHow  samples : {len(wh_test)}")

# ── Load SMALL corpus slice for BM25 ──────────
# Only first 2000 docs — fast enough ✅
def load_corpus(source, limit=2000):
    path = (f"{BASE_DIR}/corpus/{source}/"
            f"g2_paragraph/{source}_g2.json")
    docs = json.load(open(path))
    return [d["text"]
            for d in docs[:limit]
            if d.get("text","").strip()]

print("Loading corpora (2000 docs each)...")
hp_corpus = load_corpus("hotpotqa", 2000)
wh_corpus = load_corpus("wikihow",  2000)
print(f"HotpotQA corpus : {len(hp_corpus)} docs")
print(f"WikiHow  corpus : {len(wh_corpus)} docs")

# ── Build BM25 indexes ─────────────────────────
print("Building BM25 indexes...")
hp_bm25 = BM25Okapi(
    [d.lower().split() for d in hp_corpus])
wh_bm25 = BM25Okapi(
    [d.lower().split() for d in wh_corpus])
print("BM25 ready ✅")

# ── Shared prompt ──────────────────────────────
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
        return r.json().get("response",
                            "").strip()
    except:
        return ""

# ── Evaluate ───────────────────────────────────
def evaluate_bm25(samples, bm25,
                  corpus, label):
    scores = {
        "f1"     : [],
        "rouge"  : [],
        "overlap": [],
        "halluc" : []
    }
    print(f"\n── BM25 + LLM — {label} ──")
    for i, s in enumerate(samples, 1):
        q, gold = s["query"], s["answer"]
        print(f"  [{i}/{len(samples)}]"
              f" {q[:40]}",
              end=" ", flush=True)

        tokens = q.lower().split()
        top5   = bm25.get_top_n(
            tokens, corpus, n=5)
        ctx    = " ".join(top5)[:400]

        a = _gen(PROMPT.format(
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

H = evaluate_bm25(hp_test, hp_bm25,
                  hp_corpus, "HotpotQA")
W = evaluate_bm25(wh_test, wh_bm25,
                  wh_corpus, "WikiHow")

# ── Print results ──────────────────────────────
rows = [
    ["BM25 + LLM",
     "HotpotQA",
     f"{round(avg(H['f1'])*100,1)}"
     f"±{round(std(H['f1'])*100,1)}",
     f"{round(avg(H['rouge'])*100,1)}",
     f"{round(avg(H['overlap'])*100,1)}",
     f"{round(avg(H['halluc'])*100,1)}"],
    ["BM25 + LLM",
     "WikiHow",
     f"{round(avg(W['f1'])*100,1)}"
     f"±{round(std(W['f1'])*100,1)}",
     f"{round(avg(W['rouge'])*100,1)}",
     f"{round(avg(W['overlap'])*100,1)}",
     f"{round(avg(W['halluc'])*100,1)}"],
]

print_table(
    "BM25 + LLM Baseline Results",
    ["Model","Dataset","F1±Std",
     "RougeL","Overlap","Halluc"],
    rows,
    [14, 10, 10, 8, 8, 8]
)

# ── Save ───────────────────────────────────────
result = {
    "BM25_hotpotqa": {
        "f1"     : H["f1"],
        "rouge"  : H["rouge"],
        "overlap": H["overlap"],
        "halluc" : H["halluc"],
        "f1_mean": round(avg(H["f1"])*100,1),
        "f1_std" : round(std(H["f1"])*100,1),
    },
    "BM25_wikihow": {
        "f1"     : W["f1"],
        "rouge"  : W["rouge"],
        "overlap": W["overlap"],
        "halluc" : W["halluc"],
        "f1_mean": round(avg(W["f1"])*100,1),
        "f1_std" : round(std(W["f1"])*100,1),
    }
}

json.dump(result,
          open(f"{RESULTS}/bm25_results.json",
               "w"), indent=2)
print(f"\nSaved → {RESULTS}/bm25_results.json ✅")
