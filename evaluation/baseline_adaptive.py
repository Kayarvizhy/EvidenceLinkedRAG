import os, sys, json
import numpy as np

BASE_DIR = os.path.expanduser(
    "~/Desktop/AdaptiveRAG")
DATA_DIR = os.path.join(BASE_DIR,
    "evaluation/test_sets")
RESULTS  = os.path.join(BASE_DIR, "results")

sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(
    BASE_DIR, "evaluation"))

from rag_pipeline import run_rag
from metrics import (f1_score, rouge_l,
                     hallucination_rate,
                     evidence_overlap,
                     avg, std, print_table)

os.makedirs(RESULTS, exist_ok=True)

# ── WikiHow ONLY ───────────────────────────────
wh_test = json.load(open(
    f"{DATA_DIR}/wikihow_test100.json"))[:20]

print(f"WikiHow : {len(wh_test)} samples")

# ── Evaluate ───────────────────────────────────
def evaluate_adaptive(samples, label):
    scores = {
        "f1"     : [],
        "rouge"  : [],
        "overlap": [],
        "halluc" : [],
    }
    print(f"\n── Adaptive RAG — {label} ──")
    for i, s in enumerate(samples, 1):
        q, gold = s["query"], s["answer"]
        print(f"  [{i}/{len(samples)}]"
              f" {q[:50]}",
              end=" ", flush=True)

        try:
            result = run_rag(q, verbose=False)
            a   = result.get("answer",  "")
            ctx = result.get("context", "")
        except Exception as e:
            print(f"[ERR: {e}]", end=" ")
            a, ctx = "", ""

        scores["f1"].append(
            f1_score(a, gold))
        scores["rouge"].append(
            rouge_l(a, gold))
        scores["overlap"].append(
            evidence_overlap(a, ctx))
        scores["halluc"].append(
            hallucination_rate(a, ctx, gold))
        print("✓")

    return scores

# ── Run WikiHow only ───────────────────────────
W = evaluate_adaptive(wh_test, "WikiHow")

# ── Load previous HotpotQA results ────────────
prev = json.load(open(
    f"{RESULTS}/adaptive_results.json"))
H_f1    = prev["Adaptive_hotpotqa"]["f1"]
H_rouge = prev["Adaptive_hotpotqa"]["rouge"]
H_over  = prev["Adaptive_hotpotqa"]["overlap"]
H_hal   = prev["Adaptive_hotpotqa"]["halluc"]

# ── Print results ──────────────────────────────
rows = [
    ["Adaptive RAG",
     "HotpotQA",
     f"{round(avg(H_f1)*100,1)}"
     f"±{round(std(H_f1)*100,1)}",
     f"{round(avg(H_rouge)*100,1)}",
     f"{round(avg(H_over)*100,1)}",
     f"{round(avg(H_hal)*100,1)}"],
    ["Adaptive RAG",
     "WikiHow",
     f"{round(avg(W['f1'])*100,1)}"
     f"±{round(std(W['f1'])*100,1)}",
     f"{round(avg(W['rouge'])*100,1)}",
     f"{round(avg(W['overlap'])*100,1)}",
     f"{round(avg(W['halluc'])*100,1)}"],
]

print_table(
    "Adaptive RAG (Ours) Results",
    ["Model","Dataset","F1±Std",
     "RougeL","Overlap","Halluc"],
    rows,
    [14, 10, 10, 8, 8, 8]
)

# ── Save ───────────────────────────────────────
result = {
    "Adaptive_hotpotqa": {
        "f1"     : H_f1,
        "rouge"  : H_rouge,
        "overlap": H_over,
        "halluc" : H_hal,
        "f1_mean": round(avg(H_f1)*100,1),
        "f1_std" : round(std(H_f1)*100,1),
    },
    "Adaptive_wikihow": {
        "f1"     : W["f1"],
        "rouge"  : W["rouge"],
        "overlap": W["overlap"],
        "halluc" : W["halluc"],
        "f1_mean": round(avg(W["f1"])*100,1),
        "f1_std" : round(std(W["f1"])*100,1),
    }
}

json.dump(result,
    open(f"{RESULTS}/adaptive_results.json",
         "w"), indent=2)
print(f"\nSaved → {RESULTS}/"
      f"adaptive_results.json ✅")
