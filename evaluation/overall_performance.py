import os, sys, json, math

BASE_DIR = os.path.expanduser(
    "~/Desktop/AdaptiveRAG")
RESULTS  = os.path.join(BASE_DIR, "results")

sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(
    BASE_DIR, "evaluation"))

from metrics import avg, print_table

# ── Load all existing result files ────────────
bm25    = json.load(open(
    f"{RESULTS}/bm25_results.json"))
fixed   = json.load(open(
    f"{RESULTS}/standard_rag_results.json"))
rerank  = json.load(open(
    f"{RESULTS}/reranker_results.json"))
ada     = json.load(open(
    f"{RESULTS}/adaptive_results.json"))

# ── Helper functions ───────────────────────────
def std(scores):
    if len(scores) < 2:
        return 0.0
    m = avg(scores)
    v = sum((x - m) ** 2
        for x in scores) / (len(scores) - 1)
    return round(math.sqrt(v) * 100, 1)

def mean_pct(scores):
    return round(avg(scores) * 100, 1)

def halluc_pct(scores):
    return round(avg(scores) * 100, 1)

# ── Extract scores from each file ─────────────
configs = {
    "BM25 + LLM": {
        "hp_f1"   : bm25["BM25_hotpotqa"]["f1"],
        "hp_hal"  : bm25["BM25_hotpotqa"]["halluc"],
        "wh_f1"   : bm25["BM25_wikihow"]["f1"],
        "wh_hal"  : bm25["BM25_wikihow"]["halluc"],
    },
    "Standard RAG": {
        "hp_f1"   : fixed["FixedRAG_hotpotqa"]["f1"],
        "hp_hal"  : fixed["FixedRAG_hotpotqa"]["halluc"],
        "wh_f1"   : fixed["FixedRAG_wikihow"]["f1"],
        "wh_hal"  : fixed["FixedRAG_wikihow"]["halluc"],
    },
    "RAG + Reranker": {
        "hp_f1"   : rerank["Reranker_hotpotqa"]["f1"],
        "hp_hal"  : rerank["Reranker_hotpotqa"]["halluc"],
        "wh_f1"   : rerank["Reranker_wikihow"]["f1"],
        "wh_hal"  : rerank["Reranker_wikihow"]["halluc"],
    },
    "Adaptive RAG (Ours)": {
        "hp_f1"   : ada["Adaptive_hotpotqa"]["f1"],
        "hp_hal"  : ada["Adaptive_hotpotqa"]["halluc"],
        "wh_f1"   : ada["Adaptive_wikihow"]["f1"],
        "wh_hal"  : ada["Adaptive_wikihow"]["halluc"],
    },
}

# ── Build table rows and result dict ──────────
rows   = []
result = {}

for model, d in configs.items():
    hp_f1  = mean_pct(d["hp_f1"])
    hp_sd  = std(d["hp_f1"])
    hp_hal = halluc_pct(d["hp_hal"])
    wh_f1  = mean_pct(d["wh_f1"])
    wh_sd  = std(d["wh_f1"])
    wh_hal = halluc_pct(d["wh_hal"])

    rows.append([
        model,
        f"{hp_f1} ± {hp_sd}",
        f"{wh_f1} ± {wh_sd}",
        f"{hp_hal}",
        f"{wh_hal}",
    ])

    result[model] = {
        "HotpotQA": {
            "f1_mean"   : hp_f1,
            "f1_std"    : hp_sd,
            "halluc_mean": hp_hal,
            "n"         : len(d["hp_f1"]),
        },
        "WikiHow": {
            "f1_mean"   : wh_f1,
            "f1_std"    : wh_sd,
            "halluc_mean": wh_hal,
            "n"         : len(d["wh_f1"]),
        },
    }

# ── Print Table ────────────────────────────────
print_table(
    "OVERALL PERFORMANCE TABLE",
    ["Model", "HotpotQA F1(%)",
     "WikiHow F1(%)",
     "HotpotQA Halluc%",
     "WikiHow Halluc%"],
    rows,
    [22, 16, 16, 18, 16]
)

print(f"\nSample sizes:")
print(f"  HotpotQA → n=50")
print(f"  WikiHow  → n=20")

# ── Save JSON ──────────────────────────────────
json.dump(result,
    open(f"{RESULTS}/overall_performance.json",
    "w"), indent=2)
print(f"\n✅ Saved → "
      f"results/overall_performance.json")
