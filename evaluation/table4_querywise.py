import os, sys, json
from collections import Counter

BASE_DIR = os.path.expanduser(
    "~/Desktop/AdaptiveRAG")
DATA_DIR = os.path.join(BASE_DIR,
    "evaluation/test_sets")
RESULTS  = os.path.join(BASE_DIR, "results")

sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(
    BASE_DIR, "evaluation"))

from metrics import (avg, std, print_table)

# ── Load saved per-query results ──────────────
ada = json.load(open(
    f"{RESULTS}/adaptive_results.json"))
rag = json.load(open(
    f"{RESULTS}/standard_rag_results.json"))

ada_hp_f1 = ada["Adaptive_hotpotqa"]["f1"]
rag_hp_f1 = rag["FixedRAG_hotpotqa"]["f1"]
ada_wh_f1 = ada["Adaptive_wikihow"]["f1"]
rag_wh_f1 = rag["FixedRAG_wikihow"]["f1"]

# ── Load test sets ─────────────────────────────
hp_test = json.load(open(
    f"{DATA_DIR}/hotpotqa_test100.json"))[:50]
wh_test = json.load(open(
    f"{DATA_DIR}/wikihow_test100.json"))[:50]  # ← FIXED

# ── Smarter label function ─────────────────────
def label_hotpotqa(q):
    q = q.lower()
    # Strong single-fact signals → Factual
    factual = [
        "what is the capital",
        "who invented",
        "when was",
        "where is",
        "what is the name",
        "how tall",
        "how old"
    ]
    if any(f in q for f in factual):
        return "Factual"
    return "Multi-hop"  # default for HotpotQA

def label_wikihow(q):
    return "Procedural"

# ── Assign labels ──────────────────────────────
hp_labels = [label_hotpotqa(s["query"])
             for s in hp_test]
wh_labels = [label_wikihow(s["query"])
             for s in wh_test]

print("HotpotQA label dist:",
      Counter(hp_labels))
print("WikiHow  label dist:",
      Counter(wh_labels))

# ── Group F1 by type ───────────────────────────
def group_by_type(labels, f1_list):
    groups = {}
    for lbl, f1 in zip(labels, f1_list):
        groups.setdefault(lbl, []).append(f1)
    return groups

ada_hp_grp = group_by_type(
    hp_labels, ada_hp_f1)
rag_hp_grp = group_by_type(
    hp_labels, rag_hp_f1)
ada_wh_grp = group_by_type(
    wh_labels, ada_wh_f1)
rag_wh_grp = group_by_type(
    wh_labels, rag_wh_f1)

# ── Build rows ─────────────────────────────────
rows = []
all_types = ["Factual", "Multi-hop",
             "Procedural"]

type_map = {
    "Factual"   : (rag_hp_grp, ada_hp_grp),
    "Multi-hop" : (rag_hp_grp, ada_hp_grp),
    "Procedural": (rag_wh_grp, ada_wh_grp),
}

result = {}
for qtype in all_types:
    rag_grp, ada_grp = type_map[qtype]
    r_f1 = rag_grp.get(qtype, [])
    a_f1 = ada_grp.get(qtype, [])
    r_mean = round(avg(r_f1)*100, 1) \
        if r_f1 else 0.0
    a_mean = round(avg(a_f1)*100, 1) \
        if a_f1 else 0.0
    delta  = round(a_mean - r_mean, 1)
    sign   = "+" if delta >= 0 else ""
    rows.append([
        qtype,
        "Standard RAG",
        f"{r_mean}",
        str(len(r_f1)),
        ""])
    rows.append([
        qtype,
        "Adaptive RAG (Ours)",
        f"{a_mean}",
        str(len(a_f1)),
        f"{sign}{delta}"])
    result[qtype] = {
        "Standard_RAG" : r_mean,
        "Adaptive_RAG" : a_mean,
        "delta"        : delta,
        "n"            : len(a_f1),
    }

print_table(
    "TABLE 4: Query-Type Wise Performance",
    ["Query Type", "Model",
     "F1 (%)", "n", "Δ"],
    rows,
    [12, 22, 8, 5, 7]
)

# ── Save ───────────────────────────────────────
json.dump(result,
    open(f"{RESULTS}/table4_data.json",
    "w"), indent=2)
print(f"\nSaved → "
      f"{RESULTS}/table4_data.json ✅")
