import os, sys, json
import numpy as np

BASE_DIR = os.path.expanduser(
    "~/Desktop/AdaptiveRAG")
RESULTS  = os.path.join(BASE_DIR, "results")

sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(
    BASE_DIR, "evaluation"))

from metrics import (avg, std, print_table)

# ── Load saved per-query results ──────────────
full    = json.load(open(
    f"{RESULTS}/adaptive_results.json"))
no_int  = json.load(open(
    f"{RESULTS}/ablation_no_intent.json"))
no_eff  = json.load(open(
    f"{RESULTS}/ablation_no_effort.json"))
no_rr   = json.load(open(
    f"{RESULTS}/ablation_no_reretrieval.json"))
no_unc  = json.load(open(
    f"{RESULTS}/ablation_no_uncertainty.json"))
no_gran = json.load(open(
    f"{RESULTS}/ablation_no_granularity.json"))
fixed_k = json.load(open(
    f"{RESULTS}/ablation_fixed_k.json"))

# ── Extract F1 scores (first 50 samples) ──────
def get_f1(data, key):
    return data[key]["f1"][:50]

configs = {
    "Full Model (Ours)": get_f1(
        full,    "Adaptive_hotpotqa"),
    "w/o Intent Detection": get_f1(
        no_int,  "Ablation_hotpotqa"),
    "w/o Effort Estimation": get_f1(
        no_eff,  "Ablation_hotpotqa"),
    "w/o Re-Retrieval": get_f1(
        no_rr,   "Ablation_hotpotqa"),
    "w/o Uncertainty Estimation": get_f1(
        no_unc,  "Ablation_hotpotqa"),
    "w/o Adaptive Granularity": get_f1(
        no_gran, "Ablation_hotpotqa"),
    "Fixed k (k=3, no adaptation)": get_f1(
        fixed_k, "Ablation_hotpotqa"),
}

# ── Compute metrics ────────────────────────────
full_f1 = round(
    avg(configs["Full Model (Ours)"]) * 100, 1)

rows   = []
result = {}

for config_name, f1_list in configs.items():
    mean  = round(avg(f1_list) * 100, 1)
    stdev = round(std(f1_list) * 100, 1)
    delta = round(mean - full_f1, 1)
    sign  = "+" if delta > 0 else ""

    rows.append([
        config_name,
        f"{mean}",
        f"{stdev}",
        f"{sign}{delta}" if delta != 0 else "-"
    ])

    result[config_name] = {
        "F1_Score": mean,
        "Std"     : stdev,
        "delta"   : delta,
        "n"       : len(f1_list)
    }

# ── Print Table ────────────────────────────────
print_table(
    "TABLE 6: Component-wise Contribution",
    ["Configuration", "F1 (%)", "Std", "Δ vs Full"],
    rows,
    [30, 9, 7, 10]
)

# ── Save JSON ──────────────────────────────────
json.dump(result,
    open(f"{RESULTS}/table6_data.json",
    "w"), indent=2)
print(f"\nSaved → "
      f"{RESULTS}/table6_data.json ✅")
