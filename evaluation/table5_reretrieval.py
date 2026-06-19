import os, sys, json, math

BASE_DIR = os.path.expanduser(
    "~/Desktop/AdaptiveRAG")
RESULTS  = os.path.join(BASE_DIR, "results")

sys.path.insert(0, os.path.join(
    BASE_DIR, "evaluation"))

from metrics import print_table

# ── load already computed scores ─────────────
data = json.load(open(
    f"{RESULTS}/table5_data.json"))

# ── helpers ───────────────────────────────────
def std(scores):
    if len(scores) < 2:
        return 0.0
    m = sum(scores) / len(scores)
    v = sum((x-m)**2
            for x in scores) / (len(scores)-1)
    return round(math.sqrt(v)*100, 1)

# ── build table ───────────────────────────────
configs = [
    "Without Re-Retrieval",
    "With Re-Retrieval (1 cycle)",
    "With Re-Retrieval (2 cycles)",
]

rows     = []
out_data = {}

for label in configs:
    scores = data[label]["scores"]
    f1     = data[label]["F1"]
    sd     = std(scores)
    n      = len(scores)

    rows.append([
        label,
        f"{f1}",
        f"±{sd}",
        "HotpotQA",
        str(n),
    ])

    out_data[label] = {
        "F1"     : f1,
        "Std"    : sd,
        "Dataset": "HotpotQA",
        "n"      : n,
        "scores" : scores,
    }

    print(f"📊 [{label}]"
          f" F1={f1}% ±{sd}"
          f" | n={n}")

print_table(
    "TABLE 5: Effect of Re-Retrieval (REAL)",
    ["Setting", "F1 (%)",
     "Std", "Dataset", "n"],
    rows,
    [35, 8, 7, 12, 5]
)

json.dump(out_data,
    open(f"{RESULTS}/table5_data.json","w"),
    indent=2)
print(f"\n✅ Saved → results/table5_data.json")
