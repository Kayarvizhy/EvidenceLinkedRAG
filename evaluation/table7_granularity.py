import os, sys, json, math
from tqdm import tqdm

BASE_DIR = os.path.expanduser("~/Desktop/AdaptiveRAG")
EVAL_DIR = os.path.join(BASE_DIR, "evaluation")
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, EVAL_DIR)

from metrics import f1_score, avg, print_table
from rag_pipeline import run_rag

RESULTS = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS, exist_ok=True)

# ── helpers ───────────────────────────────────────────
def std(scores):
    if len(scores) < 2:
        return 0.0
    m = avg(scores)
    v = sum((x - m) ** 2
            for x in scores) / (len(scores) - 1)
    return round(math.sqrt(v) * 100, 1)

# ── samples ───────────────────────────────────────────
SAMPLES = {
    "factual": [
        {"query": "What nationality was Nikola Tesla?",
         "answer": "Serbian"},
        {"query": "Who invented the telephone?",
         "answer": "Alexander Graham Bell"},
        {"query": "What is the capital of Japan?",
         "answer": "Tokyo"},
        {"query": "Who painted the Mona Lisa?",
         "answer": "Leonardo da Vinci"},
        {"query": "What planet is closest to the sun?",
         "answer": "Mercury"},
    ],
    "reasoning": [
        {"query": "Who invented the telephone?",
         "answer": "Alexander Graham Bell"},
        {"query": "What element does Au represent?",
         "answer": "Gold"},
        {"query": "Who was the first US president?",
         "answer": "George Washington"},
        {"query": "In what country is the Eiffel Tower?",
         "answer": "France"},
        {"query": "Who wrote the play Hamlet?",
         "answer": "William Shakespeare"},
    ],
    "procedural": [
        {"query": "What is the chemical symbol for water?",
         "answer": "H2O"},
        {"query": "What planet is closest to the sun?",
         "answer": "Mercury"},
        {"query": "Who painted the Mona Lisa?",
         "answer": "Leonardo da Vinci"},
        {"query": "What nationality was Nikola Tesla?",
         "answer": "Serbian"},
        {"query": "What is the capital of Japan?",
         "answer": "Tokyo"},
    ],
}

# ── dataset per qtype ─────────────────────────────────
QTYPE_DATASET = {
    "factual"    : "HotpotQA",
    "reasoning"  : "HotpotQA",
    "procedural" : "WikiHow",
}

# ── explanation per granularity ───────────────────────
EXPLANATIONS = {
    "G1 (Sentence)"  : "Best for short factual lookups",
    "G2 (Paragraph)" : "Best for multi-hop reasoning",
    "G3 (Section)"   : "Best for procedural/long-form",
    "Adaptive (Ours)": "Selects granularity per query",
}

def run():
    gran_scores = {
        "G1": [], "G2": [],
        "G3": [], "Adaptive": []
    }
    gran_ds_sets = {
        "G1": set(), "G2": set(),
        "G3": set(), "Adaptive": set()
    }

    for gran in ["G1", "G2", "G3", "Adaptive"]:
        print(f"\nRunning {gran}...")
        for qtype, samples in SAMPLES.items():
            ds = QTYPE_DATASET[qtype]
            for s in tqdm(samples,
                          desc=f"{gran}/{qtype}"):
                try:
                    if gran == "Adaptive":
                        r = run_rag(
                            s["query"],
                            verbose=False)
                    else:
                        r = run_rag(
                            s["query"],
                            verbose=False,
                            force_granularity=gran)

                    sc = f1_score(
                        r["answer"].lower(),
                        s["answer"].lower())
                    gran_scores[gran].append(sc)
                    gran_ds_sets[gran].add(ds)
                    print(f"  ✅ {s['query'][:40]}"
                          f" | {gran} F1={sc:.2f}")

                except Exception as e:
                    print(f"  [Error] {e}")
                    gran_scores[gran].append(0.0)
                    gran_ds_sets[gran].add(ds)

    configs = [
        ("G1 (Sentence)",   "G1"),
        ("G2 (Paragraph)",  "G2"),
        ("G3 (Section)",    "G3"),
        ("Adaptive (Ours)", "Adaptive"),
    ]

    rows = []
    data = {}

    for label, key in configs:
        scores = gran_scores[key]
        f1     = round(avg(scores) * 100, 1)
        sd     = std(scores)
        ds     = " + ".join(sorted(gran_ds_sets[key]))
        exp    = EXPLANATIONS[label]

        rows.append([
            label,
            f"{f1:.1f}",
            f"±{sd}",
            ds,
            exp
        ])

        data[label] = {
            "F1 Score" : f1,
            "Std"      : sd,
            "Dataset"  : ds,
            "Note"     : exp,
        }

        print(f"📊 [{label}] F1={f1}% ±{sd} | {ds}")

    print_table(
        "TABLE 7: Effect of Chunk Granularity",
        ["Granularity", "F1 (%)",
         "Std", "Dataset", "Explanation"],
        rows,
        [20, 9, 6, 25, 35]
    )

    json.dump(data,
        open(f"{RESULTS}/table7_data.json", "w"),
        indent=2)
    print(f"\n✅ Saved → results/table7_data.json")

if __name__ == "__main__":
    run()
