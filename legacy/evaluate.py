# evaluate.py
import json, os, re, string
from collections import Counter
from tqdm import tqdm
from uncertainty import run_rag_with_uncertainty

BASE    = os.path.expanduser("~/Desktop/AdaptiveRAG")
RESULTS = os.path.join(BASE, "results")
os.makedirs(RESULTS, exist_ok=True)

# ── Metrics ───────────────────────────────────────────────────────────────────
def normalize(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def exact_match(pred: str, gold: str) -> float:
    return float(normalize(pred) == normalize(gold))

def f1_score(pred: str, gold: str) -> float:
    pred_tokens = normalize(pred).split()
    gold_tokens = normalize(gold).split()
    common      = Counter(pred_tokens) & Counter(gold_tokens)
    num_same    = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall    = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)

def rouge_l(pred: str, gold: str) -> float:
    """Simplified ROUGE-L using LCS"""
    pred_tokens = normalize(pred).split()
    gold_tokens = normalize(gold).split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    m, n = len(gold_tokens), len(pred_tokens)
    dp   = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if gold_tokens[i-1] == pred_tokens[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    lcs       = dp[m][n]
    precision = lcs / n if n > 0 else 0
    recall    = lcs / m if m > 0 else 0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

# ── Load Test Samples ─────────────────────────────────────────────────────────
def load_hotpotqa_samples(n: int = 50) -> list:
    path = os.path.join(BASE, "corpus/hotpotqa/raw/hotpot_dev_distractor_v1.json")
    data = json.load(open(path))
    samples = []
    for item in data[:n]:
        samples.append({
            "query"  : item["question"],
            "answer" : item["answer"],
            "type"   : item["type"],
            "level"  : item["level"],
            "source" : "hotpotqa"
        })
    return samples

# ── Run Evaluation ────────────────────────────────────────────────────────────
def evaluate(n_samples: int = 50):
    print(f"\n{'='*65}")
    print(f"{'ADAPTIVE RAG — EVALUATION':^65}")
    print(f"{'='*65}")
    print(f"[INFO] Evaluating on {n_samples} HotpotQA samples...")

    samples = load_hotpotqa_samples(n_samples)

    all_results = []
    em_scores, f1_scores, rouge_scores = [], [], []

    # per intent tracking
    intent_scores = {
        "factual"    : {"em": [], "f1": [], "rouge": []},
        "analytical" : {"em": [], "f1": [], "rouge": []},
        "procedural" : {"em": [], "f1": [], "rouge": []},
    }

    # re-retrieval tracking
    reruns_triggered = 0

    for sample in tqdm(samples, desc="Evaluating"):
        try:
            result = run_rag_with_uncertainty(sample["query"], verbose=False)

            pred   = result["answer"]
            gold   = sample["answer"]
            intent = result["intent"]

            em    = exact_match(pred, gold)
            f1    = f1_score(pred, gold)
            rouge = rouge_l(pred, gold)

            em_scores.append(em)
            f1_scores.append(f1)
            rouge_scores.append(rouge)

            intent_scores[intent]["em"].append(em)
            intent_scores[intent]["f1"].append(f1)
            intent_scores[intent]["rouge"].append(rouge)

            if result["re_retrieval_cycles"] > 0:
                reruns_triggered += 1

            all_results.append({
                "query"              : sample["query"],
                "gold_answer"        : gold,
                "pred_answer"        : pred,
                "intent"             : intent,
                "effort"             : result["effort"],
                "granularity"        : result["granularity"],
                "k"                  : result["k"],
                "model"              : result["model"],
                "re_retrieval_cycles": result["re_retrieval_cycles"],
                "em"                 : em,
                "f1"                 : f1,
                "rouge_l"            : rouge,
            })

        except Exception as e:
            print(f"[ERROR] {sample['query'][:50]} → {e}")
            continue

    # ── Print Results ─────────────────────────────────────────────────────────
    n = len(em_scores)
    print(f"\n{'='*65}")
    print(f"{'OVERALL RESULTS':^65}")
    print(f"{'='*65}")
    print(f"   Samples evaluated    : {n}")
    print(f"   EM  (Exact Match)    : {sum(em_scores)/n*100:.2f}%")
    print(f"   F1  Score            : {sum(f1_scores)/n*100:.2f}%")
    print(f"   ROUGE-L              : {sum(rouge_scores)/n*100:.2f}%")
    print(f"   Re-retrieval triggered: {reruns_triggered}/{n} ({reruns_triggered/n*100:.1f}%)")

    print(f"\n{'='*65}")
    print(f"{'PER INTENT RESULTS':^65}")
    print(f"{'='*65}")
    for intent, scores in intent_scores.items():
        if scores["f1"]:
            n_i = len(scores["f1"])
            print(f"\n   Intent : {intent.upper()} ({n_i} samples)")
            print(f"   EM     : {sum(scores['em'])/n_i*100:.2f}%")
            print(f"   F1     : {sum(scores['f1'])/n_i*100:.2f}%")
            print(f"   ROUGE-L: {sum(scores['rouge'])/n_i*100:.2f}%")

    # ── Save Results ──────────────────────────────────────────────────────────
    out_path = os.path.join(RESULTS, "evaluation_results.json")
    json.dump(all_results, open(out_path, "w"), indent=2)
    print(f"\n✅ Results saved → {out_path}")
    print(f"{'='*65}")

    return all_results


if __name__ == "__main__":
    evaluate(n_samples=50)
