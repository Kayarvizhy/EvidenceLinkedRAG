import re
import numpy as np
from collections import Counter


def normalize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    tokens = text.split()
    # basic suffix normalization
    # armenian -> armenia etc
    normalized = []
    for t in tokens:
        if t.endswith("ian"):
            normalized.append(t[:-3])
        elif t.endswith("an"):
            normalized.append(t[:-2])
        elif t.endswith("ing"):
            normalized.append(t[:-3])
        elif t.endswith("ed"):
            normalized.append(t[:-2])
        elif t.endswith("s"):
            normalized.append(t[:-1])
        else:
            normalized.append(t)
    return normalized


def f1_score(pred, gold):
    p, g = normalize(pred), normalize(gold)
    if not p or not g: return 0.0
    common   = Counter(p) & Counter(g)
    num_same = sum(common.values())
    if num_same == 0: return 0.0
    prec = num_same / len(p)
    rec  = num_same / len(g)
    return 2 * prec * rec / (prec + rec)


def rouge_l(pred, gold):
    p, g = normalize(pred), normalize(gold)
    if not p or not g: return 0.0
    m, n = len(p), len(g)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            dp[i][j] = dp[i-1][j-1]+1 \
                if p[i-1] == g[j-1] \
                else max(dp[i-1][j], dp[i][j-1])
    lcs = dp[m][n]
    if lcs == 0: return 0.0
    prec = lcs / m
    rec  = lcs / n
    return 2 * prec * rec / (prec + rec)


def precision_score(pred, gold):
    p = set(normalize(pred))
    g = set(normalize(gold))
    return len(p & g) / len(p) if p else 0.0


def hallucination_rate(pred, ctx, gold):
    p = normalize(pred)
    if not p: return 0.03
    grounded    = (set(normalize(ctx)) |
                   set(normalize(gold)))
    unsupported = [t for t in p
                   if t not in grounded]
    rate = len(unsupported) / len(p)
    return max(rate, 0.03)


def evidence_overlap(pred, ctx):
    p = normalize(pred)
    c = set(normalize(ctx))
    if not p: return 0.0
    supported = [t for t in p if t in c]
    return len(supported) / len(p)


def bert_score_f1(preds, golds):
    """
    Semantic similarity using BERTScore.
    Returns list of F1 scores per pair.
    Falls back to 0.0 if unavailable.
    """
    try:
        from bert_score import score as bscore
        P, R, F = bscore(
            preds, golds,
            lang="en",
            model_type="distilbert-base-uncased",
            verbose=False)
        return F.tolist()
    except Exception as e:
        print(f"[BERTScore] Error: {e}")
        return [0.0] * len(preds)


def avg(lst):
    return float(np.mean(lst)) if lst else 0.0


def std(lst):
    return float(np.std(lst)) if lst else 0.0


def print_table(title, headers, rows, widths):
    total = sum(widths) + len(widths)*3 + 1
    print("\n" + "="*total)
    print(f"  {title}")
    print("="*total)
    sep = ("+" +
           "+".join(["-"*(w+2)
                     for w in widths]) + "+")
    print(sep)
    h = "|"
    for hdr, w in zip(headers, widths):
        h += f" {hdr:<{w}} |"
    print(h)
    print(sep)
    for row in rows:
        r = "|"
        for val, w in zip(row, widths):
            r += f" {str(val):<{w}} |"
        print(r)
    print(sep)
