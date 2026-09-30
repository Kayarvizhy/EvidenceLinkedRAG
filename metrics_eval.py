"""
metrics_eval.py  --  answer metrics

em, f1      official HotpotQA / SQuAD definition (lower-case, no punctuation, no articles)
rouge_l     longest-common-subsequence F-measure on the same tokens
overlap     evidence overlap, Eq. (11): share of answer tokens that occur in the retrieved context
unsupported Eq. (12), token-level proxy for hallucination: share of answer tokens (stop words excluded) that occur
            neither in the retrieved context nor in the question.  Unlike the earlier implementation it
            does not use the reference answer and has no floor.
"""
import re, string, collections

ARTICLES = re.compile(r"\b(a|an|the)\b")
PUNCT = set(string.punctuation)
STOP = set("""a an the of in on at and or to for by with from as is was are were be been it this that these those
he she they his her their its i you we not no yes do does did has have had will would can could should may might
which who whom what when where why how than then so if but also there here""".split())


def normalize(s):
    s = "".join(ch for ch in s.lower() if ch not in PUNCT)
    return " ".join(ARTICLES.sub(" ", s).split())


def em(pred, gold):
    return float(normalize(pred) == normalize(gold))


def f1(pred, gold):
    p, g = normalize(pred), normalize(gold)
    if (p in ("yes", "no", "noanswer") or g in ("yes", "no", "noanswer")) and p != g:
        return 0.0
    pt, gt = p.split(), g.split()
    same = sum((collections.Counter(pt) & collections.Counter(gt)).values())
    if same == 0:
        return 0.0
    pr, rc = same / len(pt), same / len(gt)
    return 2 * pr * rc / (pr + rc)


def rouge_l(pred, gold):
    p, g = normalize(pred).split(), normalize(gold).split()
    if not p or not g:
        return 0.0
    prev = [0] * (len(g) + 1)
    for x in p:
        cur = [0]
        for j, y in enumerate(g, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    lcs = prev[-1]
    if lcs == 0:
        return 0.0
    pr, rc = lcs / len(p), lcs / len(g)
    return 2 * pr * rc / (pr + rc)


def overlap(pred, ctx):
    p = normalize(pred).split()
    c = set(normalize(ctx).split())
    return sum(t in c for t in p) / len(p) if p else 0.0


def unsupported(pred, ctx, question):
    p = [t for t in normalize(pred).split() if t not in STOP]
    ok = set(normalize(ctx).split()) | set(normalize(question).split())
    return sum(t not in ok for t in p) / len(p) if p else 0.0


def contains(pred, gold):
    """Reference string occurs in the answer (robust to verbose answers)."""
    return float(normalize(gold) in normalize(pred)) if normalize(gold) else 0.0
