"""
retrieval_lib.py  --  retrievers used by the pilot (no LLM involved)

* Dense     : sentence-transformers/all-MiniLM-L6-v2, mean pooling, cosine  (the paper's encoder)
* BM25      : Okapi BM25 with the formula and defaults of rank_bm25.BM25Okapi (k1=1.5, b=0.75,
              epsilon=0.25), computed with a sparse matrix so that thousands of queries run in seconds.
              check_bm25() verifies the scores against rank_bm25 itself.
* Reranker  : cross-encoder/ms-marco-MiniLM-L-6-v2                           (the paper's reranker)

Embeddings are cached in cache/*.npy.  Models are stored in models/.
"""
import os, re, sys, json, math, collections
import numpy as np
import scipy.sparse as sp
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
MODELS = os.path.join(HERE, "models")
sys.path.insert(0, os.path.join(HERE, "lib"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BI_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CE_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def load_json(name):
    return json.load(open(os.path.join(CACHE, name), encoding="utf8"))


# ── tokenisers for BM25 ──────────────────────────────────────────────────────
def tok_split(text):                       # what run_unified_baselines.py does
    return text.lower().split()


def tok_word(text):                        # strips punctuation: "Wood?" matches "Wood"
    return re.findall(r"[a-z0-9]+", text.lower())


# ── BM25 ─────────────────────────────────────────────────────────────────────
class BM25:
    def __init__(self, docs, tokenizer=tok_word, k1=1.5, b=0.75, epsilon=0.25):
        self.tok = tokenizer
        vocab, rows, cols, vals, dl = {}, [], [], [], []
        for d, text in enumerate(docs):
            toks = tokenizer(text)
            dl.append(len(toks))
            for t, f in collections.Counter(toks).items():
                j = vocab.setdefault(t, len(vocab))
                rows.append(d)
                cols.append(j)
                vals.append(f)
        n_docs, n_terms = len(docs), len(vocab)
        dl = np.asarray(dl, dtype=np.float64)
        avgdl = dl.sum() / n_docs
        rows, cols = np.asarray(rows), np.asarray(cols)
        f = np.asarray(vals, dtype=np.float64)
        df = np.bincount(cols, minlength=n_terms).astype(np.float64)
        idf = np.log(n_docs - df + 0.5) - np.log(df + 0.5)
        idf[idf < 0] = epsilon * (idf.sum() / n_terms)     # rank_bm25: floor for very common terms
        w = idf[cols] * f * (k1 + 1) / (f + k1 * (1 - b + b * dl[rows] / avgdl))
        self.W = sp.csr_matrix((w, (rows, cols)), shape=(n_docs, n_terms)).T.tocsr()   # terms x docs
        self.vocab = vocab
        self.n_docs = n_docs

    def _qmat(self, queries):
        rows, cols, vals = [], [], []
        for i, q in enumerate(queries):
            for t, c in collections.Counter(self.tok(q)).items():
                j = self.vocab.get(t)
                if j is not None:
                    rows.append(i)
                    cols.append(j)
                    vals.append(float(c))
        return sp.csr_matrix((vals, (rows, cols)), shape=(len(queries), len(self.vocab)))

    def scores(self, queries):
        return np.asarray((self._qmat(queries) @ self.W).todense())

    def search(self, queries, k, batch=64):
        ids = np.zeros((len(queries), k), dtype=np.int64)
        sc = np.zeros((len(queries), k), dtype=np.float32)
        for s in range(0, len(queries), batch):
            S = self.scores(queries[s:s + batch])
            top = np.argpartition(-S, k - 1, axis=1)[:, :k]
            part = np.take_along_axis(S, top, axis=1)
            order = np.argsort(-part, axis=1, kind="stable")
            ids[s:s + batch] = np.take_along_axis(top, order, axis=1)
            sc[s:s + batch] = np.take_along_axis(part, order, axis=1)
        return ids, sc


def check_bm25(docs, queries, tokenizer=tok_split):
    """Largest absolute score difference between this BM25 and rank_bm25.BM25Okapi."""
    from rank_bm25 import BM25Okapi
    ref = BM25Okapi([tokenizer(d) for d in docs])
    mine = BM25(docs, tokenizer)
    S = mine.scores(queries)
    return max(float(np.abs(ref.get_scores(tokenizer(q)) - S[i]).max()) for i, q in enumerate(queries))


# ── dense bi-encoder ─────────────────────────────────────────────────────────
class Dense:
    def __init__(self):
        from transformers import AutoTokenizer, AutoModel
        self.tk = AutoTokenizer.from_pretrained(BI_NAME, cache_dir=MODELS)
        self.model = AutoModel.from_pretrained(BI_NAME, cache_dir=MODELS).to(DEVICE).eval()
        if DEVICE == "cuda":
            self.model.half()

    @torch.no_grad()
    def encode(self, texts, batch=256, max_len=256, desc=None):
        order = np.argsort([-len(t) for t in texts])          # long texts first: less padding
        out = np.zeros((len(texts), 384), dtype=np.float32)
        for s in range(0, len(texts), batch):
            idx = order[s:s + batch]
            enc = self.tk([texts[i] for i in idx], padding=True, truncation=True, max_length=max_len,
                          return_tensors="pt").to(DEVICE)
            h = self.model(**enc).last_hidden_state.float()
            m = enc["attention_mask"].unsqueeze(-1).float()
            e = torch.nn.functional.normalize((h * m).sum(1) / m.sum(1).clamp(min=1e-9), dim=1)
            out[idx] = e.cpu().numpy()
            if desc and (s // batch) % 50 == 0:
                print(f"   {desc}: {s}/{len(texts)}", flush=True)
        return out

    def corpus(self, name, texts):
        path = os.path.join(CACHE, f"emb_{name}.npy")
        if os.path.exists(path):
            e = np.load(path)
            if len(e) == len(texts):
                return e.astype(np.float32)
        e = self.encode(texts, desc=f"embedding {name}")
        np.save(path, e.astype(np.float16))
        return e


def dense_search(q_emb, d_emb, k, batch=256):
    D = torch.from_numpy(d_emb).to(DEVICE)
    ids, sc = [], []
    for s in range(0, len(q_emb), batch):
        S = torch.from_numpy(q_emb[s:s + batch]).to(DEVICE) @ D.T
        v, i = S.topk(k, dim=1)
        ids.append(i.cpu().numpy())
        sc.append(v.cpu().numpy())
    return np.vstack(ids), np.vstack(sc)


# ── cross-encoder ────────────────────────────────────────────────────────────
class Cross:
    def __init__(self):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.tk = AutoTokenizer.from_pretrained(CE_NAME, cache_dir=MODELS)
        self.model = AutoModelForSequenceClassification.from_pretrained(CE_NAME, cache_dir=MODELS).to(DEVICE).eval()
        if DEVICE == "cuda":
            self.model.half()
        self.memo = {}

    @torch.no_grad()
    def score(self, pairs, batch=128, max_len=512):
        """pairs: list of (query, passage).  Scores are memoised, so repeated pairs cost nothing."""
        todo = [p for p in dict.fromkeys(pairs) if p not in self.memo]
        order = np.argsort([-len(a) - len(b) for a, b in todo])
        for s in range(0, len(todo), batch):
            chunk = [todo[i] for i in order[s:s + batch]]
            enc = self.tk([a for a, _ in chunk], [b for _, b in chunk], padding=True, truncation=True,
                          max_length=max_len, return_tensors="pt").to(DEVICE)
            z = self.model(**enc).logits.float().squeeze(-1).cpu().numpy()
            for p, v in zip(chunk, z):
                self.memo[p] = float(v)
        return np.asarray([self.memo[p] for p in pairs], dtype=np.float32)


# ── fusion ───────────────────────────────────────────────────────────────────
def rrf(rankings, k=60, weights=None):
    """Reciprocal rank fusion.  rankings: list of ranked id lists.  Returns ids, best first."""
    score = collections.defaultdict(float)
    for r, ranking in enumerate(rankings):
        w = 1.0 if weights is None else weights[r]
        for rank, i in enumerate(ranking):
            score[i] += w / (k + rank + 1)
    return sorted(score, key=lambda i: -score[i])


def unique_in_order(ids):
    seen, out = set(), []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


# ── statistics ───────────────────────────────────────────────────────────────
def paired_bootstrap(a, b, iters=5000, seed=0):
    """Mean of a-b with a 95% confidence interval, questions resampled in pairs."""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), size=(iters, len(d)))].mean(axis=1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))
