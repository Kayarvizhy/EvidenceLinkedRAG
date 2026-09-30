"""
systems.py  --  the retrieval systems compared in the generation experiment

Every system returns, for each question, the list of passages that go into the prompt.
All settings of the proposed method are the frozen values of retrieval_pilot.py.

  bm25      BM25 over title + text, word tokens, k = 5
  fixed     Fixed RAG: dense over text, k = 5                                   (as in the paper)
  rerank    RAG + Reranker: dense over text, top 20 -> cross-encoder, k = 5     (as in the paper)
  strong    dense over title + text, top 50 -> cross-encoder, k = 5             (strongest baseline)
  hop1      proposed without second hop
  full      proposed: fusion -> cross-encoder -> title-linked second hop (factual and analytical
            questions); procedural questions use hop1, because a how-to article answers them alone
  uamg_v1   Heuristic Adaptive RAG; it retrieves inside the generation loop, see UamgV1 below
"""
import os, re, json, time, collections
import numpy as np
import retrieval_lib as R
from repo_query_understanding import understand_query

CONFIG = dict(k=5, first_stage=100, rrf_k=60, pool=50, bridges=3, reserved_slots=1, rerank_pool_baseline=20)
WIDE = dict(k=8, bridges=5, reserved_slots=3)
SPLITS = {"hotpot": {"tune": (0, 500), "test": (500, 1500)},
          "wikihow": {"tune": (0, 100), "test": (100, 1100)},
          "wiki2": {"test": (0, 1000)},             # 2WikiMultihopQA: nothing is tuned on it
          "musique": {"test": (0, 1000)}}           # MuSiQue: nothing is tuned on it
# LLM-driven second hop (baseline): the generator writes the query of the second retrieval step.
# "query" and "slots" were chosen on the HotpotQA tuning questions (tune_llm_hop.py).
LLM_HOP = dict(query="followup", slots=1)
PREFIX = {"hotpot": "", "wikihow": "wh_", "wiki2": "w2_", "musique": "mq_"}
STOP = set("the a an of in on at and or to for by with from as is was it this that".split())


def chunk_by_tokens(sentences, min_tok, max_tok):          # the repo's chunker
    chunks, buf, buf_len = [], [], 0
    for s in sentences:
        s_len = len(s.split())
        if buf_len + s_len > max_tok and buf:
            chunks.append(" ".join(buf))
            buf, buf_len = [], 0
        buf.append(s)
        buf_len += s_len
        if buf_len >= min_tok:
            chunks.append(" ".join(buf))
            buf, buf_len = [], 0
    if buf:
        chunks.append(" ".join(buf))
    return chunks


class Corpus:
    def __init__(self, name, dense=None, cross=None):
        self.name = name
        pre = PREFIX[name]
        self.P = R.load_json("wh_passages.json" if name == "wikihow" else f"{pre}paragraphs.json")
        self.S = R.load_json(f"{pre}sentences.json")
        self.Q = R.load_json(f"{pre}questions.json")
        self.title = [p["title"] for p in self.P]
        self.text = [p["text"] for p in self.P]
        self.full = [p["title"] + ". " + p["text"] for p in self.P]
        self.tok = np.asarray([p["tokens"] for p in self.P])
        self.s2p = np.asarray([s["pid"] for s in self.S])
        self.D = dense or R.Dense()
        self.CE = cross or R.Cross()
        sfull = [s["title"] + ". " + s["text"] for s in self.S]
        self.Ep = self.D.corpus(f"{pre}par", self.full)
        self.Es = self.D.corpus(f"{pre}sent", sfull)
        self.Ep0 = self.D.corpus(f"{pre}par_notitle", self.text)
        self.bm_par = R.BM25(self.full, R.tok_word)
        self.bm_sent = R.BM25(sfull, R.tok_word)
        self._links = None
        self.timing = {}

    # ── shared pieces ───────────────────────────────────────────────────────
    def _rerank(self, qs, cands, text):
        sc = self.CE.score([(qs[qi], text[i]) for qi, c in enumerate(cands) for i in c])
        out, pos = [], 0
        for c in cands:
            s = sc[pos:pos + len(c)]
            pos += len(c)
            out.append([c[j] for j in np.argsort(-s, kind="stable")])
        return out

    def _title_index(self):
        if self._links is None:
            norm = lambda t: " ".join(re.findall(r"[a-z0-9]+", t.lower()))
            d = collections.defaultdict(list)
            for i, t in enumerate(self.title):
                d[norm(t)].append(i)
                b = norm(re.sub(r"\s*\([^)]*\)\s*$", "", t))
                if b and b != norm(t):
                    d[b].append(i)
            self._links = d
        return self._links

    def mentioned(self, pid):
        idx = self._title_index()
        w = re.findall(r"[a-z0-9]+", self.text[pid].lower())
        out = []
        for n in range(8, 0, -1):
            for s in range(len(w) - n + 1):
                g = " ".join(w[s:s + n])
                if n == 1 and (g in STOP or len(g) < 4):
                    continue
                out += [i for i in idx.get(g, ()) if i != pid]
        return R.unique_in_order(out)

    def second_hop(self, qs, hop1, slots=None, k=None, bridges=None):
        C = CONFIG
        slots = C["reserved_slots"] if slots is None else slots
        k = C["k"] if k is None else k
        bridges = C["bridges"] if bridges is None else bridges
        links = [[(b, x) for b in hop1[qi][:bridges] for x in self.mentioned(b)] for qi in range(len(qs))]
        sc = self.CE.score([(qs[qi], self.title[b] + ". " + self.text[b] + " " + self.full[x])
                            for qi, c in enumerate(links) for b, x in c])
        out, pos = [], 0
        for qi, c in enumerate(links):
            best = {}
            for b, x in c:
                best[x] = max(best.get(x, -1e9), sc[pos])
                pos += 1
            keep = hop1[qi][:k - slots]
            extra = [x for x in sorted(best, key=lambda x: -best[x]) if x not in keep][:slots]
            out.append(keep + extra + [x for x in hop1[qi] if x not in keep and x not in extra])
        return out

    def hop1_single(self, q):
        """First-hop ranking (fusion and cross-encoder) for one query."""
        C, N = CONFIG, CONFIG["first_stage"]
        qe = self.D.encode([q])
        s2 = lambda ids: R.unique_in_order(self.s2p[ids[0]].tolist())
        lists = [self.bm_par.search([q], N)[0][0].tolist(), R.dense_search(qe, self.Ep, N)[0][0].tolist(),
                 s2(self.bm_sent.search([q], N)[0]), s2(R.dense_search(qe, self.Es, N)[0])]
        return self._rerank([q], [R.rrf(lists, k=C["rrf_k"])[:C["pool"]]], self.full)[0]

    # ── all static systems for a list of questions ──────────────────────────
    def rankings(self, qs, intents):
        C, N = CONFIG, CONFIG["first_stage"]
        L = lambda ids: [r.tolist() for r in ids]
        s2 = lambda ids: [R.unique_in_order(self.s2p[r].tolist()) for r in ids]
        T = {}

        def timed(name, fn):
            t = time.perf_counter()
            r = fn()
            T[name] = (time.perf_counter() - t) / len(qs)
            return r

        qe = timed("encode", lambda: self.D.encode(qs))
        bm_par = timed("bm25_par", lambda: L(self.bm_par.search(qs, N)[0]))
        bm_sent = timed("bm25_sent", lambda: s2(self.bm_sent.search(qs, N)[0]))
        de_text = timed("dense_text", lambda: L(R.dense_search(qe, self.Ep0, N)[0]))
        de_par = timed("dense_par", lambda: L(R.dense_search(qe, self.Ep, N)[0]))
        de_sent = timed("dense_sent", lambda: s2(R.dense_search(qe, self.Es, N)[0]))
        out = {"bm25": bm_par, "fixed": de_text}
        out["rerank"] = timed("ce_rerank", lambda: self._rerank(qs, [r[:C["rerank_pool_baseline"]] for r in de_text], self.text))
        out["strong"] = timed("ce_strong", lambda: self._rerank(qs, [r[:C["pool"]] for r in de_par], self.full))
        f4 = [R.rrf([bm_par[i], de_par[i], bm_sent[i], de_sent[i]], k=C["rrf_k"])[:C["pool"]] for i in range(len(qs))]
        out["hop1"] = timed("ce_hop1", lambda: self._rerank(qs, f4, self.full))
        hop2 = timed("hop2", lambda: self.second_hop(qs, out["hop1"]))
        out["full"] = [h1 if it == "procedural" else h2 for h1, h2, it in zip(out["hop1"], hop2, intents)]
        # wider context, used only when the verification step rejects the first answer
        W = WIDE
        wide = self.second_hop(qs, out["hop1"], slots=W["reserved_slots"], k=W["k"], bridges=W["bridges"])
        self.wide = {"full": [(h1 if it == "procedural" else h2)[:W["k"]] for h1, h2, it in zip(out["hop1"], wide, intents)],
                     "strong": [r[:W["k"]] for r in out["strong"]], "fixed": [r[:W["k"]] for r in out["fixed"]]}
        first = T["encode"] + T["bm25_par"] + T["bm25_sent"] + T["dense_par"] + T["dense_sent"]
        self.timing = {"bm25": T["bm25_par"], "fixed": T["encode"] + T["dense_text"],
                       "rerank": T["encode"] + T["dense_text"] + T["ce_rerank"],
                       "strong": T["encode"] + T["dense_par"] + T["ce_strong"],
                       "hop1": first + T["ce_hop1"], "full": first + T["ce_hop1"] + T["hop2"]}
        return {k: [r[:CONFIG["k"]] for r in v] for k, v in out.items()}


class UamgV1:
    """Heuristic Adaptive RAG (rag_pipeline.run_rag with max_iter = 2, tau = 0.5).

    Policy from query_understanding.py: k = 7 / 8 / 10 for low / medium / high effort; G3 for procedural
    questions, otherwise G2 (G3 for analytical questions of high effort).  Retrieval is dense over the
    chunk text, duplicates of a title are dropped, the uncertainty is 1 - lexical overlap, and the refined
    query is the question plus the last three long words of the answer.
    The corpus is the one of the dataset, as for every other system."""

    def __init__(self, corpus):
        c = self.c = corpus
        if c.name != "wikihow":           # G2 and G3 equal the paragraph for 99.3% / 100% of HotpotQA paragraphs
            self.chunks = {g: (c.title, c.text, c.Ep0) for g in ("G1", "G2", "G3")}
        else:
            by_art = collections.OrderedDict()
            for p in c.P:
                by_art.setdefault(p["aid"], [p["title"], []])[1].append(p["text"])
            split = lambda t: [x for x in re.split(r"(?<=[.!?])\s+", t.strip()) if x]
            self.chunks = {"G2": (c.title, c.text, c.Ep0)}
            titles, texts = [], []
            for aid, (title, parts) in by_art.items():
                for ch in chunk_by_tokens(split(" ".join(parts)), 600, 900):
                    titles.append(title)
                    texts.append(ch)
            self.chunks["G3"] = (titles, texts, c.D.corpus("wh_g3_notitle", texts))
            self.chunks["G1"] = self.chunks["G2"]

    def retrieve(self, query, granularity, k):
        titles, texts, E = self.chunks[granularity]
        ids = R.dense_search(self.c.D.encode([query]), E, k)[0][0].tolist()
        seen, out = set(), []
        for i in ids:
            if titles[i] not in seen:
                seen.add(titles[i])
                out.append({"title": titles[i], "text": texts[i]})
        return out

    @staticmethod
    def uncertainty(answer, results):
        a = set(answer.lower().split())
        ctx = set()
        for r in results:
            ctx.update(r["text"].lower().split())
        return 1.0 if not a or not results else round(1.0 - len(a & ctx) / len(a), 4)

    @staticmethod
    def refine(query, answer):
        words = [w for w in answer.split() if len(w) > 5]
        hint = " ".join(words[-3:]) if words else ""
        return f"{query} {hint}" if hint else query

    def run(self, query, generate, max_iter=2, tau=0.5):
        prof = understand_query(query)
        cur, calls, t_ret = query, 0, 0.0
        for it in range(max_iter + 1):
            t = time.perf_counter()
            res = self.retrieve(cur, prof["granularity"], prof["k"])
            t_ret += time.perf_counter() - t
            answer, meta = generate(query, res, prof["intent"])
            calls += 1
            if it < max_iter:
                if self.uncertainty(answer, res) <= tau:
                    break
                cur = self.refine(query, answer)
        return answer, res, dict(calls=calls, retrieval_s=t_ret, k=prof["k"], granularity=prof["granularity"], **meta)
