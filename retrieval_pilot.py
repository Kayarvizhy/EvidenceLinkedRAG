"""
retrieval_pilot.py  --  retrieval-only evaluation on HotpotQA (no LLM)
======================================================================

    python retrieval_pilot.py --split test      # held-out questions 500-1499 (report these)
    python retrieval_pilot.py --split tune      # questions 0-499, used to choose the settings below

Every setting in CONFIG was chosen on the tuning split (explore.py, explore2.py) and then frozen.

Metrics, all at a context of k paragraphs
  recall   mean share of the two gold supporting paragraphs that were retrieved
  both     share of questions with BOTH gold paragraphs retrieved (needed for a multi-hop answer)
  answer   share of questions whose answer string occurs in the retrieved text (yes/no questions excluded)
  tokens   mean number of whitespace tokens in the k paragraphs (cost of the prompt)
"""
import os, re, sys, json, time, argparse, collections
import numpy as np
import scipy.sparse as sp
import retrieval_lib as R
from repo_query_understanding import understand_query      # unchanged copy from the AdaptiveRAG repo

CONFIG = dict(k=5, first_stage=100, rrf_k=60, pool=50, bridges=3, reserved_slots=1, rerank_pool_baseline=20)
SPLITS = {"hotpot": {"tune": (0, 500), "test": (500, 1500)}, "wiki2": {"test": (0, 1000)}, "musique": {"test": (0, 1000)}}
PREFIX = {"hotpot": "", "wiki2": "w2_", "musique": "mq_"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["tune", "test"], default="test")
    ap.add_argument("--dataset", choices=list(SPLITS), default="hotpot")
    a = ap.parse_args()
    lo, hi = SPLITS[a.dataset][a.split]
    pre = PREFIX[a.dataset]
    C = CONFIG
    K = C["k"]

    P = R.load_json(pre + "paragraphs.json")
    S = R.load_json(pre + "sentences.json")
    Q = R.load_json(pre + "questions.json")[lo:hi]
    qs = [q["question"] for q in Q]
    nq = len(Q)
    ptitle = [p["title"] for p in P]
    ptext = [p["text"] for p in P]
    pfull = [p["title"] + ". " + p["text"] for p in P]
    sfull = [s["title"] + ". " + s["text"] for s in S]
    s2p = np.asarray([s["pid"] for s in S])
    ptok = np.asarray([p["tokens"] for p in P])

    t0 = time.time()
    D = R.Dense()
    Ep, Es, Ep0 = D.corpus(pre + "par", pfull), D.corpus(pre + "sent", sfull), D.corpus(pre + "par_notitle", ptext)
    qe = D.encode(qs)
    bm_paper = R.BM25(ptext, R.tok_split)
    bm_par = R.BM25(pfull, R.tok_word)
    bm_sent = R.BM25(sfull, R.tok_word)
    CE = R.Cross()
    N = C["first_stage"]
    L = lambda ids: [r.tolist() for r in ids]
    sent2par = lambda ids: [R.unique_in_order(s2p[r].tolist()) for r in ids]
    bm_paper_r = L(bm_paper.search(qs, N)[0])
    bm_par_r = L(bm_par.search(qs, N)[0])
    bm_sent_r = sent2par(bm_sent.search(qs, N)[0])
    de_text_r = L(R.dense_search(qe, Ep0, N)[0])
    de_par_r = L(R.dense_search(qe, Ep, N)[0])
    de_sent_r = sent2par(R.dense_search(qe, Es, N)[0])

    def rerank(cands, text=pfull):
        sc = CE.score([(qs[qi], text[i]) for qi, c in enumerate(cands) for i in c])
        out, scores, pos = [], [], 0
        for c in cands:
            s = sc[pos:pos + len(c)]
            pos += len(c)
            o = np.argsort(-s, kind="stable")
            out.append([c[j] for j in o])
            scores.append(s[o])
        return out, scores

    def fuse(lists):
        return [R.rrf([l[qi] for l in lists], k=C["rrf_k"])[:C["pool"]] for qi in range(nq)]

    # ── title links for the second hop ──────────────────────────────────────
    norm = lambda t: " ".join(re.findall(r"[a-z0-9]+", t.lower()))
    by_title = collections.defaultdict(list)
    for i, t in enumerate(ptitle):
        by_title[norm(t)].append(i)
        b = norm(re.sub(r"\s*\([^)]*\)\s*$", "", t))        # "Kiss and Tell (1945 film)" -> "kiss and tell"
        if b and b != norm(t):
            by_title[b].append(i)
    stop = set("the a an of in on at and or to for by with from as is was it this that".split())

    def mentioned(pid):
        w = re.findall(r"[a-z0-9]+", ptext[pid].lower())
        out = []
        for n in range(8, 0, -1):
            for s in range(len(w) - n + 1):
                g = " ".join(w[s:s + n])
                if n == 1 and (g in stop or len(g) < 4):
                    continue
                out += [i for i in by_title.get(g, ()) if i != pid]
        return R.unique_in_order(out)

    def second_hop(hop1, score_mode="path", bridges=C["bridges"], slots=C["reserved_slots"]):
        links = [[(b, x) for b in hop1[qi][:bridges] for x in mentioned(b)] for qi in range(nq)]
        if score_mode == "path":
            pair = lambda qi, b, x: (qs[qi], ptitle[b] + ". " + ptext[b] + " " + pfull[x])
        else:
            pair = lambda qi, b, x: (qs[qi], pfull[x])
        sc = CE.score([pair(qi, b, x) for qi, c in enumerate(links) for b, x in c])
        out, pos, n_links = [], 0, []
        for qi, c in enumerate(links):
            best = {}
            for b, x in c:
                best[x] = max(best.get(x, -1e9), sc[pos])
                pos += 1
            keep = hop1[qi][:K - slots]
            extra = [x for x in sorted(best, key=lambda x: -best[x]) if x not in keep][:slots]
            out.append(keep + extra + [x for x in hop1[qi] if x not in keep and x not in extra])
            n_links.append(len(best))
        return out, float(np.mean(n_links))

    # ── systems ─────────────────────────────────────────────────────────────
    sys_ = collections.OrderedDict()
    sys_["BM25 (as in the paper: text, split tokens)"] = bm_paper_r
    sys_["BM25 (title + text, word tokens)"] = bm_par_r
    sys_["Fixed RAG: dense, text only"] = de_text_r
    sys_["Dense, title + text"] = de_par_r
    sys_["RAG + Reranker: dense top-20 -> cross-encoder"] = rerank([r[:C["rerank_pool_baseline"]] for r in de_text_r], ptext)[0]
    sys_["Strong reranker baseline: dense(title) top-50 -> cross-encoder"] = rerank([r[:C["pool"]] for r in de_par_r])[0]
    f4 = fuse([bm_par_r, de_par_r, bm_sent_r, de_sent_r])
    hop1, hop1_sc = rerank(f4)
    sys_["PROPOSED hop 1: fusion (BM25+dense, paragraph+sentence) -> cross-encoder"] = hop1
    full, n_links = second_hop(hop1)
    sys_["PROPOSED full: hop 1 + title-linked second hop"] = full

    # ── BRIDGE-style baseline (Yazidi et al., IEEE Access 2026): same first stage, then Personalized
    #    PageRank over the title-link graph seeded by the best cross-encoder candidates, and a unified
    #    cross-encoder rerank of the merged pool against the question.  Settings are those published for
    #    BRIDGE (30 kept after reranking, 8 seeds, budget 15, alpha 0.85, seed threshold 0.35); nothing tuned.
    def title_graph():
        path = os.path.join(R.CACHE, f"{pre}title_graph.npz")
        if os.path.exists(path):
            z = np.load(path)
            rows, cols = z["rows"], z["cols"]
        else:
            rows, cols = [], []
            for i in range(len(P)):
                for j in mentioned(i):
                    rows.append(i)
                    cols.append(j)
            rows, cols = np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)
            np.savez(path, rows=rows, cols=cols)
        n = len(P)
        A = sp.csr_matrix((np.ones(len(rows), dtype=np.float64), (rows, cols)), shape=(n, n))  # A[i, j]: i mentions j
        out_deg = np.asarray(A.sum(axis=1)).ravel()
        inv = np.where(out_deg > 0, 1.0 / np.maximum(out_deg, 1), 0.0)
        M = sp.diags(inv) @ A                                       # row-stochastic transition matrix
        return M.T.tocsr(), len(rows)                              # column-stochastic for pi = alpha M^T pi + (1-alpha) p

    def ppr_system(keep=30, seeds=8, budget=15, alpha=0.85, tau=0.35, iters=60):
        MT, n_edges = title_graph()
        n = len(P)
        sig = lambda z: 1.0 / (1.0 + np.exp(-z))
        out, n_disc = [], []
        for qi in range(nq):
            p1 = hop1[qi][:keep]
            cand = [(x, sig(hop1_sc[qi][r])) for r, x in enumerate(p1[:seeds])]
            cand = [(x, w) for x, w in cand if w >= tau] or cand[:1]
            p = np.zeros(n)
            for x, w in cand:
                p[x] = w
            p /= p.sum()
            pi = p.copy()
            for _ in range(iters):
                pi = alpha * (MT @ pi) + (1 - alpha) * p
            inpool = set(p1)
            order = np.argsort(-pi, kind="stable")
            disc = []
            for x in order:
                if pi[x] <= 0:
                    break
                if x not in inpool:
                    disc.append(int(x))
                    if len(disc) == budget:
                        break
            n_disc.append(len(disc))
            out.append(p1 + disc)
        merged, _ = rerank(out)                                    # unified rerank of P1 ∪ D against the question
        print(f"   title graph: {n_edges} edges; PPR discovers {np.mean(n_disc):.1f} passages per question", flush=True)
        return merged

    ppr = ppr_system()
    sys_["BRIDGE-style: hop 1 -> PPR over title links -> unified cross-encoder rerank"] = ppr
    abl = collections.OrderedDict()
    abl["full method"] = full
    abl["no second hop"] = hop1
    abl["second hop scored by question only (no path)"] = second_hop(hop1, "question")[0]
    abl["no sentence-level lists"] = second_hop(rerank(fuse([bm_par_r, de_par_r]))[0])[0]
    abl["no BM25 lists (dense only)"] = second_hop(rerank(fuse([de_par_r, de_sent_r]))[0])[0]
    abl["no dense lists (BM25 only)"] = second_hop(rerank(fuse([bm_par_r, bm_sent_r]))[0])[0]
    abl["no cross-encoder on hop 1 (fusion order)"] = second_hop(f4)[0]
    print(f"computed in {time.time()-t0:.0f}s; {n_links:.1f} linked candidates per question", flush=True)

    # ── metrics ─────────────────────────────────────────────────────────────
    def m(ranks, k=K, kq=None):
        rec, both, ans, tok = [], [], [], []
        for qi, (q, r) in enumerate(zip(Q, ranks)):
            top = r[:(k if kq is None else kq[qi])]
            if "gold_pids" in q:                                   # MuSiQue: the same title occurs with several texts
                h, ng = len(set(top) & set(q["gold_pids"])), len(q["gold_pids"])
            else:
                titles = {ptitle[i] for i in top}
                h, ng = sum(g in titles for g in q["gold_titles"]), len(q["gold_titles"])
            rec.append(h / ng)
            both.append(float(h == ng))
            ans.append(np.nan if q["answer"].lower() in ("yes", "no")
                       else float(any(q["answer"].lower() in ptext[i].lower() for i in top)))
            tok.append(float(ptok[top].sum()))
        return dict(recall=np.asarray(rec), both=np.asarray(both), answer=np.asarray(ans), tokens=np.asarray(tok))

    types = np.asarray([q["type"] for q in Q])
    kinds = sorted(set(types))
    ref_fixed = m(sys_["Fixed RAG: dense, text only"])
    ref_rr = m(sys_["RAG + Reranker: dense top-20 -> cross-encoder"])
    ref_strong = m(sys_["Strong reranker baseline: dense(title) top-50 -> cross-encoder"])
    ci = lambda t: f"{t[0]*100:+.1f} [{t[1]*100:+.1f}, {t[2]*100:+.1f}]"
    rows, out = [], {"split": a.split, "n": nq, "config": C, "systems": {}, "ablation": {}}

    def line(name, r, k=K, kq=None, store=None):
        x = m(r, k, kq)
        rec = dict(k=(k if kq is None else float(np.mean(kq))), recall=x["recall"].mean() * 100, both=x["both"].mean() * 100,
                   answer=np.nanmean(x["answer"]) * 100, tokens=x["tokens"].mean(),
                   both_by_type={t: x["both"][types == t].mean() * 100 for t in kinds},
                   d_both_vs_fixed=R.paired_bootstrap(x["both"], ref_fixed["both"]),
                   d_both_vs_reranker=R.paired_bootstrap(x["both"], ref_rr["both"]),
                   d_both_vs_strong=R.paired_bootstrap(x["both"], ref_strong["both"]))
        if store is not None:
            store[name] = rec
        return rec

    print(f"\n{a.dataset} retrieval, split = {a.split}, n = {nq} ("
          + ", ".join(f"{t} {int((types==t).sum())}" for t in kinds) + f"), corpus = {len(P)} paragraphs\n")
    hdr = (f"{'system':72s} {'k':>4s} {'recall':>7s} {'both':>6s} {'answer':>7s} {'tokens':>7s} "
           + " ".join(f"{t[:11]:>11s}" for t in kinds) + "  d(both) vs Fixed RAG [95% CI]")
    print(hdr)
    print("-" * len(hdr))

    def pr(name, r):
        print(f"{name:72s} {r['k']:4.1f} {r['recall']:7.1f} {r['both']:6.1f} {r['answer']:7.1f} {r['tokens']:7.0f} "
              + " ".join(f"{r['both_by_type'][t]:11.1f}" for t in kinds) + f"  {ci(r['d_both_vs_fixed'])}")

    for name, r in sys_.items():
        pr(name, line(name, r, store=out["systems"]))
    kq = [understand_query(q)["k"] for q in qs]
    pr("Heuristic Adaptive RAG: dense, k = 7/8/10 from keyword effort", line("Heuristic Adaptive RAG (policy k)", de_text_r, kq=kq, store=out["systems"]))
    for k2 in (10,):
        pr(f"Fixed RAG at k = {k2}", line(f"Fixed RAG at k={k2}", de_text_r, k=k2, store=out["systems"]))
        pr(f"RAG + Reranker at k = {k2}", line(f"RAG + Reranker at k={k2}", sys_["RAG + Reranker: dense top-20 -> cross-encoder"], k=k2, store=out["systems"]))
    for k2 in (3,):
        pr(f"PROPOSED full at k = {k2}", line(f"PROPOSED full at k={k2}", second_hop(hop1, slots=1)[0] if False else
           [r[:2] + [x for x in r[K - 1:K]] + r[2:K - 1] for r in full], k=k2, store=out["systems"]))

    p = out["systems"]["PROPOSED full: hop 1 + title-linked second hop"]
    p["d_both_vs_ppr"] = R.paired_bootstrap(m(full)["both"], m(ppr)["both"])
    p["d_recall_vs_ppr"] = R.paired_bootstrap(m(full)["recall"], m(ppr)["recall"])
    print(f"\nPROPOSED full, difference in 'both' at k = {K}:")
    print(f"   vs Fixed RAG                 {ci(p['d_both_vs_fixed'])}")
    print(f"   vs RAG + Reranker            {ci(p['d_both_vs_reranker'])}")
    print(f"   vs strong reranker baseline  {ci(p['d_both_vs_strong'])}")
    print(f"   vs BRIDGE-style PPR          {ci(p['d_both_vs_ppr'])}")

    print(f"\nAblation of the proposed method (k = {K}); d = difference in 'both' to the full method")
    fm = m(full)
    for name, r in abl.items():
        x = line(name, r, store=out["ablation"])
        d = R.paired_bootstrap(m(r)["both"], fm["both"])
        x["d_both_vs_full"] = d
        print(f"   {name:50s} recall {x['recall']:5.1f}  both {x['both']:5.1f}  answer {x['answer']:5.1f}   {'' if name == 'full method' else ci(d)}")

    os.makedirs(os.path.join(R.HERE, "results"), exist_ok=True)
    tag = a.split if a.dataset == "hotpot" else f"{a.dataset}_{a.split}"
    path = os.path.join(R.HERE, "results", f"retrieval_{tag}.json")
    json.dump(out, open(path, "w"), indent=1)
    np.save(os.path.join(R.HERE, "results", f"ranks_{tag}_full.npy"), np.asarray([r[:10] for r in full]))
    np.save(os.path.join(R.HERE, "results", f"ranks_{tag}_fixed.npy"), np.asarray([r[:10] for r in de_text_r]))
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
