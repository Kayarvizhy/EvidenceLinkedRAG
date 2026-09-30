"""explore.py -- design exploration on the TUNING questions only (qidx 0-499).
Nothing here touches the held-out test questions (qidx 500-1499)."""
import sys, time, pickle, os
import numpy as np
import retrieval_lib as R

P = R.load_json("paragraphs.json")
S = R.load_json("sentences.json")
Q = R.load_json("questions.json")[:500]
qs = [q["question"] for q in Q]
ptitle = [p["title"] for p in P]
ptext = [p["text"] for p in P]
pfull = [p["title"] + ". " + p["text"] for p in P]
s2p = np.asarray([s["pid"] for s in S])
ptok = np.asarray([p["tokens"] for p in P])

t = time.time()
D = R.Dense()
Ep, Es, Ep0 = D.corpus("par", pfull), D.corpus("sent", [s["title"] + ". " + s["text"] for s in S]), D.corpus("par_notitle", ptext)
qe = D.encode(qs)
bm_split = R.BM25(ptext, R.tok_split)
bm_par = R.BM25(pfull, R.tok_word)
bm_sent = R.BM25([s["title"] + ". " + s["text"] for s in S], R.tok_word)
CE = R.Cross()
print(f"setup {time.time()-t:.0f}s", flush=True)

N = 100
bm_split_ids, _ = bm_split.search(qs, N)
bm_par_ids, _ = bm_par.search(qs, N)
bm_sent_ids, _ = bm_sent.search(qs, N)
de_par0_ids, _ = R.dense_search(qe, Ep0, N)
de_par_ids, _ = R.dense_search(qe, Ep, N)
de_sent_ids, _ = R.dense_search(qe, Es, N)
sent2par = lambda ids: [R.unique_in_order(s2p[r].tolist()) for r in ids]
bm_sent_p, de_sent_p = sent2par(bm_sent_ids), sent2par(de_sent_ids)


def metrics(rank_lists, k=5):
    rec, both, ans, tok = [], [], [], []
    for q, r in zip(Q, rank_lists):
        top = list(r[:k])
        titles = {ptitle[i] for i in top}
        hit = sum(1 for g in q["gold_titles"] if g in titles)
        rec.append(hit / 2)
        both.append(float(hit == 2))
        if q["answer"].lower() not in ("yes", "no"):
            ans.append(float(any(q["answer"].lower() in ptext[i].lower() for i in top)))
        tok.append(ptok[top].sum())
    return np.asarray(rec), np.asarray(both), np.asarray(ans), np.asarray(tok)


def show(name, rl, ks=(2, 5, 10), ref=None):
    out = f"{name:44s}"
    for k in ks:
        rec, both, ans, tok = metrics(rl, k)
        out += f" | k={k}: rec {rec.mean()*100:5.1f} both {both.mean()*100:5.1f} ans {ans.mean()*100:5.1f}"
    rec, both, ans, tok = metrics(rl, 5)
    out += f" | tok@5 {tok.mean():4.0f}"
    if ref is not None:
        d, lo, hi = R.paired_bootstrap(both, metrics(ref, 5)[1])
        out += f" | d_both@5 {d*100:+5.1f} [{lo*100:+5.1f},{hi*100:+5.1f}]"
    print(out, flush=True)


def rerank(cands, query_of=lambda qi: qs[qi], text_of=lambda i: pfull[i]):
    pairs, owner = [], []
    for qi, c in enumerate(cands):
        for i in c:
            pairs.append((query_of(qi), text_of(i)))
    sc = CE.score(pairs)
    out, scores, pos = [], [], 0
    for qi, c in enumerate(cands):
        s = sc[pos:pos + len(c)]
        pos += len(c)
        o = np.argsort(-s, kind="stable")
        out.append([c[j] for j in o])
        scores.append(s[o])
    return out, scores


if __name__ == "__main__":
    print("\n=== single retrievers (tuning set, n=500) ===")
    fixed = [r.tolist() for r in de_par0_ids]
    show("BM25 paragraph, split tokens (paper)", [r.tolist() for r in bm_split_ids])
    show("BM25 paragraph + title, word tokens", [r.tolist() for r in bm_par_ids])
    show("BM25 sentence -> parent paragraph", bm_sent_p)
    show("Dense paragraph, text only (Fixed RAG)", fixed)
    show("Dense paragraph + title", [r.tolist() for r in de_par_ids])
    show("Dense sentence -> parent paragraph", de_sent_p)

    print("\n=== reranking and fusion ===")
    t = time.time()
    rr20, _ = rerank([r[:20].tolist() for r in de_par0_ids], text_of=lambda i: ptext[i])
    show("Dense(text) top-20 -> CE (RAG + Reranker)", rr20, ref=fixed)
    for pool in (30, 50, 100):
        f4 = [R.rrf([bm_par_ids[i][:N].tolist(), de_par_ids[i][:N].tolist(), bm_sent_p[i], de_sent_p[i]])[:pool] for i in range(len(Q))]
        if pool == 50:
            show("RRF 4 lists, no rerank", f4, ref=fixed)
            f2 = [R.rrf([bm_par_ids[i].tolist(), de_par_ids[i].tolist()])[:pool] for i in range(len(Q))]
            show("RRF paragraph lists only, no rerank", f2, ref=fixed)
            r2, _ = rerank(f2)
            show("RRF paragraph lists only -> CE", r2, ref=fixed)
        r4, s4 = rerank(f4)
        show(f"RRF 4 lists, pool {pool} -> CE", r4, ref=fixed)
        if pool == 50:
            pickle.dump((f4, r4, s4), open(os.path.join(R.CACHE, "tune_hop1.pkl"), "wb"))
    print(f"rerank time {time.time()-t:.0f}s")
    # upper bound: is the gold paragraph in the pool at all?
    for pool in (20, 50, 100):
        f4 = [R.rrf([bm_par_ids[i].tolist(), de_par_ids[i].tolist(), bm_sent_p[i], de_sent_p[i]])[:pool] for i in range(len(Q))]
        rec, both, ans, _ = metrics(f4, pool)
        print(f"   pool {pool}: gold recall in pool {rec.mean()*100:.1f}, both in pool {both.mean()*100:.1f}")
    for t_ in ("bridge", "comparison"):
        idx = [i for i, q in enumerate(Q) if q["type"] == t_]
        b_fix = metrics(fixed, 5)[1][idx].mean() * 100
        b_new = metrics(r4, 5)[1][idx].mean() * 100
        print(f"   {t_:10s} n={len(idx):3d}  both@5: Fixed RAG {b_fix:.1f}  fusion+CE(pool 100) {b_new:.1f}")
