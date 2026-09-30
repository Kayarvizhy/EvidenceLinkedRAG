"""explore2.py -- second-hop variants, TUNING questions only (qidx 0-499)."""
import re, pickle, os, time, collections
import numpy as np
import retrieval_lib as R
import explore as E
from explore import Q, qs, P, ptitle, ptext, pfull, metrics, show, rerank, CE

f4, r4, s4 = pickle.load(open(os.path.join(R.CACHE, "tune_hop1.pkl"), "rb"))
fixed = [r.tolist() for r in E.de_par0_ids]
print("hop-1 reference:")
show("RRF 4 lists, pool 50 -> CE", r4, ref=fixed)

# ── title index: which corpus titles are mentioned in a text? ────────────────
norm = lambda t: " ".join(re.findall(r"[a-z0-9]+", t.lower()))
title_pids = collections.defaultdict(list)
for i, t in enumerate(ptitle):
    title_pids[norm(t)].append(i)
base_title = collections.defaultdict(list)          # "Kiss and Tell (1945 film)" -> also "kiss and tell"
for i, t in enumerate(ptitle):
    b = norm(re.sub(r"\s*\([^)]*\)\s*$", "", t))
    if b and b != norm(t):
        base_title[b].append(i)
MAXN = 8
STOP = set("the a an of in on at and or to for by with from as is was it this that".split())


def mentioned(text, exclude_pid):
    w = re.findall(r"[a-z0-9]+", text.lower())
    out = []
    for n in range(MAXN, 0, -1):
        for s in range(len(w) - n + 1):
            g = " ".join(w[s:s + n])
            if n == 1 and (g in STOP or len(g) < 4):
                continue
            for d in (title_pids, base_title):
                if g in d:
                    out += [i for i in d[g] if i != exclude_pid]
    return R.unique_in_order(out)


def hop2_links(m):
    """Candidates linked from the top-m hop-1 paragraphs: titles they mention, per question."""
    cands = []
    for qi in range(len(Q)):
        c = []
        for b in r4[qi][:m]:
            c += [(b, x) for x in mentioned(ptext[b], b)]
        cands.append(c)
    return cands


def evaluate(name, lists):
    show(name, lists, ref=fixed)
    for t_ in ("bridge", "comparison"):
        idx = [i for i, q in enumerate(Q) if q["type"] == t_]
        print(f"      {t_:10s} both@5 {metrics(lists, 5)[1][idx].mean()*100:.1f}   rec@5 {metrics(lists, 5)[0][idx].mean()*100:.1f}")


if __name__ == "__main__":
    t = time.time()
    for m in (2, 3):
        links = hop2_links(m)
        n_c = np.mean([len(set(x for _, x in c)) for c in links])
        gold_in = np.mean([all(g in ({ptitle[x] for _, x in c} | {ptitle[b] for b in r4[qi][:m]}) for g in Q[qi]["gold_titles"])
                           for qi, c in enumerate(links)])
        print(f"\n--- bridges = top-{m} hop-1 paragraphs; {n_c:.1f} linked candidates per question; "
              f"both gold within bridges+links: {gold_in*100:.1f}%")
        # score each linked candidate with the question extended by the bridge title
        pairs = []
        for qi, c in enumerate(links):
            for b, x in c:
                pairs.append((qs[qi] + " " + ptitle[b], pfull[x]))
                pairs.append((qs[qi], pfull[x]))
                pairs.append((qs[qi], ptitle[b] + ". " + ptext[b] + " " + pfull[x]))
        sc = CE.score(pairs)
        pos = 0
        S_ext, S_q, S_path = [], [], []
        for qi, c in enumerate(links):
            a, b_, p_ = {}, {}, {}
            for b, x in c:
                a[x] = max(a.get(x, -99), sc[pos]); b_[x] = max(b_.get(x, -99), sc[pos + 1]); p_[x] = max(p_.get(x, -99), sc[pos + 2])
                pos += 3
            S_ext.append(a); S_q.append(b_); S_path.append(p_)
        hop1 = [dict(zip(r4[qi], s4[qi])) for qi in range(len(Q))]

        for label, SS in (("question + bridge title", S_ext), ("question only", S_q), ("path: bridge text + candidate", S_path)):
            # A: reserve slots -- keep the hop-1 top (5-j), give j slots to the best linked candidates
            for j in (1, 2):
                out = []
                for qi in range(len(Q)):
                    keep = r4[qi][:5 - j]
                    extra = [x for x in sorted(SS[qi], key=lambda x: -SS[qi][x]) if x not in keep][:j]
                    out.append(keep + extra + [x for x in r4[qi] if x not in keep and x not in extra])
                evaluate(f"m={m} [{label}] reserve {j} slot(s)", out)
            # B: one ranking -- every paragraph gets max(hop-1 score, linked score)
            out = []
            for qi in range(len(Q)):
                s = dict(hop1[qi])
                for x, v in SS[qi].items():
                    s[x] = max(s.get(x, -99), v)
                out.append(sorted(s, key=lambda x: -s[x]))
            evaluate(f"m={m} [{label}] merged by max score", out)
    print(f"time {time.time()-t:.0f}s")
