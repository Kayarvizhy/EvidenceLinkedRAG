"""
generate_eval.py  --  answer generation for every retrieval system (resumable)
==============================================================================

    python generate_eval.py --dataset hotpot  --split test --n 500
    python generate_eval.py --dataset wikihow --split test --n 150
    python generate_eval.py --dataset hotpot  --split test --n 500 --tables     # print tables only
    python generate_eval.py --dataset hotpot  --split test --check              # failed generations

Same generator, prompt and decoding for every system.  Generator and prompt follow the detected intent,
as in rag_pipeline.py: llama3 for factual and analytical questions, mistral for procedural ones.
Decoding is deterministic: temperature 0, seed 0.

Results are appended to results/gen_<dataset>_<split>.jsonl.  A failed generation is never scored;
it is run again on the next call.
"""
import os, re, sys, json, time, argparse, collections
import numpy as np
import requests
import retrieval_lib as R
import metrics_eval as ME
from systems import Corpus, UamgV1, SPLITS, CONFIG, LLM_HOP, PREFIX
from repo_query_understanding import understand_query

OLLAMA = "http://localhost:11434/api/generate"
MODEL_MAP = {"factual": "llama3", "analytical": "llama3", "procedural": "mistral"}
OPTIONS = {"temperature": 0, "seed": 0, "num_predict": 256, "num_ctx": 8192}
PROMPTS = {
    "factual": """You are a precise factual assistant.
Use the context below to answer the question.
Extract the most relevant answer even if
details differ slightly from the question.
Give the shortest possible answer - just
the key fact, name, date or term.
Do not say I don't know if related
information exists in the context.

Context:
{context}

Question: {query}

Answer concisely and directly:""",
    "analytical": """You are an analytical assistant.
Use the context below to provide a well-reasoned explanation.
Think step by step.

Context:
{context}

Question: {query}

Detailed Answer:""",
    # The repo's procedural prompt asks for "clear, numbered steps" and gives answers of about 225 tokens
    # against references of about 70.  The length limit below is the only change, for every system.
    "procedural": """You are a helpful step-by-step assistant.
Use the context below to answer the question with the main steps.
Write one short sentence per step, at most 100 words in total, without an introduction.

Context:
{context}

Question: {query}

Steps:""",
}
SYSTEMS = ["bm25", "fixed", "rerank", "strong", "uamg_v1", "hop1", "llm_hop", "full", "full_v", "full_vh", "full_vr", "strong_v", "fixed_v"]
FOLLOWUP = """You are helping to search a document collection.

Question: {query}

Paragraphs found so far:
{context}

The paragraphs may not contain everything that is needed to answer the question.
Write one short search query for the missing information. Name the person, place,
work or organisation that has to be looked up. Reply with the search query only.

Search query:"""
# verification variants: (system whose first answer is checked, use the wider context, add the answer-type hint)
VERIFY = {"full_v": ("full", True, True), "full_vh": ("full", False, True), "full_vr": ("full", True, False),
          "strong_v": ("strong", True, True), "fixed_v": ("fixed", True, True)}
LABEL = {"bm25": "BM25 + LLM", "fixed": "Fixed RAG (k=5)", "rerank": "RAG + Reranker", "strong": "Strong reranker baseline",
         "uamg_v1": "Heuristic Adaptive RAG", "hop1": "Proposed, no second hop", "full": "Proposed retrieval",
         "full_v": "Proposed + verification", "full_vh": "  verification: hint only", "full_vr": "  verification: context only",
         "strong_v": "Strong baseline + verif.", "fixed_v": "Fixed RAG + verification",
         "llm_hop": "LLM-driven second hop"}


# ── verification of the first answer ────────────────────────────────────────
AUX = ("is", "are", "was", "were", "do", "does", "did", "has", "have", "had", "can", "could", "will", "would", "should")
NUMWORD = set("zero one two three four five six seven eight nine ten eleven twelve twenty thirty forty fifty hundred thousand million billion".split())
ABSTAIN = re.compile(r"^\W*(none|unknown|n/a|not (mentioned|specified|provided|stated|available|given))\b|"
                     r"(does|do|did) not (mention|provide|contain|specify|say|state|give)|no (information|mention|direct answer)|"
                     r"not (mentioned|specified|stated) in|cannot be determined|unable to determine", re.I)
HINT = {"yesno": "Answer with yes or no only.", "date": "Answer with the year or date only.",
        "number": "Answer with the number only.", "other": "Answer with the name or short phrase that the question asks for."}


def expected_type(question):
    q = question.lower().strip()
    w = q.split()
    if w and w[0] in AUX and " or " not in q:
        return "yesno"
    if re.search(r"\b(what|which|in what|in which) (year|date|decade|century)\b|^when\b|\bwhen (was|were|did|is)\b", q):
        return "date"
    if re.search(r"\bhow (many|much|old|long|far|tall|large|big)\b|\bwhat is the population\b", q):
        return "number"
    return "other"


def verify(answer, context, question):
    """Reason why the answer is rejected, or None if it is accepted."""
    a = answer.strip()
    if ABSTAIN.search(a):
        return "abstention"
    t = expected_type(question)
    low = ME.normalize(a).split()
    if t == "yesno" and not (low and low[0] in ("yes", "no")):
        return "type: yes/no expected"
    if t == "date" and not re.search(r"\b\d{3,4}\b", a):
        return "type: date expected"
    if t == "number" and not (re.search(r"\d", a) or NUMWORD & set(low)):
        return "type: number expected"
    if ME.overlap(a, context) < 0.5 and t != "yesno":
        return "low evidence overlap"
    return None


def build_context(results):
    return "\n\n".join(f"[{i+1}] {r['title']}\n{r['text']}" for i, r in enumerate(results) if r["text"])


def generate(query, results, intent, extra=""):
    prompt = PROMPTS[intent].format(context=build_context(results), query=query + ("\n" + extra if extra else ""))
    t = time.perf_counter()
    r = requests.post(OLLAMA, json={"model": MODEL_MAP[intent], "prompt": prompt, "stream": False, "options": OPTIONS},
                      timeout=1200)
    r.raise_for_status()
    j = r.json()
    ans = j.get("response", "").strip()
    if not ans:
        raise RuntimeError("empty answer")
    return ans, dict(gen_s=time.perf_counter() - t, prompt_tokens=j.get("prompt_eval_count", 0),
                     out_tokens=j.get("eval_count", 0), model=MODEL_MAP[intent])


def followup_query(query, results):
    """Second-hop query written by the generator (one additional generator call)."""
    t = time.perf_counter()
    r = requests.post(OLLAMA, json={"model": MODEL_MAP["factual"], "stream": False,
                                    "prompt": FOLLOWUP.format(context=build_context(results), query=query),
                                    "options": dict(OPTIONS, num_predict=40)}, timeout=1200)
    r.raise_for_status()
    text = r.json().get("response", "").strip().splitlines()
    text = text[0].strip().strip('"') if text else ""
    return text, time.perf_counter() - t


def llm_hop_context(C, query, hop1, followup, mode=None, slots=None):
    mode = LLM_HOP["query"] if mode is None else mode
    slots = LLM_HOP["slots"] if slots is None else slots
    if not followup:
        return list(hop1[:CONFIG["k"]])
    q2 = followup if mode == "followup" else f"{query} {followup}"
    second = C.hop1_single(q2)
    keep = list(hop1[:CONFIG["k"] - slots])
    return keep + [x for x in second if x not in keep][:slots]


def load(path):
    cache = {}
    if os.path.exists(path):
        for line in open(path, encoding="utf8"):
            if line.strip():
                r = json.loads(line)
                cache[(r["system"], r["qidx"])] = r
    return cache


def bad(r):
    return bool(r.get("error")) or not (r.get("answer") or "").strip()


def score(r):
    a, g, c = r["answer"], r["gold"], r["context"]
    return dict(em=ME.em(a, g), f1=ME.f1(a, g), rouge=ME.rouge_l(a, g), contains=ME.contains(a, g),
                overlap=ME.overlap(a, c), unsupported=ME.unsupported(a, c, r["question"]))


def run(a, path, cache):
    lo, hi = SPLITS[a.dataset][a.split]
    C = Corpus(a.dataset)
    Q = C.Q[lo:hi][:a.n]
    qs = [q["question"] for q in Q]
    intents = [understand_query(q)["intent"] for q in qs]
    print("detected intents:", dict(collections.Counter(intents)), flush=True)
    ranks = C.rankings(qs, intents)
    v1 = UamgV1(C)
    systems = [s for s in SYSTEMS if s in a.systems.split(",")]
    todo = [(s, i) for s in systems for i in range(len(Q))
            if (s, Q[i]["qidx"]) not in cache or bad(cache[(s, Q[i]["qidx"])])]
    # a system that gives the same context as one already run reuses its answer (decoding is deterministic)
    print(f"{len(todo)} generations to run, {len(cache)} cached", flush=True)
    memo = {}
    for r in cache.values():
        if not bad(r) and r["system"] not in ("uamg_v1", "llm_hop") and r["system"] not in VERIFY:
            memo[(r["qidx"], tuple(r["pids"]))] = r
    t_start = time.time()
    for n, (s, i) in enumerate(todo, 1):
        q = Q[i]
        rec = dict(system=s, dataset=a.dataset, qidx=q["qidx"], question=q["question"], gold=q["answer"],
                   intent=intents[i], error=None)
        try:
            if s == "uamg_v1":
                ans, res, meta = v1.run(q["question"], generate)
                rec.update(pids=[], retrieval_s=meta.pop("retrieval_s"))
            elif s == "llm_hop":
                h1 = ranks["hop1"][i]
                if intents[i] == "procedural":
                    pids, fq, t_f, t_r = h1, "", 0.0, 0.0
                else:
                    fq, t_f = followup_query(q["question"], [{"title": C.title[p], "text": C.text[p]} for p in h1])
                    t0 = time.perf_counter()
                    pids = llm_hop_context(C, q["question"], h1, fq)
                    t_r = time.perf_counter() - t0
                res = [{"title": C.title[p], "text": C.text[p]} for p in pids]
                ans, meta = generate(q["question"], res, intents[i])
                meta.update(gen_s=meta["gen_s"] + t_f, calls=1 if intents[i] == "procedural" else 2, k=len(pids), followup=fq)
                rec.update(pids=[int(p) for p in pids], retrieval_s=C.timing["hop1"] + t_r)
            elif s in VERIFY:
                base, wider, hint = VERIFY[s]
                b = cache.get((base, q["qidx"]))
                if b is None or bad(b):
                    raise RuntimeError(f"run system '{base}' first")
                pids = b["pids"]
                res = [{"title": C.title[p], "text": C.text[p]} for p in pids]
                ans, reason = b["answer"], verify(b["answer"], b["context"], q["question"])
                meta = dict(gen_s=b["gen_s"], prompt_tokens=b["prompt_tokens"], out_tokens=b["out_tokens"],
                            model=b["model"], calls=1, k=len(pids), rejected=reason, first_answer=b["answer"])
                if reason and intents[i] != "procedural":
                    if wider:
                        pids = C.wide[base][i]
                        res = [{"title": C.title[p], "text": C.text[p]} for p in pids]
                    a2, m2 = generate(q["question"], res, intents[i], HINT[expected_type(q["question"])] if hint else "")
                    meta.update(gen_s=b["gen_s"] + m2["gen_s"], prompt_tokens=m2["prompt_tokens"], calls=2, k=len(pids),
                                out_tokens=m2["out_tokens"])
                    if not ABSTAIN.search(a2):
                        ans = a2
                    else:
                        pids, res = b["pids"], [{"title": C.title[p], "text": C.text[p]} for p in b["pids"]]
                rec.update(pids=[int(p) for p in pids], retrieval_s=C.timing[base])
            else:
                pids = ranks[s][i]
                res = [{"title": C.title[p], "text": C.text[p]} for p in pids]
                key = (q["qidx"], tuple(pids))
                if key in memo:
                    m = memo[key]
                    ans, meta = m["answer"], dict(gen_s=m["gen_s"], prompt_tokens=m["prompt_tokens"],
                                                  out_tokens=m["out_tokens"], model=m["model"], reused=m["system"])
                else:
                    ans, meta = generate(q["question"], res, intents[i])
                meta.update(calls=1, k=len(pids))
                rec.update(pids=[int(p) for p in pids], retrieval_s=C.timing[s])
            rec.update(answer=ans, context=build_context(res), context_titles=[r["title"] for r in res],
                       context_tokens=sum(len(r["text"].split()) for r in res), **meta)
            if a.dataset != "wikihow":
                if "gold_pids" in q and rec.get("pids"):            # MuSiQue: the same title occurs with several texts
                    rec["both_gold"] = float(all(p in set(rec["pids"]) for p in q["gold_pids"]))
                else:
                    got = set(rec["context_titles"])
                    rec["both_gold"] = float(all(g in got for g in q["gold_titles"]))
            else:
                rec["gold_article"] = float(C.title[q["gold_pids"][0]] in set(rec["context_titles"]))
            rec.update(score(rec))
            if s not in ("uamg_v1", "llm_hop") and s not in VERIFY:
                memo[(q["qidx"], tuple(rec["pids"]))] = rec
        except Exception as e:
            rec.update(answer="", error=str(e)[:300])
        cache[(s, q["qidx"])] = rec
        with open(path, "a", encoding="utf8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if n % 25 == 0 or n == len(todo) or rec["error"]:
            el = time.time() - t_start
            print(f"[{n}/{len(todo)}] {s:8s} q{q['qidx']:5d} f1={rec.get('f1', 0):.2f} "
                  f"elapsed {el/60:.0f} min, remaining about {el/n*(len(todo)-n)/60:.0f} min"
                  + (f"  ERROR {rec['error']}" if rec["error"] else ""), flush=True)


def tables(a, cache):
    lo, hi = SPLITS[a.dataset][a.split]
    qidx = list(range(lo, hi))[:a.n]
    systems = [s for s in SYSTEMS if all((s, i) in cache and not bad(cache[(s, i)]) for i in qidx)]
    missing = [s for s in SYSTEMS if s not in systems]
    print(f"\n{a.dataset} / {a.split}: n = {len(qidx)}; complete systems: {systems}" + (f"; incomplete: {missing}" if missing else ""))
    if not systems:
        return
    V = lambda s, m: np.asarray([cache[(s, i)][m] for i in qidx], dtype=np.float64)
    lat = lambda s: V(s, "retrieval_s") + V(s, "gen_s") * (V(s, "calls") if s == "uamg_v1" else 1.0)
    ret = "both_gold" if a.dataset != "wikihow" else "gold_article"
    cols = [("em", "EM"), ("f1", "F1"), ("rouge", "RougeL"), ("contains", "Contains"), ("overlap", "Overlap"), ("unsupported", "Unsupp."), (ret, "Evidence")]
    print(f"\n{'system':28s}" + "".join(f"{h:>10s}" for _, h in cols) + f"{'ctx tok':>9s}{'calls':>7s}{'latency s':>11s}")
    out = {}
    for s in systems:
        row = {m: V(s, m).mean() * 100 for m, _ in cols}
        row.update(f1_sd=V(s, "f1").std(ddof=1) * 100, ctx=V(s, "context_tokens").mean(), calls=V(s, "calls").mean(), latency=lat(s).mean())
        out[s] = row
        print(f"{LABEL[s]:28s}" + "".join(f"{row[m]:10.1f}" for m, _ in cols) + f"{row['ctx']:9.0f}{row['calls']:7.2f}{row['latency']:11.2f}")
    ci = lambda t: f"{t[0]*100:+5.1f} [{t[1]*100:+5.1f}, {t[2]*100:+5.1f}]" + ("*" if t[1] > 0 or t[2] < 0 else " ")
    ref = a.ref if a.ref in systems else "full"
    if ref in systems:
        print(f"\n{LABEL[ref].strip()} minus other system, paired bootstrap 95% CI (* = interval excludes zero)")
        print(f"{'':28s}{'F1':>24s}{'EM':>24s}{'Overlap':>24s}{'Unsupported':>24s}")
        for s in systems:
            if s == ref:
                continue
            d = {m: R.paired_bootstrap(V(ref, m), V(s, m)) for m in ("f1", "em", "overlap", "unsupported")}
            out[s]["delta_full"] = d
            print(f"{LABEL[s]:28s}" + "".join(f"{ci(d[m]):>24s}" for m in ("f1", "em", "overlap", "unsupported")))
    if a.dataset != "wikihow":
        Q = R.load_json(PREFIX[a.dataset] + "questions.json")
        ty = np.asarray([Q[i]["type"] for i in qidx])
        yn = np.asarray([Q[i]["answer"].lower() in ("yes", "no") for i in qidx])
        kinds = sorted(set(ty))
        print("\nF1 by question type (" + ", ".join(f"{k} n={int((ty==k).sum())}" for k in kinds) + f", yes/no answers n={int(yn.sum())})")
        for s in systems:
            f = V(s, "f1")
            print(f"{LABEL[s]:28s} " + "   ".join(f"{k} {f[ty==k].mean()*100:5.1f}" for k in kinds)
                  + f"   yes/no {f[yn].mean()*100:5.1f}   span {f[~yn].mean()*100:5.1f}")
        if "full" in systems:
            f, b = V("full", "f1"), V("full", "both_gold")
            print(f"\nProposed (full): F1 when both gold paragraphs are in the context {f[b==1].mean()*100:.1f} (n={int(b.sum())}), "
                  f"otherwise {f[b==0].mean()*100:.1f} (n={int((b==0).sum())})")
    json.dump(out, open(os.path.join(R.HERE, "results", f"gen_{a.dataset}_{a.split}_summary.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["hotpot", "wikihow", "wiki2", "musique"], required=True)
    ap.add_argument("--split", choices=["tune", "test"], default="test")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--systems", default="bm25,fixed,rerank,strong,uamg_v1,hop1,full")
    ap.add_argument("--ref", default="full", help="system that the difference table compares against")
    ap.add_argument("--tables", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    path = os.path.join(R.HERE, "results", f"gen_{a.dataset}_{a.split}.jsonl")
    cache = load(path)
    if a.check:
        c = collections.Counter((r["system"], bad(r)) for r in cache.values())
        for s in SYSTEMS:
            print(f"{s:10s} ok {c[(s, False)]:5d}  failed {c[(s, True)]:5d}")
        return
    if not a.tables:
        run(a, path, cache)
    tables(a, cache)


if __name__ == "__main__":
    main()
