"""
latency_bench.py  --  end-to-end latency per question, measured one question at a time

    python latency_bench.py --n 100

Run this while nothing else uses the machine.  Every system answers the same questions of the
HotpotQA test split.  Retrieval is computed for the single question (no batching, no score cache)
and every answer is generated again, so the time is what a user would wait.

Two precautions keep the comparison fair:
  * the order of the systems is shuffled for every question (fixed seed);
  * before every timed generator call an unrelated one-token request is sent, because the server
    keeps the previous prompt in memory and would otherwise evaluate a repeated context faster.
    The time of that request is not counted.
Output: results/latency_hotpot.json
"""
import os, json, time, random, argparse
import numpy as np
import requests
import retrieval_lib as R
import generate_eval as G
from systems import Corpus, UamgV1, SPLITS, CONFIG, WIDE
from repo_query_understanding import understand_query

SYSTEMS = ["bm25", "fixed", "rerank", "strong", "uamg_v1", "llm_hop", "hop1", "full", "full_v"]
CLOCK = {"gen": 0.0, "calls": 0}


def flush(model):
    requests.post(G.OLLAMA, json={"model": model, "prompt": "Hello", "stream": False,
                                  "options": dict(G.OPTIONS, num_predict=1)}, timeout=600)


def generate(query, results, intent, extra=""):
    flush(G.MODEL_MAP[intent])
    t = time.perf_counter()
    out = G.generate(query, results, intent, extra)
    CLOCK["gen"] += time.perf_counter() - t
    CLOCK["calls"] += 1
    return out


def followup(query, results):
    flush(G.MODEL_MAP["factual"])
    t = time.perf_counter()
    out = G.followup_query(query, results)[0]
    CLOCK["gen"] += time.perf_counter() - t
    CLOCK["calls"] += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    a = ap.parse_args()
    C = Corpus("hotpot")
    v1 = UamgV1(C)
    lo, hi = SPLITS["hotpot"]["test"]
    Q = C.Q[lo:hi][:a.n]
    N, K, cfg = CONFIG["first_stage"], CONFIG["k"], CONFIG
    L = lambda ids: ids[0].tolist()
    ctx = lambda pids: [{"title": C.title[p], "text": C.text[p]} for p in pids]

    def retrieve(s, q, intent):
        if s == "bm25":
            return L(C.bm_par.search([q], N)[0])[:K], None
        if s == "fixed":
            return L(R.dense_search(C.D.encode([q]), C.Ep0, N)[0])[:K], None
        if s == "rerank":
            c = L(R.dense_search(C.D.encode([q]), C.Ep0, N)[0])[:cfg["rerank_pool_baseline"]]
            return C._rerank([q], [c], C.text)[0][:K], None
        if s == "strong":
            c = L(R.dense_search(C.D.encode([q]), C.Ep, N)[0])[:cfg["pool"]]
            return C._rerank([q], [c], C.full)[0][:K], None
        h1 = C.hop1_single(q)
        if s in ("hop1", "llm_hop") or intent == "procedural":
            return h1[:K], h1
        return C.second_hop([q], [h1])[0][:K], h1

    generate(Q[0]["question"], ctx(retrieve("fixed", Q[0]["question"], "factual")[0]), "factual")      # load the model
    out = {s: {"retrieval_s": [], "generation_s": [], "total_s": [], "calls": []} for s in SYSTEMS}
    rnd = random.Random(0)
    for i, q in enumerate(Q):
        intent = understand_query(q["question"])["intent"]
        order = SYSTEMS[:]
        rnd.shuffle(order)
        for s in order:
            C.CE.memo.clear()
            CLOCK.update(gen=0.0, calls=0)
            if s == "uamg_v1":
                _, _, meta = v1.run(q["question"], generate)
                t_ret = meta["retrieval_s"]
            else:
                t0 = time.perf_counter()
                pids, h1 = retrieve("full" if s == "full_v" else s, q["question"], intent)
                t_ret = time.perf_counter() - t0
                if s == "llm_hop" and intent != "procedural":
                    fq = followup(q["question"], ctx(pids))
                    t0 = time.perf_counter()
                    pids = G.llm_hop_context(C, q["question"], h1, fq)
                    t_ret += time.perf_counter() - t0
                ans, _ = generate(q["question"], ctx(pids), intent)
                if s == "full_v" and intent != "procedural" and G.verify(ans, G.build_context(ctx(pids)), q["question"]):
                    t0 = time.perf_counter()
                    wide = C.second_hop([q["question"]], [h1], slots=WIDE["reserved_slots"], k=WIDE["k"],
                                        bridges=WIDE["bridges"])[0][:WIDE["k"]]
                    t_ret += time.perf_counter() - t0
                    generate(q["question"], ctx(wide), intent, G.HINT[G.expected_type(q["question"])])
            o = out[s]
            o["retrieval_s"].append(t_ret)
            o["generation_s"].append(CLOCK["gen"])
            o["total_s"].append(t_ret + CLOCK["gen"])
            o["calls"].append(CLOCK["calls"])
        if (i + 1) % 10 == 0:
            print(f"{i+1}/{len(Q)}  " + "  ".join(f"{s} {np.mean(out[s]['total_s']):.2f}s" for s in SYSTEMS), flush=True)
    json.dump({"n": len(Q), "systems": out}, open(os.path.join(R.HERE, "results", "latency_hotpot.json"), "w"))
    print("\nmean seconds per question (retrieval / generation / total / LLM calls)")
    for s in SYSTEMS:
        o = out[s]
        print(f"  {G.LABEL[s].strip():28s} {np.mean(o['retrieval_s']):6.2f} {np.mean(o['generation_s']):6.2f} "
              f"{np.mean(o['total_s']):6.2f} {np.mean(o['calls']):5.2f}")


if __name__ == "__main__":
    main()
