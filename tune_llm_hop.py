"""
tune_llm_hop.py  --  choose the settings of the LLM-driven second-hop baseline

    python tune_llm_hop.py --n 200

Uses HotpotQA tuning questions only.  The generator writes one follow-up query per question
(cached in results/followup_tune.json); the variants differ in how the query is used and in the
number of context positions given to the second retrieval step.  The variant with the highest
share of questions with both supporting paragraphs is written into systems.LLM_HOP by hand.
The baseline is thereby given its best configuration.
"""
import os, json, argparse
import numpy as np
import retrieval_lib as R
import generate_eval as G
from systems import Corpus, SPLITS, CONFIG
from repo_query_understanding import understand_query


PROMPT_B = """You are helping to search a document collection.

Question: {query}

Paragraphs found so far:
{context}

To answer the question, one more document has to be found. Name the person, place, work or
organisation whose document is needed. Take the name from the paragraphs if it occurs there.
Reply with the name only.

Name:"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--prompt", choices=["A", "B"], default="A", help="A: search query; B: name of the entity to look up")
    a = ap.parse_args()
    if a.prompt == "B":
        G.FOLLOWUP = PROMPT_B
    C = Corpus("hotpot")
    lo, hi = SPLITS["hotpot"]["tune"]
    Q = C.Q[lo:hi][:a.n]
    qs = [q["question"] for q in Q]
    intents = [understand_query(q)["intent"] for q in qs]
    ranks = C.rankings(qs, intents)
    path = os.path.join(R.HERE, "results", "followup_tune.json" if a.prompt == "A" else "followup_tune_B.json")
    fq = json.load(open(path, encoding="utf8")) if os.path.exists(path) else {}
    for i, q in enumerate(Q):
        if str(q["qidx"]) not in fq:
            ctx = [{"title": C.title[p], "text": C.text[p]} for p in ranks["hop1"][i]]
            fq[str(q["qidx"])] = G.followup_query(q["question"], ctx)[0]
            json.dump(fq, open(path, "w", encoding="utf8"), ensure_ascii=False, indent=0)
        if (i + 1) % 25 == 0:
            print(f"follow-up queries {i+1}/{len(Q)}", flush=True)

    def both(lists):
        return np.asarray([float(all(g in {C.title[p] for p in l[:CONFIG['k']]} for g in q["gold_titles"]))
                           for q, l in zip(Q, lists)])

    print(f"\nHotpotQA tuning questions, n = {len(Q)}, k = {CONFIG['k']}")
    print(f"  {'first hop only':44s} both {both(ranks['hop1']).mean()*100:5.1f}")
    print(f"  {'title-linked second hop (proposed)':44s} both {both(ranks['full']).mean()*100:5.1f}")
    for mode in ("followup", "question+followup"):
        for slots in (1, 2):
            ls = [G.llm_hop_context(C, q["question"], ranks["hop1"][i], fq[str(q["qidx"])], mode, slots) for i, q in enumerate(Q)]
            print(f"  LLM-driven, query = {mode:18s} slots = {slots}  both {both(ls).mean()*100:5.1f}")
    print("\nexamples of follow-up queries:")
    for q in Q[:6]:
        print("   Q:", q["question"][:90], "\n      ->", fq[str(q["qidx"])][:90])


if __name__ == "__main__":
    main()
