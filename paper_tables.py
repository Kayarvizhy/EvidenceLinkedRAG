"""
paper_tables.py  --  the tables of the paper that are computed from the per-question results

    python paper_tables.py

Reads    results/gen_hotpot_test.jsonl, results/gen_wiki2_test.jsonl   (one record per system and question)
         cache/questions.json, cache/w2_questions.json                 (question types; written by prepare*.py)

Prints   answer quality by question type            (Table 8 of the paper)
         ablation of the second hop and verification (Table 10)
         rejected answers and effect of refinement   (Table 12)
         classification of the final answers         (Table 14)

The other tables are printed by the scripts that produce the results:
    retrieval_pilot.py      evidence retrieval and retrieval ablation   (Tables 6 and 9; results/retrieval_*.txt)
    generate_eval.py        answer quality                              (Table 7;  results/tables_*.txt)
    latency_bench.py        latency                                     (Table 13; results/run_latency.log)
    prepare.py              fixed-size chunking statistics              (Table 11; cache/repo_chunks.json)

No model is loaded and no generator is called; the script only reads the result files.
"""
import os, collections
import numpy as np
import retrieval_lib as R
import generate_eval as G

RES = os.path.join(R.HERE, "results")
OURS = "full_v"          # proposed pipeline with verification
SETS = collections.OrderedDict([
    ("HotpotQA", ("gen_hotpot_test.jsonl", list(range(500, 1000)), "questions.json", ("bridge", "comparison"))),
    ("2WikiMultihopQA", ("gen_wiki2_test.jsonl", list(range(0, 500)), "w2_questions.json",
                         ("compositional", "inference", "comparison", "bridge_comparison"))),
])


def V(c, s, m, idx):
    return np.asarray([c[(s, i)][m] for i in idx], dtype=np.float64)


def ci(a, b):
    d, lo, hi = R.paired_bootstrap(a, b)
    return f"{d*100:+5.1f} [{lo*100:+5.1f}, {hi*100:+5.1f}]" + ("*" if lo > 0 or hi < 0 else " ")


def by_type(name, c, idx, Q, types):
    print(f"\n{name}: answer quality by question type (F1 in percent; All = all supporting paragraphs in the context)")
    print(f"{'type':20s}{'n':>5s}{'Fixed':>8s}{'Strong':>8s}{'LLM hop':>9s}{'Ours':>7s}   {'Ours - Strong [95% CI]':26s}{'All strong':>11s}{'All ours':>10s}")
    for t in types:
        ii = [i for i in idx if Q[i]["type"] == t]
        f = lambda s, m="f1": V(c, s, m, ii).mean() * 100
        print(f"{t:20s}{len(ii):5d}{f('fixed'):8.1f}{f('strong'):8.1f}{f('llm_hop'):9.1f}{f(OURS):7.1f}   "
              f"{ci(V(c, OURS, 'f1', ii), V(c, 'strong', 'f1', ii)):26s}{f('strong', 'both_gold'):11.1f}{f(OURS, 'both_gold'):10.1f}")


def ablation(name, c, idx):
    print(f"\n{name}: second hop and verification (values in percent; difference = configuration minus full pipeline)")
    print(f"{'configuration':40s}{'EM':>7s}{'F1':>7s}{'Unsupp.':>9s}   F1 difference [95% CI]")
    for lab, s in (("proposed pipeline", OURS), ("without verification", "full"), ("without verification and second hop", "hop1"),
                   ("verification: hint only", "full_vh"), ("verification: wider context only", "full_vr")):
        if not all((s, i) in c for i in idx):
            continue
        m = lambda k: V(c, s, k, idx).mean() * 100
        print(f"{lab:40s}{m('em'):7.1f}{m('f1'):7.1f}{m('unsupported'):9.1f}   "
              + ("" if s == OURS else ci(V(c, s, "f1", idx), V(c, OURS, "f1", idx))))


def rejections(name, c, idx):
    print(f"\n{name}: rejected first answers and effect of the refinement (F1 in percent)")
    rows = [c[(OURS, i)] for i in idx]
    rej = [r for r in rows if r.get("rejected")]
    first = lambda rs: np.mean([c[("full", r["qidx"])]["f1"] for r in rs]) * 100 if rs else float("nan")
    final = lambda rs: np.mean([r["f1"] for r in rs]) * 100 if rs else float("nan")
    print(f"{'reason':28s}{'n':>5s}{'F1 first':>10s}{'F1 final':>10s}")
    for k in ("abstention", "type: yes/no expected", "type: date expected", "type: number expected", "low evidence overlap"):
        rs = [r for r in rej if r["rejected"] == k]
        print(f"{k:28s}{len(rs):5d}{first(rs):10.1f}{final(rs):10.1f}")
    acc = [r for r in rows if not r.get("rejected")]
    print(f"{'all rejected answers':28s}{len(rej):5d}{first(rej):10.1f}{final(rej):10.1f}")
    print(f"{'accepted answers':28s}{len(acc):5d}{final(acc):10.1f}{final(acc):10.1f}")
    print(f"generator calls per question: {V(c, OURS, 'calls', idx).mean():.2f}")


def classify(c, s, idx):
    in_ctx = lambda r: G.ME.normalize(r["gold"]) in G.ME.normalize(r["context"])
    rows = [c[(s, i)] for i in idx]
    part = [r for r in rows if 0 < r["f1"] and r["em"] == 0]
    zero = [r for r in rows if r["f1"] == 0]
    yn = [r for r in zero if r["gold"].lower() in ("yes", "no")]
    ab = [r for r in zero if r not in yn and G.ABSTAIN.search(r["answer"])]
    span = [r for r in zero if r not in yn and r not in ab]
    return collections.OrderedDict([
        ("exact match", [r for r in rows if r["em"] == 1]),
        ("partial match, answer contains the reference", [r for r in part if r["contains"] == 1]),
        ("partial match, other", [r for r in part if r["contains"] != 1]),
        ("F1 = 0, yes/no reference", yn),
        ("F1 = 0, abstention", ab),
        ("F1 = 0, reference absent from the context (retrieval error)", [r for r in span if not in_ctx(r)]),
        ("F1 = 0, reference present in the context (reading error)", [r for r in span if in_ctx(r)])])


def errors(name, c, idx):
    print(f"\n{name}: classification of the final answers (n = {len(idx)})")
    e, e1 = classify(c, OURS, idx), classify(c, "hop1", idx)
    for k, rs in e.items():
        print(f"{k:62s}{len(rs):5d}{len(rs)/len(idx)*100:7.1f}%")
    k = "F1 = 0, reference absent from the context (retrieval error)"
    print(f"retrieval errors without the second hop: {len(e1[k])}")


def main():
    for name, (fname, idx, qfile, types) in SETS.items():
        c = G.load(os.path.join(RES, fname))
        Q = R.load_json(qfile)
        print("\n" + "=" * 100 + f"\n{name}")
        by_type(name, c, idx, Q, types)
        ablation(name, c, idx)
        rejections(name, c, idx)
        errors(name, c, idx)


if __name__ == "__main__":
    main()
