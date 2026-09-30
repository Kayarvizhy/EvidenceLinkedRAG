"""
prepare_musique.py  --  build the MuSiQue retrieval corpus

    python prepare_musique.py

Input   data/musique_dev.parquet   validation split of MuSiQue-Ans (Trivedi et al., 2022), from the
                                   Hugging Face dataset dgslibisey/MuSiQue
Output  cache/mq_questions.json, cache/mq_paragraphs.json, cache/mq_sentences.json

Same construction as for HotpotQA (prepare.py) and 2WikiMultihopQA (prepare_2wiki.py): the corpus is
the pool of all context paragraphs of the split, without duplicates, and it is searched as a whole.
Two differences follow from the data.  (1) MuSiQue paragraphs are arbitrary paragraphs of Wikipedia
articles, so the same title occurs with several different texts; gold evidence is therefore recorded
as paragraph ids (gold_pids) as well as titles.  (2) The file is ordered by hop count, so the
questions are shuffled once with a fixed seed before the first 1,000 are used.  Sentences are split
with the same rule as in prepare.py.  No setting of the method is tuned on this dataset.
"""
import os, re, json, collections
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
clean = lambda t: re.sub(r"\s+", " ", t).strip()


def split_sentences(text):
    return [p for p in re.split(r"(?<=[.!?])\s+", text.strip()) if p]


def main():
    df = pd.read_parquet(os.path.join(HERE, "data", "musique_dev.parquet"))
    order = np.random.default_rng(0).permutation(len(df))
    df = df.iloc[order].reset_index(drop=True)
    questions, paragraphs, sentences, par_id, n_raw = [], [], [], {}, 0
    for qi, row in enumerate(df.itertuples(index=False)):
        gold = []
        for p in row.paragraphs:
            n_raw += 1
            title, text = clean(p["title"]), clean(p["paragraph_text"])
            if not text:
                continue
            if (title, text) not in par_id:
                pid = par_id[(title, text)] = len(paragraphs)
                paragraphs.append({"pid": pid, "title": title, "text": text, "tokens": len(text.split())})
                for si, s in enumerate(split_sentences(text)):
                    s = clean(s)
                    if s:
                        sentences.append({"sid": len(sentences), "pid": pid, "title": title, "sent_idx": si, "text": s})
            if p["is_supporting"]:
                gold.append(par_id[(title, text)])
        hops = row.id.split("__")[0]
        questions.append({"qidx": qi, "id": row.id, "question": row.question, "answer": row.answer,
                          "answer_aliases": list(row.answer_aliases), "type": hops[:4],
                          "gold_titles": sorted({paragraphs[g]["title"] for g in gold}), "gold_pids": sorted(set(gold))})
    for name, obj in (("mq_questions.json", questions), ("mq_paragraphs.json", paragraphs), ("mq_sentences.json", sentences)):
        json.dump(obj, open(os.path.join(CACHE, name), "w", encoding="utf8"))
    tok = sorted(p["tokens"] for p in paragraphs)
    print(f"questions {len(questions)}; paragraphs {n_raw} with duplicates, {len(paragraphs)} unique; sentences {len(sentences)}")
    print(f"paragraph tokens: median {tok[len(tok)//2]}, mean {sum(tok)/len(tok):.0f}")
    print("gold paragraphs per question:", dict(collections.Counter(len(q["gold_pids"]) for q in questions)))
    print("types, first 1000:", dict(collections.Counter(q["type"] for q in questions[:1000])))
    dup = sum(1 for q in questions if len(q["gold_titles"]) < len(q["gold_pids"]))
    print(f"questions whose gold paragraphs share a title: {dup}")


if __name__ == "__main__":
    main()
