"""
prepare_2wiki.py  --  build the 2WikiMultihopQA retrieval corpus

    python prepare_2wiki.py

Input   data/2wiki_dev.parquet   validation split of 2WikiMultihopQA (Ho et al., 2020)
Output  cache/w2_questions.json, cache/w2_paragraphs.json, cache/w2_sentences.json

Same construction as for HotpotQA (prepare.py): the corpus is the pool of all context paragraphs of
the split, without duplicates, and it is searched as a whole.  No setting of the method is tuned on
this dataset; it is used for testing only.
"""
import os, re, json, collections
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
clean = lambda t: re.sub(r"\s+", " ", t).strip()


def main():
    df = pd.read_parquet(os.path.join(HERE, "data", "2wiki_dev.parquet"))
    questions, paragraphs, sentences, par_id, n_raw = [], [], [], {}, 0
    for qi, row in enumerate(df.itertuples(index=False)):
        for title, ss in zip(list(row.context["title"]), [list(s) for s in row.context["sentences"]]):
            n_raw += 1
            text = clean(" ".join(ss))
            if not text or (title, text) in par_id:
                continue
            pid = par_id[(title, text)] = len(paragraphs)
            paragraphs.append({"pid": pid, "title": title, "text": text, "tokens": len(text.split())})
            for si, s in enumerate(ss):
                s = clean(s)
                if s:
                    sentences.append({"sid": len(sentences), "pid": pid, "title": title, "sent_idx": si, "text": s})
        questions.append({"qidx": qi, "id": row.id, "question": row.question, "answer": row.answer, "type": row.type,
                          "gold_titles": sorted(set(row.supporting_facts["title"]))})
    for name, obj in (("w2_questions.json", questions), ("w2_paragraphs.json", paragraphs), ("w2_sentences.json", sentences)):
        json.dump(obj, open(os.path.join(CACHE, name), "w", encoding="utf8"))
    tok = sorted(p["tokens"] for p in paragraphs)
    print(f"questions {len(questions)}; paragraphs {n_raw} with duplicates, {len(paragraphs)} unique; sentences {len(sentences)}")
    print(f"paragraph tokens: median {tok[len(tok)//2]}, mean {sum(tok)/len(tok):.0f}")
    print("gold paragraphs per question:", dict(collections.Counter(len(q["gold_titles"]) for q in questions)))
    print("types, first 1000:", dict(collections.Counter(q["type"] for q in questions[:1000])))


if __name__ == "__main__":
    main()
