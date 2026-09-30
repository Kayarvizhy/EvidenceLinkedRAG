"""
prepare.py  --  build the HotpotQA retrieval corpus for the pilot
=================================================================

    python prepare.py

Input   data/hotpot_dev_distractor.parquet   (HotpotQA distractor, validation split,
        the same source that download_hotpotqa.py in the AdaptiveRAG repo uses)

Output  cache/questions.json    question, answer, type, level, gold supporting titles
        cache/paragraphs.json   unique paragraphs (title + text), one entry per title/text pair
        cache/sentences.json    every sentence with a pointer to its parent paragraph
        cache/repo_chunks.json  G1/G2/G3 chunk statistics produced by the repo's own chunker

The corpus is the pool of all context paragraphs of all 7,405 validation questions,
as in process_hotpotqa.py.  The repo keeps one copy of a paragraph for every question
it appears in; here duplicates are removed so that top-k never holds the same text twice.
"""
import os, re, json, collections
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "data", "hotpot_dev_distractor.parquet")
CACHE = os.path.join(HERE, "cache")
os.makedirs(CACHE, exist_ok=True)


def clean(text):
    return re.sub(r"\s+", " ", text).strip()


# ── the repo's chunker, copied unchanged from process_hotpotqa.py ────────────
def chunk_by_tokens(sentences, min_tok, max_tok):
    chunks, buf = [], []
    buf_len = 0
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


def split_sentences(text):
    return [p for p in re.split(r"(?<=[.!?])\s+", text.strip()) if p]


def main():
    df = pd.read_parquet(RAW)
    print(f"questions in split: {len(df)}")

    questions, paragraphs, sentences = [], [], []
    par_id = {}                                   # (title, text) -> index in paragraphs
    n_raw = 0
    for qi, row in enumerate(df.itertuples(index=False)):
        titles = list(row.context["title"])
        sents = [list(s) for s in row.context["sentences"]]
        for title, ss in zip(titles, sents):
            n_raw += 1
            text = clean(" ".join(ss))
            key = (title, text)
            if key in par_id or not text:
                continue
            pid = len(paragraphs)
            par_id[key] = pid
            paragraphs.append({"pid": pid, "title": title, "text": text,
                               "tokens": len(text.split())})
            for si, s in enumerate(ss):
                s = clean(s)
                if s:
                    sentences.append({"sid": len(sentences), "pid": pid, "title": title,
                                      "sent_idx": si, "text": s})
        gold = sorted(set(row.supporting_facts["title"]))
        questions.append({"qidx": qi, "id": row.id, "question": row.question, "answer": row.answer,
                          "type": row.type, "level": row.level, "gold_titles": gold})

    json.dump(questions, open(os.path.join(CACHE, "questions.json"), "w", encoding="utf8"))
    json.dump(paragraphs, open(os.path.join(CACHE, "paragraphs.json"), "w", encoding="utf8"))
    json.dump(sentences, open(os.path.join(CACHE, "sentences.json"), "w", encoding="utf8"))

    tok = [p["tokens"] for p in paragraphs]
    tok.sort()
    print(f"paragraphs: {n_raw} in the repo corpus (with duplicates), {len(paragraphs)} unique")
    print(f"sentences : {len(sentences)}")
    print(f"paragraph length in tokens: median {tok[len(tok)//2]}, mean {sum(tok)/len(tok):.0f}, "
          f"90th pct {tok[int(0.9*len(tok))]}, max {tok[-1]}")
    print(f"gold paragraphs per question: {collections.Counter(len(q['gold_titles']) for q in questions)}")

    # ── how different are the repo's three granularities on this corpus? ────
    stats = {}
    chunks = {}
    for g, lo, hi in (("G1", 100, 150), ("G2", 250, 400), ("G3", 600, 900)):
        n, whole = 0, 0
        per_par = []
        for p in paragraphs:
            cs = chunk_by_tokens(split_sentences(p["text"]), lo, hi)
            per_par.append(cs)
            n += len(cs)
            whole += (len(cs) == 1)
        chunks[g] = per_par
        stats[g] = {"chunks": n, "paragraphs_kept_whole": whole,
                    "share_kept_whole": round(whole / len(paragraphs), 4)}
        print(f"{g}: {n} chunks; {whole/len(paragraphs)*100:.1f}% of paragraphs are a single chunk")
    same12 = sum(a == b for a, b in zip(chunks["G1"], chunks["G2"])) / len(paragraphs)
    same23 = sum(a == b for a, b in zip(chunks["G2"], chunks["G3"])) / len(paragraphs)
    same13 = sum(a == b for a, b in zip(chunks["G1"], chunks["G3"])) / len(paragraphs)
    print(f"paragraphs with identical chunks: G1=G2 {same12*100:.1f}%, G2=G3 {same23*100:.1f}%, G1=G3 {same13*100:.1f}%")
    stats["identical"] = {"G1=G2": round(same12, 4), "G2=G3": round(same23, 4), "G1=G3": round(same13, 4)}
    json.dump(stats, open(os.path.join(CACHE, "repo_chunks.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
