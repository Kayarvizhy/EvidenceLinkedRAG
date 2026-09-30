"""
prepare_wikihow.py  --  build the WikiHow corpus and questions

    python prepare_wikihow.py

Input   data/wikihowAll_head.csv   first part of wikihowAll.csv (Koupaee and Wang, 2018).
        As in convert_wikihow.py of the AdaptiveRAG repo, the first 10,000 rows are used.

Output  cache/wh_questions.json   "How to ...?" question built from the title; reference = article summary
        cache/wh_passages.json    passages of 250-400 tokens (the repo's G2 rule) with title and article id
        cache/wh_sentences.json   sentences with a pointer to their parent passage

Difference to the repo: the repo indexes summary + text, so the reference answer itself is part of the
corpus.  Here only the article text is indexed, so an answer has to be produced from the article body.
"""
import os, re, json
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")


def clean(t):
    return re.sub(r"\s+", " ", t).strip() if isinstance(t, str) else ""


def split_sentences(text):
    return [p for p in re.split(r"(?<=[.!?])\s+", text.strip()) if p]


def chunk_by_tokens(sentences, min_tok, max_tok):          # unchanged from process_wikihow.py
    chunks, buf, buf_len = [], [], 0
    for s in sentences:
        s_len = len(s.split())
        if buf_len + s_len > max_tok and buf:
            chunks.append(buf)
            buf, buf_len = [], 0
        buf.append(s)
        buf_len += s_len
        if buf_len >= min_tok:
            chunks.append(buf)
            buf, buf_len = [], 0
    if buf:
        chunks.append(buf)
    return chunks


def main():
    df = pd.read_csv(os.path.join(HERE, "data", "wikihowAll_head.csv"), nrows=10000)
    questions, passages, sentences = [], [], []
    seen_q = set()
    for aid, row in enumerate(df.itertuples(index=False)):
        title, summary, text = clean(row.title), clean(row.headline), clean(row.text)
        summary = re.sub(r"\s*\.,\s*", ". ", summary).strip(" ,")       # step headlines are joined by ".,"
        if not title or title == "nan" or not text:
            continue
        name = re.sub(r"\s*\d+$", "", title).strip()                    # "How to Be an Organized Artist1"
        first = len(passages)
        for ch in chunk_by_tokens(split_sentences(text), 250, 400):
            pid = len(passages)
            body = " ".join(ch)
            passages.append({"pid": pid, "aid": aid, "title": name, "text": body, "tokens": len(body.split())})
            for s in ch:
                sentences.append({"sid": len(sentences), "pid": pid, "title": name, "text": s})
        if len(summary.split()) < 10 or len(passages) == first:
            continue
        q = name if name.lower().startswith("how to") else "How to " + name[0].lower() + name[1:]
        q = q[0].upper() + q[1:].rstrip("?") + "?"
        if q.lower() in seen_q:                                          # several articles share a title
            continue
        seen_q.add(q.lower())
        questions.append({"qidx": len(questions), "aid": aid, "question": q, "answer": summary,
                          "gold_pids": list(range(first, len(passages)))})
    for name, obj in (("wh_questions.json", questions), ("wh_passages.json", passages), ("wh_sentences.json", sentences)):
        json.dump(obj, open(os.path.join(CACHE, name), "w", encoding="utf8"))
    tok = sorted(p["tokens"] for p in passages)
    print(f"articles with a question: {len(questions)}; passages: {len(passages)}; sentences: {len(sentences)}")
    print(f"passage tokens: median {tok[len(tok)//2]}, mean {sum(tok)/len(tok):.0f}")
    ref = sorted(len(q["answer"].split()) for q in questions)
    print(f"reference summary tokens: median {ref[len(ref)//2]}, mean {sum(ref)/len(ref):.0f}")
    print(f"passages per article: mean {len(passages)/len(set(p['aid'] for p in passages)):.1f}")
    print("first questions:", [q["question"] for q in questions[:3]])


if __name__ == "__main__":
    main()
