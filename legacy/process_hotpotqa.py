# process_hotpotqa.py
import json, os, re
from tqdm import tqdm

RAW      = os.path.expanduser("~/OneDrive/Desktop/AdaptiveRAG/corpus/hotpotqa/raw/hotpot_dev_distractor_v1.json")
PROC     = os.path.expanduser("~/OneDrive/Desktop/AdaptiveRAG/corpus/hotpotqa/processed/hotpotqa_docs.json")
G1_OUT   = os.path.expanduser("~/OneDrive/Desktop/AdaptiveRAG/corpus/hotpotqa/g1_sentence")
G2_OUT   = os.path.expanduser("~/OneDrive/Desktop/AdaptiveRAG/corpus/hotpotqa/g2_paragraph")
G3_OUT   = os.path.expanduser("~/OneDrive/Desktop/AdaptiveRAG/corpus/hotpotqa/g3_section")

os.makedirs(os.path.dirname(PROC), exist_ok=True)
os.makedirs(G1_OUT, exist_ok=True)
os.makedirs(G2_OUT, exist_ok=True)
os.makedirs(G3_OUT, exist_ok=True)

# ── helpers ──────────────────────────────────────────────────────────────────
def clean(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def token_len(text):
    return len(text.split())

def chunk_by_tokens(sentences, min_tok, max_tok):
    chunks, buf = [], []
    buf_len = 0
    for s in sentences:
        s_len = token_len(s)
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
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p for p in parts if p]

# ── load raw ─────────────────────────────────────────────────────────────────
print("[INFO] Loading HotpotQA...")
raw = json.load(open(RAW))

# ── build source docs ─────────────────────────────────────────────────────────
print("[INFO] Building source documents...")
docs = []
for item in tqdm(raw):
    qid = item["id"]
    titles    = item["context"]["title"]
    sentences = item["context"]["sentences"]
    for title, sents in zip(titles, sentences):
        full_text = clean(" ".join(sents))
        docs.append({
            "doc_id"      : f"{qid}_{title}",
            "title"       : title,
            "text"        : full_text,
            "source"      : "hotpotqa",
            "type"        : "reasoning",
            "question_id" : qid,
            "answer"      : item["answer"],
            "question"    : item["question"]
        })

json.dump(docs, open(PROC, "w"), indent=2)
print(f"✅ Processed docs   : {len(docs)}")

# ── G1 — sentence level (100–150 tokens) ─────────────────────────────────────
print("[INFO] Creating G1 chunks (sentence-level 100-150 tokens)...")
g1_chunks = []
for doc in tqdm(docs):
    sents  = split_sentences(doc["text"])
    chunks = chunk_by_tokens(sents, min_tok=100, max_tok=150)
    for i, chunk in enumerate(chunks):
        g1_chunks.append({
            "chunk_id"    : f"{doc['doc_id']}_g1_{i}",
            "text"        : chunk,
            "granularity" : "G1",
            "source"      : "hotpotqa",
            "type"        : "reasoning",
            "doc_id"      : doc["doc_id"],
            "title"       : doc["title"],
            "token_count" : token_len(chunk)
        })

g1_path = os.path.join(G1_OUT, "hotpotqa_g1.json")
json.dump(g1_chunks, open(g1_path, "w"), indent=2)
print(f"✅ G1 chunks        : {len(g1_chunks)}")

# ── G2 — paragraph level (250–400 tokens) ────────────────────────────────────
print("[INFO] Creating G2 chunks (paragraph-level 250-400 tokens)...")
g2_chunks = []
for doc in tqdm(docs):
    sents  = split_sentences(doc["text"])
    chunks = chunk_by_tokens(sents, min_tok=250, max_tok=400)
    for i, chunk in enumerate(chunks):
        g2_chunks.append({
            "chunk_id"    : f"{doc['doc_id']}_g2_{i}",
            "text"        : chunk,
            "granularity" : "G2",
            "source"      : "hotpotqa",
            "type"        : "reasoning",
            "doc_id"      : doc["doc_id"],
            "title"       : doc["title"],
            "token_count" : token_len(chunk)
        })

g2_path = os.path.join(G2_OUT, "hotpotqa_g2.json")
json.dump(g2_chunks, open(g2_path, "w"), indent=2)
print(f"✅ G2 chunks        : {len(g2_chunks)}")

# ── G3 — section level (600–900 tokens) ──────────────────────────────────────
print("[INFO] Creating G3 chunks (section-level 600-900 tokens)...")
g3_chunks = []
for doc in tqdm(docs):
    sents  = split_sentences(doc["text"])
    chunks = chunk_by_tokens(sents, min_tok=600, max_tok=900)
    for i, chunk in enumerate(chunks):
        g3_chunks.append({
            "chunk_id"    : f"{doc['doc_id']}_g3_{i}",
            "text"        : chunk,
            "granularity" : "G3",
            "source"      : "hotpotqa",
            "type"        : "reasoning",
            "doc_id"      : doc["doc_id"],
            "title"       : doc["title"],
            "token_count" : token_len(chunk)
        })

g3_path = os.path.join(G3_OUT, "hotpotqa_g3.json")
json.dump(g3_chunks, open(g3_path, "w"), indent=2)
print(f"✅ G3 chunks        : {len(g3_chunks)}")

print("\n✅ HotpotQA processing complete!")
print(f"   G1 (sentence)   : {len(g1_chunks)} chunks")
print(f"   G2 (paragraph)  : {len(g2_chunks)} chunks")
print(f"   G3 (section)    : {len(g3_chunks)} chunks")
