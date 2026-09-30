# process_wikihow.py
import json, os, re
from tqdm import tqdm

RAW    = os.path.expanduser("~/OneDrive/Desktop/AdaptiveRAG/corpus/wikihow/raw/wikihow.json")
PROC   = os.path.expanduser("~/OneDrive/Desktop/AdaptiveRAG/corpus/wikihow/processed/wikihow_docs.json")
G1_OUT = os.path.expanduser("~/OneDrive/Desktop/AdaptiveRAG/corpus/wikihow/g1_sentence")
G2_OUT = os.path.expanduser("~/OneDrive/Desktop/AdaptiveRAG/corpus/wikihow/g2_paragraph")
G3_OUT = os.path.expanduser("~/OneDrive/Desktop/AdaptiveRAG/corpus/wikihow/g3_section")
os.makedirs(os.path.dirname(PROC), exist_ok=True)

os.makedirs(G1_OUT, exist_ok=True)
os.makedirs(G2_OUT, exist_ok=True)
os.makedirs(G3_OUT, exist_ok=True)
# ── helpers ───────────────────────────────────────────────────────────────────
def clean(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def token_len(text):
    return len(text.split())

def split_sentences(text):
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p for p in parts if p]

def chunk_by_tokens(sentences, min_tok, max_tok):
    chunks, buf, buf_len = [], [], 0
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

# ── load raw ──────────────────────────────────────────────────────────────────
print("[INFO] Loading WikiHow...")
raw = json.load(open(RAW))
print(f"   Total articles  : {len(raw)}")

# ── build source docs ─────────────────────────────────────────────────────────
# WikiHow keys: summary, title, text
# As per document:
#   type        = procedural
#   source      = wikihow
#   intent_hint = procedural
# G1 → individual steps
# G2 → grouped steps (procedural reasoning)
# G3 → whole procedure

print("[INFO] Building source documents...")
docs = []
for i, item in enumerate(tqdm(raw)):
    title   = clean(str(item.get("title",   "")))
    summary = clean(str(item.get("summary", "")))
    text    = clean(str(item.get("text",    "")))

    # combine summary + text as full article
    full_text = (summary + " " + text).strip()
    if not full_text:
        continue

    docs.append({
        "doc_id"      : f"wikihow_{i}",
        "title"       : title,
        "summary"     : summary,
        "text"        : full_text,
        "source"      : "wikihow",
        "type"        : "procedural",
        "intent_hint" : "procedural"
    })

json.dump(docs, open(PROC, "w"), indent=2)
print(f"✅ Processed docs   : {len(docs)}")

# ── G1 — individual steps / sentence-level (100–150 tokens) ──────────────────
print("[INFO] Creating G1 chunks — individual steps (100-150 tokens)...")
g1_chunks = []
for doc in tqdm(docs):
    sents  = split_sentences(doc["text"])
    chunks = chunk_by_tokens(sents, min_tok=100, max_tok=150)
    for i, chunk in enumerate(chunks):
        g1_chunks.append({
            "chunk_id"    : f"{doc['doc_id']}_g1_{i}",
            "text"        : chunk,
            "granularity" : "G1",
            "source"      : "wikihow",
            "type"        : "procedural",
            "intent_hint" : "procedural",
            "doc_id"      : doc["doc_id"],
            "title"       : doc["title"],
            "token_count" : token_len(chunk)
        })

g1_path = os.path.join(G1_OUT, "wikihow_g1.json")
json.dump(g1_chunks, open(g1_path, "w"), indent=2)
print(f"✅ G1 chunks        : {len(g1_chunks)}")

# ── G2 — grouped steps / paragraph-level (250–400 tokens) ────────────────────
print("[INFO] Creating G2 chunks — grouped steps (250-400 tokens)...")
g2_chunks = []
for doc in tqdm(docs):
    sents  = split_sentences(doc["text"])
    chunks = chunk_by_tokens(sents, min_tok=250, max_tok=400)
    for i, chunk in enumerate(chunks):
        g2_chunks.append({
            "chunk_id"    : f"{doc['doc_id']}_g2_{i}",
            "text"        : chunk,
            "granularity" : "G2",
            "source"      : "wikihow",
            "type"        : "procedural",
            "intent_hint" : "procedural",
            "doc_id"      : doc["doc_id"],
            "title"       : doc["title"],
            "token_count" : token_len(chunk)
        })

g2_path = os.path.join(G2_OUT, "wikihow_g2.json")
json.dump(g2_chunks, open(g2_path, "w"), indent=2)
print(f"✅ G2 chunks        : {len(g2_chunks)}")

# ── G3 — whole procedure / section-level (600–900 tokens) ────────────────────
print("[INFO] Creating G3 chunks — whole procedure (600-900 tokens)...")
g3_chunks = []
for doc in tqdm(docs):
    sents  = split_sentences(doc["text"])
    chunks = chunk_by_tokens(sents, min_tok=600, max_tok=900)
    for i, chunk in enumerate(chunks):
        g3_chunks.append({
            "chunk_id"    : f"{doc['doc_id']}_g3_{i}",
            "text"        : chunk,
            "granularity" : "G3",
            "source"      : "wikihow",
            "type"        : "procedural",
            "intent_hint" : "procedural",
            "doc_id"      : doc["doc_id"],
            "title"       : doc["title"],
            "token_count" : token_len(chunk)
        })

g3_path = os.path.join(G3_OUT, "wikihow_g3.json")
json.dump(g3_chunks, open(g3_path, "w"), indent=2)
print(f"✅ G3 chunks        : {len(g3_chunks)}")

print("\n✅ WikiHow processing complete!")
print(f"   G1 (individual steps)  : {len(g1_chunks)} chunks")
print(f"   G2 (grouped steps)     : {len(g2_chunks)} chunks")
print(f"   G3 (whole procedure)   : {len(g3_chunks)} chunks")