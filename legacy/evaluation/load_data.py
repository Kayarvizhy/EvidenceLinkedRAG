import json, os, sys

BASE = os.path.expanduser(
    "~/Desktop/AdaptiveRAG")

def load_hotpotqa_samples(n: int = 10) -> list:
    path = (f"{BASE}/corpus/hotpotqa/raw/"
            f"hotpot_dev_distractor_v1.json")
    data = json.load(open(path))
    samples = []
    for item in data[:n]:
        samples.append({
            "query" : item["question"],
            "answer": item["answer"],
            "type"  : item.get("type", "factual"),
            "level" : item.get("level", "medium"),
        })
    return samples

def load_wikihow_samples(n: int = 10) -> list:
    # ── try real large file first ──────────────
    real_path = (f"{BASE}/corpus/wikihow/raw/"
                 f"wikihow.json")
    qa_path   = (f"{BASE}/corpus/wikihow/raw/"
                 f"wikihow_qa.json")

    if os.path.exists(real_path):
        data = json.load(open(real_path))
        samples = []
        for item in data:
            title   = item.get("title", "").strip()
            summary = item.get("summary", "").strip()

            # skip if too short or empty
            if not title or not summary:
                continue
            if len(summary.split()) < 10:
                continue

            # convert title → How to query
            title_clean = title.lower().strip()
            # remove trailing numbers like "1", "2"
            import re
            title_clean = re.sub(
                r'\s+\d+$', '', title_clean).strip()

            if not title_clean.startswith("how to"):
                title_clean = "How to " + title_clean
            else:
                title_clean = title_clean.capitalize()

            samples.append({
                "query" : title_clean + "?",
                "answer": summary,
                "type"  : "procedural",
                "level" : "medium",
            })

            if len(samples) >= n:
                break

        print(f"[INFO] Loaded {len(samples)} "
              f"real WikiHow samples")
        return samples

    # ── fallback to small qa file ──────────────
    data = json.load(open(qa_path))
    samples = []
    for item in data[:n]:
        samples.append({
            "query" : item["query"],
            "answer": item["answer"],
            "type"  : "procedural",
            "level" : "medium",
        })
    return samples
