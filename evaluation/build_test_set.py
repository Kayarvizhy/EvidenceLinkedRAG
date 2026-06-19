import os, sys, json, re

BASE_DIR = os.path.expanduser("~/Desktop/AdaptiveRAG")
DATA_DIR = os.path.join(BASE_DIR,
                        "evaluation/test_sets")
os.makedirs(DATA_DIR, exist_ok=True)

def sample_chunks(source, n=100):
    path = (f"{BASE_DIR}/corpus/{source}/"
            f"g2_paragraph/{source}_g2.json")
    data    = json.load(open(path))
    samples = []
    seen    = set()

    for item in data:
        text = item.get("text","").strip()
        if not text or len(text.split()) < 10:
            continue
        sents = re.split(r'(?<=[.!?])\s+', text)
        if len(sents) < 2:
            continue
        q    = sents[0].strip()
        gold = sents[1].strip()
        if len(gold.split()) < 3:
            continue
        if q in seen:
            continue
        seen.add(q)
        samples.append({
            "query" : q,
            "answer": gold,
            "source": source
        })
        if len(samples) >= n:
            break
    return samples

print("Sampling HotpotQA...")
hp = sample_chunks("hotpotqa", 100)

print("Sampling WikiHow...")
wh = sample_chunks("wikihow",  100)

json.dump(hp,
          open(f"{DATA_DIR}/hotpotqa_test100.json",
               "w"), indent=2)
json.dump(wh,
          open(f"{DATA_DIR}/wikihow_test100.json",
               "w"), indent=2)

print(f"HotpotQA : {len(hp)} samples ✅")
print(f"WikiHow  : {len(wh)} samples ✅")
print(f"Saved to : {DATA_DIR}")
