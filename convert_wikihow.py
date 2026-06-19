import pandas as pd, json

IN_PATH  = r"C:\Users\bsshr\OneDrive\Desktop\AdaptiveRAG\corpus\wikihow\raw\wikihow.csv"
OUT_PATH = r"C:\Users\bsshr\OneDrive\Desktop\AdaptiveRAG\corpus\wikihow\raw\wikihow.json"

print("Reading 10000 rows...")
df = pd.read_csv(IN_PATH, nrows=10000)
records = [{"title": str(r.get("title","")), "summary": str(r.get("headline","")), "text": str(r.get("text",""))} for _, r in df.iterrows()]

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(records, f, indent=2)

print(f"Done: {len(records)} articles")