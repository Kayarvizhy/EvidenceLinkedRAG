from datasets import load_dataset
import json

ds = load_dataset("hotpot_qa", "distractor", split="validation")
data = [dict(item) for item in ds]
with open(r"C:\Users\bsshr\OneDrive\Desktop\AdaptiveRAG\corpus\hotpotqa\raw\hotpot_dev_distractor_v1.json", "w") as f:
    json.dump(data, f)

