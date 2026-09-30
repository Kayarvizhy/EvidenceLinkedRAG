"""
download_data.py  --  fetch the validation splits of the three multi-hop datasets

    python download_data.py

Writes   data/hotpot_dev_distractor.parquet   HotpotQA, distractor setting, validation (7,405 questions)
         data/2wiki_dev.parquet               2WikiMultihopQA, validation (12,576 questions)
         data/musique_dev.parquet             MuSiQue-Ans, validation (2,417 questions)

The files are taken from the Hugging Face Hub with the `datasets` library.  These are the sources
that the results in results/ were produced with.

WikiHow is not on the Hub in the form that is used here.  Download wikihowAll.csv from the link given
by Koupaee and Wang (2018), https://ucsb.box.com/s/7yq601ijl1lzvlfu4rjdbbxforzd2oag, and save it as
data/wikihowAll_head.csv.  Only the first 10,000 rows are read, so the complete file works as well.
"""
import os
from datasets import load_dataset

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCES = [("hotpot_dev_distractor.parquet", "hotpotqa/hotpot_qa", "distractor", 7405),
           ("2wiki_dev.parquet", "framolfese/2WikiMultihopQA", None, 12576),
           ("musique_dev.parquet", "dgslibisey/MuSiQue", None, 2417)]


def main():
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    for name, repo, config, n in SOURCES:
        path = os.path.join(HERE, "data", name)
        if os.path.exists(path):
            print(f"{name}: already there")
            continue
        ds = load_dataset(repo, config, split="validation") if config else load_dataset(repo, split="validation")
        assert len(ds) == n, f"{repo}: expected {n} questions, found {len(ds)}"
        ds.to_parquet(path)
        print(f"{name}: {len(ds)} questions from {repo}")
    if not os.path.exists(os.path.join(HERE, "data", "wikihowAll_head.csv")):
        print("WikiHow: download wikihowAll.csv by hand and save it as data/wikihowAll_head.csv (see the docstring).")


if __name__ == "__main__":
    main()
