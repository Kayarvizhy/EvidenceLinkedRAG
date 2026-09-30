# Evidence-Linked Multi-Hop Retrieval with Answer Verification for Retrieval-Augmented Generation

Code, evaluation scripts and per-question results for the paper of the same name.

The pipeline is training-free. It fuses sparse and dense retrieval over paragraphs and sentences, reranks the candidates with a cross-encoder, adds a second hop that follows the document titles mentioned in the top-ranked paragraphs and scores each linked paragraph together with the paragraph that leads to it, and verifies the generated answer. Everything runs on a laptop with a 4 GB graphics processor and locally served generators.

## Main results

Share of questions with all supporting paragraphs in a context of five paragraphs, on 1,000 held-out questions per dataset:

| System | HotpotQA | 2WikiMultihopQA | MuSiQue |
|---|---|---|---|
| Fixed RAG (dense, k = 5) | 35.1 | 16.1 | 9.0 |
| RAG + Reranker (strong) | 54.3 | 27.2 | 18.0 |
| Proposed, first stage only | 65.0 | 33.6 | 19.8 |
| BRIDGE-style second hop (Personalized PageRank) | 65.9 | 34.6 | 20.1 |
| Proposed | 82.2 | 52.4 | 25.3 |

Answer F1 with the same generator, prompts and context size, on 500 questions per dataset:

| System | HotpotQA | 2WikiMultihopQA |
|---|---|---|
| RAG + Reranker (strong) | 51.9 | 30.8 |
| LLM-driven second hop (two generator calls) | 60.0 | 37.0 |
| Proposed | 63.9 | 42.0 |

All settings were chosen on HotpotQA tuning questions. 2WikiMultihopQA and MuSiQue were used for testing only. MuSiQue is evaluated for retrieval only; answers were not generated on it.

## Repository layout

| Path | Content |
|---|---|
| `download_data.py` | Fetches the validation splits of HotpotQA, 2WikiMultihopQA and MuSiQue |
| `prepare.py`, `prepare_2wiki.py`, `prepare_musique.py`, `prepare_wikihow.py` | Build the retrieval corpora (paragraphs, sentences, questions) in `cache/` |
| `retrieval_lib.py` | BM25, dense encoder, cross-encoder, reciprocal rank fusion, paired bootstrap |
| `retrieval_pilot.py` | Retrieval evaluation of all systems, the BRIDGE-style baseline and the retrieval ablation |
| `systems.py` | The systems of the generation experiment, including the proposed pipeline and the baselines |
| `generate_eval.py` | Generation with Ollama, answer verification, answer metrics and result tables |
| `metrics_eval.py` | Exact match, F1, ROUGE-L, evidence overlap and unsupported-token rate |
| `latency_bench.py` | End-to-end latency, one question at a time |
| `paper_tables.py` | Tables by question type, verification and error analysis, from the result files |
| `tune_llm_hop.py`, `explore.py`, `explore2.py` | Design choices, made on tuning questions only |
| `make_figure.py` | Architecture figure |
| `repo_query_understanding.py` | Keyword-based intent detection, unchanged from the first version of this work |
| `results/` | Per-question results and printed tables (see below) |
| `legacy/` | Code and results of the earlier version of the method; not used by the paper |

## Setup

Python 3.10 and the packages in `requirements.txt`:

```
pip install -r requirements.txt
```

The generators are served by [Ollama](https://ollama.com). The experiments used `llama3` (Llama-3-8B-Instruct, Q4_0) for factual and analytical questions and `mistral` (Mistral-7B-Instruct, Q4_K_M) for procedural questions:

```
ollama pull llama3
ollama pull mistral
```

The encoder (`sentence-transformers/all-MiniLM-L6-v2`) and the cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) are downloaded on first use into `models/`.

## Reproducing the results

1. Data and corpora. The first three datasets are downloaded by the script; WikiHow has to be downloaded by hand as described in `download_data.py`.

   ```
   python download_data.py
   python prepare.py
   python prepare_2wiki.py
   python prepare_musique.py
   python prepare_wikihow.py
   ```

2. Retrieval (no generator needed). The first run of each dataset computes the embeddings.

   ```
   python retrieval_pilot.py --split test                    > results/retrieval_test.txt
   python retrieval_pilot.py --dataset wiki2   --split test  > results/retrieval_wiki2_test.txt
   python retrieval_pilot.py --dataset musique --split test  > results/retrieval_musique_test.txt
   ```

3. Generation. Results are appended to `results/gen_<dataset>_<split>.jsonl`, and finished questions are never run again, so a run can be stopped and restarted. To generate from scratch, move the existing `.jsonl` files away first.

   ```
   python generate_eval.py --dataset hotpot --split test --n 500 --ref full_v --systems bm25,fixed,rerank,strong,uamg_v1,hop1,llm_hop,full,full_v,full_vh,full_vr,strong_v,fixed_v
   python generate_eval.py --dataset wiki2  --split test --n 500 --ref full_v --systems bm25,fixed,rerank,strong,uamg_v1,hop1,llm_hop,full,full_v
   python generate_eval.py --dataset wikihow --split test --n 100
   ```

   Adding `--tables` prints the tables from the stored results without calling the generator.

4. Latency and the remaining tables.

   ```
   python latency_bench.py --n 100
   python paper_tables.py
   ```

Decoding is deterministic (temperature 0, fixed seed), and the retrieval is deterministic, so a repeated run reproduces the stored results. On the hardware below, the retrieval evaluation of one dataset takes 5 to 10 minutes once the embeddings exist, one generator call takes about 3 to 4 seconds, and the latency benchmark takes about two hours.

## System names

The result files use short keys. The paper uses the names in the second column.

| Key | Name in the paper |
|---|---|
| `bm25` | BM25 + LLM |
| `fixed` | Fixed RAG (k = 5) |
| `rerank` | RAG + Reranker |
| `strong` | RAG + Reranker (strong) |
| `uamg_v1` | Heuristic Adaptive RAG (the earlier version of the method, see `legacy/`) |
| `llm_hop` | LLM-Driven Second Hop |
| `hop1` | Proposed without second hop |
| `full` | Proposed without verification |
| `full_v` | Proposed (ours) |
| `full_vh`, `full_vr` | Verification with the hint only, with the wider context only |
| `strong_v`, `fixed_v` | Verification applied to the strong reranking baseline and to Fixed RAG |

In the retrieval result files the systems carry descriptive names. "PROPOSED full" is the proposed retrieval, "PROPOSED hop 1" is the first stage alone, and "BRIDGE-style" is the second hop of BRIDGE (Yazidi et al., IEEE Access, 2026) applied to the first stage of the proposed pipeline with its published settings.

## Where each table of the paper comes from

| Table | Script | Stored output |
|---|---|---|
| 6, evidence retrieval | `retrieval_pilot.py` | `results/retrieval_test.txt`, `retrieval_wiki2_test.txt`, `retrieval_musique_test.txt` and the `.json` files |
| 7, answer quality | `generate_eval.py --tables` | `results/tables_*_test.txt`, `results/gen_*_test.jsonl` |
| 8, by question type | `paper_tables.py` | `results/paper_tables.txt` |
| 9, retrieval ablation | `retrieval_pilot.py` | same files as Table 6 |
| 10, second hop and verification | `paper_tables.py` | `results/paper_tables.txt` |
| 11, fixed-size chunking | `prepare.py` | `cache/repo_chunks.json` after running `prepare.py` |
| 12, rejected answers | `paper_tables.py` | `results/paper_tables.txt` |
| 13, latency | `latency_bench.py` | `results/latency_hotpot.json`, `results/run_latency.log` |
| 14, error classification | `paper_tables.py` | `results/paper_tables.txt` |

Every record in `results/gen_*_test.jsonl` holds the question index, the system, the retrieved paragraph ids, the context given to the generator, the answer, the reference and all metrics, so every number can be recomputed.

## Data splits

| Dataset | Tuning | Retrieval test | Generation test |
|---|---|---|---|
| HotpotQA (distractor, validation) | questions 1 to 500 | 501 to 1,500 | 501 to 1,000 |
| 2WikiMultihopQA (validation) | none | 1 to 1,000 | 1 to 500 |
| MuSiQue (answerable, validation) | none | 1 to 1,000 after a fixed-seed shuffle | not run |
| WikiHow (first 10,000 articles) | none (1 to 100 kept apart; four used to check the prompt) | not applicable | 101 to 200 |

The retrieval corpus of each multi-hop dataset is the pool of all context paragraphs of its validation split, without duplicates, searched as a whole.

## Hardware

Intel Core i5-12500H, 16 GB of RAM, NVIDIA GeForce RTX 3050 with 4 GB of memory. The encoder and the cross-encoder run on the graphics processor, and the generators are partly offloaded to it.

## Authors and credit

Kayarvizhy N, B. S. Shreya and Smitha. The first version of this work, including the intent detection module, the prompt templates and the original pipeline in `legacy/`, was implemented by B. S. Shreya ([shreya2518/AdaptiveRAG](https://github.com/shreya2518/AdaptiveRAG)); this repository continues from that code and keeps its history.

## Licence

MIT, see `LICENSE`.

## Legacy code

The folder `legacy/` contains the code and the result files of the first version of this work. They are kept for transparency and must not be used to reproduce the paper; see `legacy/README.md`.
