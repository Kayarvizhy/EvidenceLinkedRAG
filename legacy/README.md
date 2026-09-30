# Legacy code: the earlier version of the method

This folder holds the code and result files of the first version of this work, unchanged. It is kept for transparency and is not used by the paper.

Do not use these scripts to reproduce the paper. The `evaluation/table*_*.py` scripts and the files in `results/` belong to the earlier version and are superseded. Their numbers are not comparable with the paper: the baselines in these scripts were given a smaller generation budget than the adaptive method (30 output tokens and a context cut to 500 characters, against 512 tokens and the full context), and the tables were computed on different question subsets.

The method in this folder appears in the paper as the baseline "Heuristic Adaptive RAG". It is re-implemented for that purpose in `systems.py` (class `UamgV1`) in the repository root, where it receives the same generator, prompts, decoding settings and corpus as every other system. `query_understanding.py` here is identical to `repo_query_understanding.py` in the root, which the new code imports.
