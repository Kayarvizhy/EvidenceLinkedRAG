# rag_pipeline.py
import json
import requests
from query_understanding import understand_query
from retriever import retrieve

# ── Ollama Config ─────────────────────────────────────────────────────────────
OLLAMA_URL = "http://localhost:11434/api/generate"

# As per document: LLaMA-3 for reasoning/factual, Mistral for procedural
MODEL_MAP = {
    "factual"    : "llama3",
    "analytical" : "llama3",
    "procedural" : "mistral",
}

# ── Prompt Templates ──────────────────────────────────────────────────────────
PROMPT_TEMPLATES = {
    "factual": """You are a precise factual assistant.
Use the context below to answer the question.
Extract the most relevant answer even if
details differ slightly from the question.
Give the shortest possible answer - just
the key fact, name, date or term.
Do not say I don't know if related
information exists in the context.

Context:
{context}

Question: {query}

Answer concisely and directly:""",

    "analytical": """You are an analytical assistant.
Use the context below to provide a well-reasoned explanation.
Think step by step.

Context:
{context}

Question: {query}

Detailed Answer:""",

    "procedural": """You are a helpful step-by-step assistant.
Use the context below to provide clear, numbered steps.

Context:
{context}

Question: {query}

Step-by-step Answer:""",
}

# ── Fetch chunk texts ─────────────────────────────────────────────────────────
import os, faiss as _faiss
from sentence_transformers import SentenceTransformer
import torch

BASE       = os.path.expanduser("~/OneDrive/Desktop/AdaptiveRAG")
MODEL_PATH = os.path.join(BASE, "models/all-MiniLM-L6-v2")
DEVICE     = "mps" if torch.backends.mps.is_available() else "cpu"

# chunk text lookup
CHUNK_TEXT_REGISTRY = {
    "G1": {
        "hotpotqa": f"{BASE}/corpus/hotpotqa/g1_sentence/hotpotqa_g1.json",
        "wikihow" : f"{BASE}/corpus/wikihow/g1_sentence/wikihow_g1.json",
    },
    "G2": {
        "hotpotqa": f"{BASE}/corpus/hotpotqa/g2_paragraph/hotpotqa_g2.json",
        "wikihow" : f"{BASE}/corpus/wikihow/g2_paragraph/wikihow_g2.json",
    },
    "G3": {
        "hotpotqa": f"{BASE}/corpus/hotpotqa/g3_section/hotpotqa_g3.json",
        "wikihow" : f"{BASE}/corpus/wikihow/g3_section/wikihow_g3.json",
    },
}

_chunk_text_cache = {}

def _get_chunk_texts(granularity: str, source: str) -> dict:
    key = f"{granularity}_{source}"
    if key not in _chunk_text_cache:
        path   = CHUNK_TEXT_REGISTRY[granularity][source]
        chunks = json.load(open(path))
        _chunk_text_cache[key] = {c["chunk_id"]: c["text"] for c in chunks}
    return _chunk_text_cache[key]


def _build_context(results: list, granularity: str, intent: str) -> str:
    """Build context string from retrieved chunks"""
    source     = "wikihow" if intent == "procedural" else "hotpotqa"
    chunk_texts = _get_chunk_texts(granularity, source)

    context_parts = []
    for r in results:
        text = chunk_texts.get(r["chunk_id"], "")
        if text:
            context_parts.append(f"[{r['rank']}] {r['title']}\n{text}")

    return "\n\n".join(context_parts)


def _generate(prompt: str, model: str) -> str:
    """Call Ollama API for generation"""
    payload = {
        "model"  : model,
        "prompt" : prompt,
        "stream" : False,
        "options": {
            "temperature": 0.3,
            "num_predict": 512,
        }
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except Exception as e:
        return f"[Generation Error]: {str(e)}"


def _uncertainty_score(answer, results):
    if not answer or not results:
        return 1.0
    ans_tokens = set(answer.lower().split())
    ctx_tokens = set()
    for r in results:
        ctx_tokens.update(
            r.get("text","").lower().split())
    if not ans_tokens:
        return 1.0
    overlap = len(
        ans_tokens & ctx_tokens
    ) / len(ans_tokens)
    return round(1.0 - overlap, 4)


def _refine_query(query, answer):
    words = [w for w in answer.split()
             if len(w) > 5]
    hint  = " ".join(words[-3:])             if words else ""
    if hint:
        return f"{query} {hint}"
    return query


def run_rag(query: str,
            verbose: bool = True,
            max_iter: int = 0,
            tau: float = 0.5,
            force_k: int = None,
            skip_intent: bool = False,
            skip_effort: bool = False,
            skip_granularity: bool = False,
            force_granularity: str = None,
            skip_uncertainty: bool = False) -> dict:
    """
    Full Adaptive RAG Pipeline:
    1. Query Understanding
    2. Adaptive Retrieval
    3. Context Building
    4. Prompt Construction
    5. Generation
    6. Uncertainty Check + Re-Retrieval

    max_iter=0          -> no re-retrieval
    max_iter=1          -> 1 refinement cycle
    max_iter=2          -> 2 refinement cycles
    tau                 -> uncertainty threshold
    force_k             -> override adaptive k
    skip_intent         -> use default intent
    skip_effort         -> use default effort
    skip_granularity    -> use fixed G2
    skip_uncertainty    -> skip uncertainty check
    """
    # Step 1 - understand query
    profile = understand_query(query)

    # apply ablation flags
    if skip_intent:
        profile["intent"] = "factual"

    if skip_effort:
        profile["effort"] = "medium"

    if skip_granularity:
        profile["granularity"] = "G2"

    if force_granularity is not None:
        profile["granularity"] = force_granularity

    if force_k is not None:
        profile["k"] = force_k

    if verbose:
        print(f"\n{'='*65}")
        print(f"Query      : {query}")
        print(f"Intent     : {profile['intent']}")
        print(f"Effort     : {profile['effort']}")
        print(f"Granularity: {profile['granularity']}")
        print(f"k          : {profile['k']}")

    current_query = query
    answer        = ""
    results       = []
    context       = ""
    model         = MODEL_MAP[profile["intent"]]

    for iteration in range(max_iter + 1):

        # Step 2 - retrieve
        results = retrieve(
            query       = current_query,
            intent      = profile["intent"],
            granularity = profile["granularity"],
            k           = profile["k"]
        )

        if verbose:
            print(f"\nRetrieved {len(results)}"
                  f" chunks (iter={iteration}):")
            for r in results:
                print(f"  [{r['rank']}]"
                      f" {r['title'][:55]}"
                      f" score={r['score']:.4f}")

        # Step 3 - build context
        context = _build_context(
            results,
            profile["granularity"],
            profile["intent"])

        # Step 4 - build prompt
        template = PROMPT_TEMPLATES[
            profile["intent"]]
        prompt   = template.format(
            context=context,
            query=query)

        # Step 5 - generate
        if verbose:
            print(f"\nGenerating with: {model}")

        answer = _generate(prompt, model)

        if verbose:
            print(f"\nAnswer: {answer}")

        # Step 6 - uncertainty check
        if skip_uncertainty:
            break

        if iteration < max_iter:
            u = _uncertainty_score(
                answer, results)
            if verbose:
                print(f"Uncertainty: {u:.4f}"
                      f" (tau={tau})")
            if u <= tau:
                if verbose:
                    print("Confident - stopping")
                break
            current_query = _refine_query(
                query, answer)
            if verbose:
                print(f"Refined query: "
                      f"{current_query[:60]}")

    if verbose:
        print(f"{'='*65}")

    return {
        "query"       : query,
        "intent"      : profile["intent"],
        "effort"      : profile["effort"],
        "granularity" : profile["granularity"],
        "k"           : profile["k"],
        "model"       : model,
        "context"     : context,
        "answer"      : answer,
        "retrieved"   : results,
    }

# ── Test ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_queries = [
        "Were Scott Derrickson and Ed Wood of the same nationality?",
        "How to make sourdough bread at home?",
        "Why did the Roman Empire fall?",
    ]

    for q in test_queries:
        run_rag(q, verbose=True)
