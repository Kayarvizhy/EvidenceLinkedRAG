# uncertainty.py
import re
from rag_pipeline import run_rag
from retriever import retrieve
from query_understanding import understand_query

# ── Uncertainty Signals ───────────────────────────────────────────────────────
UNCERTAINTY_PHRASES = [
    "i don't know",
    "i do not know",
    "not sure",
    "cannot determine",
    "can't determine",
    "unclear",
    "insufficient",
    "not enough information",
    "no information",
    "not mentioned",
    "not provided",
    "does not mention",
    "context does not",
    "not in the context",
    "cannot answer",
    "can't answer",
    "unable to answer",
    "not available",
    "no context",
    "not specified",
]

LOW_SCORE_THRESHOLD = 0.45   # if top retrieval score < this → uncertain
MAX_RERUNS          = 2      # max re-retrieval cycles as per document

# ── Granularity Escalation ────────────────────────────────────────────────────
GRANULARITY_ESCALATION = {
    "G1": "G2",
    "G2": "G3",
    "G3": "G3",   # already at max
}

K_ESCALATION = {
    3: 5,
    5: 7,
    7: 10,
    10: 15,  # cycle 2 gets even more
}

# ── Core Functions ────────────────────────────────────────────────────────────
def is_uncertain(answer: str, retrieved: list) -> tuple:
    """
    Detect uncertainty from:
    1. Answer text contains uncertainty phrases
    2. Top retrieval score is too low
    Returns (is_uncertain: bool, reason: str)
    """
    answer_lower = answer.lower()

    # check answer text
    for phrase in UNCERTAINTY_PHRASES:
        if phrase in answer_lower:
            return True, f"answer contains uncertainty phrase: '{phrase}'"

    # check retrieval scores
    if retrieved:
        top_score = retrieved[0]["score"]
        if top_score < LOW_SCORE_THRESHOLD:
            return True, f"low retrieval score: {top_score:.4f} < {LOW_SCORE_THRESHOLD}"

    return False, ""


def selective_re_retrieval(
    query      : str,
    intent     : str,
    granularity: str,
    k          : int,
    cycle      : int
) -> tuple:
    """
    Escalate granularity + k for re-retrieval
    Returns (new_granularity, new_k)
    """
    new_gran = GRANULARITY_ESCALATION[granularity]
    new_k    = K_ESCALATION[k]

    print(f"\n   🔄 Re-Retrieval Cycle {cycle}:")
    print(f"      Granularity : {granularity} → {new_gran}")
    print(f"      k           : {k} → {new_k}")

    return new_gran, new_k


def run_rag_with_uncertainty(query: str, verbose: bool = True) -> dict:
    """
    Full Adaptive RAG Pipeline WITH Uncertainty Monitor:
    1. Run initial RAG
    2. Check uncertainty
    3. If uncertain → escalate granularity + k → re-retrieve + regenerate
    4. Repeat up to MAX_RERUNS cycles
    5. Return final answer
    """
    from rag_pipeline import (
        _build_context,
        _generate,
        MODEL_MAP,
        PROMPT_TEMPLATES,
    )

    # Step 1 — initial RAG run
    result = run_rag(query, verbose=verbose)

    answer      = result["answer"]
    retrieved   = result["retrieved"]
    granularity = result["granularity"]
    k           = result["k"]
    intent      = result["intent"]
    cycles      = 0

    # Step 2 — uncertainty loop
    while cycles < MAX_RERUNS:
        uncertain, reason = is_uncertain(answer, retrieved)

        if not uncertain:
            if verbose:
                print(f"\n   ✅ Confident answer — no re-retrieval needed.")
            break

        if verbose:
            print(f"\n   ⚠️  Uncertainty detected: {reason}")

        cycles += 1

        # escalate
        granularity, k = selective_re_retrieval(
            query, intent, granularity, k, cycles
        )

        # re-retrieve
        retrieved = retrieve(
            query       = query,
            intent      = intent,
            granularity = granularity,
            k           = k
        )

        if verbose:
            print(f"\n   📚 Re-retrieved {len(retrieved)} chunks:")
            for r in retrieved:
                print(f"      [{r['rank']}] {r['title'][:55]} (score={r['score']:.4f})")

        # rebuild context + regenerate
        context  = _build_context(retrieved, granularity, intent)
        template = PROMPT_TEMPLATES[intent]
        prompt   = template.format(context=context, query=query)
        model    = MODEL_MAP[intent]

        if verbose:
            print(f"\n   🤖 Re-generating with : {model} (cycle {cycles})")

        answer = _generate(prompt, model)

        if verbose:
            print(f"\n   ✅ Refined Answer (cycle {cycles}):\n   {answer}")

    result["answer"]           = answer
    result["retrieved"]        = retrieved
    result["granularity"]      = granularity
    result["k"]                = k
    result["re_retrieval_cycles"] = cycles

    return result


# ── Test ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    test_queries = [
        # this one triggered "I don't know" → should re-retrieve
        "Were Scott Derrickson and Ed Wood of the same nationality?",
        # this one was confident → no re-retrieval needed
        "Why did the Roman Empire fall?",
        # procedural — confident
        "How to make sourdough bread at home?",
    ]

    print("\n" + "="*65)
    print(f"{'UNCERTAINTY MONITOR + RE-RETRIEVAL TEST':^65}")
    print("="*65)

    for q in test_queries:
        result = run_rag_with_uncertainty(q, verbose=True)
        print(f"\n   🔁 Re-retrieval cycles : {result['re_retrieval_cycles']}")
        print(f"   📊 Final granularity   : {result['granularity']}")
        print(f"   📊 Final k             : {result['k']}")
        print("\n" + "="*65)
