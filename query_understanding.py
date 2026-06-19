# query_understanding.py

import re

# ── Intent Types ──────────────────────────────────────────────────────────────
INTENT_FACTUAL    = "factual"
INTENT_ANALYTICAL = "analytical"
INTENT_PROCEDURAL = "procedural"

# ── Effort Levels ─────────────────────────────────────────────────────────────
EFFORT_LOW    = "low"
EFFORT_MEDIUM = "medium"
EFFORT_HIGH   = "high"

# ── Keyword Banks ─────────────────────────────────────────────────────────────
PROCEDURAL_KEYWORDS = [
    "how to", "how do i", "how can i", "how do you",
    "how does one", "how should i", "how would i",
    "how would you", "how can you", "how do we",
    "steps to", "guide to", "way to", "ways to",
    "instructions", "procedure", "process",
    "tutorial", "teach me", "walk me through",
    "what are the steps", "method to", "tips for",
    "how do i make", "how do you make",
    "how do i change", "how do you change",
    "how do i fix", "how do you fix",
    "how do i build", "how do you build",
    "how do i create", "how do you create",
    "how do i write", "how do you write",
    "how do i use", "how do you use",
    "how do i install", "how do you install",
    "how do i set up", "how do you set up",
    "how do i get", "how do you get",
    "how do i start", "how do you start",
]

ANALYTICAL_KEYWORDS = [
    "why", "explain", "compare", "difference between", "analyze",
    "analyse", "what is the relationship", "how does", "reason",
    "cause", "effect", "impact", "evaluate", "discuss", "elaborate",
    "what happens when", "in what way", "to what extent", "contrast"
]

FACTUAL_KEYWORDS = [
    "who", "what", "when", "where", "which", "name", "list",
    "define", "what is", "what are", "tell me", "give me",
    "how many", "how much", "how old", "how long", "how far",
    "is it", "was it", "did", "does", "do"
]

# ── Multi-hop Indicators (→ raises effort) ────────────────────────────────────
MULTIHOP_KEYWORDS = [
    "both", "and also", "as well as", "in addition",
    "furthermore", "moreover", "compared to", "versus",
    "while", "whereas", "at the same time", "nationality",
    "were they", "did they both", "same", "different"

    # multi-entity patterns
    "between",
    "shared",
    "common",
    "both of",
    "each of",
    # comparison patterns
    "which came first",
    "who came first",
    "who was born first",
    "who is older",
    "which is older",
    "came before",
    "came after",
    "earlier than",
    "later than",
    "or stardust",
    "or newton",
    "or einstein",
]

COMPLEX_KEYWORDS = [
    "why", "explain", "analyze", "evaluate", "discuss",
    "elaborate", "in what way", "to what extent", "relationship"
]

# ── Core Functions ────────────────────────────────────────────────────────────
def detect_intent(query: str) -> str:
    q = query.lower().strip()

    # procedural check first (strongest signal)
    for kw in PROCEDURAL_KEYWORDS:
        if kw in q:
            return INTENT_PROCEDURAL

    # analytical check
    for kw in ANALYTICAL_KEYWORDS:
        if re.search(rf'\b{re.escape(kw)}\b', q):
            return INTENT_ANALYTICAL

    # factual fallback
    return INTENT_FACTUAL


def estimate_effort(query: str, intent: str) -> str:
    q = query.lower().strip()

    multihop_hits = sum(1 for kw in MULTIHOP_KEYWORDS if kw in q)
    complex_hits  = sum(1 for kw in COMPLEX_KEYWORDS
                        if re.search(rf'\b{re.escape(kw)}\b', q))

    if multihop_hits >= 1 or complex_hits >= 2:
        return EFFORT_HIGH
    if complex_hits == 1:
        return EFFORT_MEDIUM
    if intent == INTENT_ANALYTICAL:
        return EFFORT_MEDIUM
    if intent == INTENT_PROCEDURAL:
        return EFFORT_MEDIUM

    return EFFORT_LOW


def plan_retrieval(intent: str, effort: str) -> dict:
    plan = {
        INTENT_FACTUAL: {
            EFFORT_LOW:    {"granularity": "G2", "k": 7},
            EFFORT_MEDIUM: {"granularity": "G2", "k": 8},
            EFFORT_HIGH:   {"granularity": "G2", "k": 10},
        },
        INTENT_ANALYTICAL: {
            EFFORT_LOW:    {"granularity": "G2", "k": 7},
            EFFORT_MEDIUM: {"granularity": "G2", "k": 8},
            EFFORT_HIGH:   {"granularity": "G3", "k": 10},
        },
        INTENT_PROCEDURAL: {
            EFFORT_LOW:    {"granularity": "G3", "k": 7},
            EFFORT_MEDIUM: {"granularity": "G3", "k": 8},
            EFFORT_HIGH:   {"granularity": "G3", "k": 10},
        },
    }
    return plan[intent][effort]


def understand_query(query: str) -> dict:
    intent         = detect_intent(query)
    effort         = estimate_effort(query, intent)
    retrieval_plan = plan_retrieval(intent, effort)

    return {
        "query"       : query,
        "intent"      : intent,
        "effort"      : effort,
        "granularity" : retrieval_plan["granularity"],
        "k"           : retrieval_plan["k"]
    }


# ── Test ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_queries = [
        "Who is the president of France?",
        "What year was the Eiffel Tower built?",
        "Were Scott Derrickson and Ed Wood of the same nationality?",
        "Why did the Roman Empire fall?",
        "Explain the difference between supervised and unsupervised learning.",
        "How to make sourdough bread at home?",
        "How do you make a cup of tea?",
        "How do you tie shoelaces?",
        "How do you bake a cake?",
        "What are the steps to change a car tyre?",
        "Analyze the relationship between inflation and unemployment.",
    ]

    print("\n" + "="*70)
    print(f"{'QUERY UNDERSTANDING MODULE — TEST':^70}")
    print("="*70)

    for q in test_queries:
        result = understand_query(q)
        print(f"\n📝 Query      : {result['query']}")
        print(f"   Intent     : {result['intent']}")
        print(f"   Effort     : {result['effort']}")
        print(f"   Granularity: {result['granularity']}")
        print(f"   k docs     : {result['k']}")
        print(f"   {'-'*60}")
