# demo.py

import time
from rag_pipeline import run_rag

# ── Run Demo ──────────────────────────────────────────────────────────────────

def run_demo():

    print("\n" + "="*70)
    print(f"{'ADAPTIVE RAG — LIVE DEMO':^70}")
    print("="*70)
    print("  Type your question and press Enter")
    print("  Type 'exit' or 'quit' to stop")
    print("="*70)

    query_count = 0

    while True:

        print(f"\n{'─'*70}")
        user_input = input("  Ask anything : ").strip()

        if user_input.lower() in ["exit", "quit", "q"]:
            print("\n" + "="*70)
            print(f"{'DEMO COMPLETE — Total Queries: ' + str(query_count):^70}")
            print("="*70 + "\n")
            break

        if not user_input:
            print("  ⚠️  Please enter a question.")
            continue

        query_count += 1

        print(f"{'─'*70}")
        print(f"  Processing Query #{query_count}...")
        print(f"{'─'*70}")

        start = time.time()

        result = run_rag(
            query    = user_input,
            verbose  = False,
            max_iter = 2,
            tau      = 0.5
        )

        elapsed = round(
            (time.time() - start) * 1000, 1
        )

        # ── Determine dataset used ────────────────────────────────
        dataset_used = (
            "WikiHow"
            if result["intent"] == "procedural"
            else "HotpotQA"
        )

        print(f"\n  ✅ Intent      : {result['intent'].upper()}")
        print(f"  ✅ Effort      : {result['effort'].upper()}")
        print(f"  ✅ Granularity : {result['granularity']}")
        print(f"  ✅ k (docs)    : {result['k']}")
        print(f"  ✅ Model Used  : {result['model']}")
        print(f"  ✅ Dataset     : {dataset_used}")
        print(f"  ✅ Latency     : {elapsed} ms")
        print(f"\n  💬 Answer :")
        print(f"  {result['answer'][:600]}")
        print(f"\n{'─'*70}")


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_demo()
