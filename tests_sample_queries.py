import os
import sys
import json
from pprint import pprint

# Ensure UTF-8 output on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Benchmark test queries defined in the assignment specification
BENCHMARK_QUERIES = [
    {
        "id": 1,
        "query": "What is Agentic AI according to the eBook?",
        "expected_grounded": True,
    },
    {
        "id": 2,
        "query": "How do AI agents differ from traditional automation systems?",
        "expected_grounded": True,
    },
    {
        "id": 3,
        "query": "What are the core components of an Agentic Architecture?",
        "expected_grounded": True,
    },
    {
        "id": 4,
        "query": "What role does memory play in Agentic AI workflows?",
        "expected_grounded": True,
    },
    {
        "id": 5,
        "query": "Who won the 2022 FIFA World Cup?",
        "expected_grounded": False,  # Strict grounding validation test: MUST refuse
    },
]


def test_via_api(base_url: str = "http://localhost:8000"):
    """Run benchmark queries against the live FastAPI endpoint."""
    import requests

    print("=" * 70)
    print(f"Running Benchmark Verification Tests against API: {base_url}/chat")
    print("=" * 70)

    all_passed = True

    for item in BENCHMARK_QUERIES:
        qid = item["id"]
        query = item["query"]
        expected_grounded = item["expected_grounded"]

        print(f"\n[Test {qid}] Query: '{query}'")
        try:
            resp = requests.post(f"{base_url}/chat", json={"query": query}, timeout=30)
            if resp.status_code != 200:
                print(f"  ❌ API returned status code {resp.status_code}: {resp.text}")
                all_passed = False
                continue

            data = resp.json()
            answer = data.get("final_answer", "")
            chunks = data.get("retrieved_context", [])
            score = data.get("confidence_score", 0.0)

            print(f"  Confidence Score: {score}")
            print(f"  Retrieved Chunks Count: {len(chunks)}")
            print(f"  Answer: {answer[:250]}..." if len(answer) > 250 else f"  Answer: {answer}")

            # Grounding check
            if not expected_grounded:
                # Must refuse or acknowledge lack of context
                refusal_phrases = ["cannot answer", "not provided", "not mentioned", "lack"]
                is_refusal = any(p in answer.lower() for p in refusal_phrases) or score == 0.0
                if is_refusal:
                    print("  ✅ PASS: Out-of-domain query was correctly refused / marked ungrounded.")
                else:
                    print("  ❌ FAIL: System hallucinated or failed to refuse out-of-domain query.")
                    all_passed = False
            else:
                if len(chunks) > 0 and score > 0.0:
                    print("  ✅ PASS: Grounded context retrieved and answer generated.")
                else:
                    print("  ❌ FAIL: Expected grounded answer but got zero chunks or zero score.")
                    all_passed = False

        except Exception as e:
            print(f"  ❌ Request failed: {e}")
            all_passed = False

    return all_passed


def test_via_graph():
    """Run benchmark queries directly using the compiled LangGraph workflow."""
    from src.graph import build_rag_graph

    print("=" * 70)
    print("Running Benchmark Verification Tests directly using LangGraph")
    print("=" * 70)

    graph = build_rag_graph()
    all_passed = True

    for item in BENCHMARK_QUERIES:
        qid = item["id"]
        query = item["query"]
        expected_grounded = item["expected_grounded"]

        print(f"\n[Test {qid}] Query: '{query}'")
        try:
            result = graph.invoke({
                "question": query,
                "context": [],
                "answer": "",
                "score": 0.0,
            })

            answer = result["answer"]
            chunks = result["context"]
            score = result["score"]

            print(f"  Confidence Score: {score}")
            print(f"  Retrieved Chunks Count: {len(chunks)}")
            print(f"  Answer: {answer[:250]}..." if len(answer) > 250 else f"  Answer: {answer}")

            if not expected_grounded:
                refusal_phrases = ["cannot answer", "not provided", "not mentioned", "lack"]
                is_refusal = any(p in answer.lower() for p in refusal_phrases) or score == 0.0
                if is_refusal:
                    print("  ✅ PASS: Out-of-domain query was correctly refused / marked ungrounded.")
                else:
                    print("  ❌ FAIL: System failed to refuse out-of-domain query.")
                    all_passed = False
            else:
                if len(chunks) > 0 and score > 0.0:
                    print("  ✅ PASS: Grounded context retrieved and answer generated.")
                else:
                    print("  ❌ FAIL: Expected grounded answer but got zero chunks or zero score.")
                    all_passed = False

        except Exception as e:
            print(f"  ❌ Execution failed: {e}")
            all_passed = False

    return all_passed


if __name__ == "__main__":
    # If API server is running, use it; otherwise fallback to graph execution
    use_api = "--api" in sys.argv
    if use_api:
        success = test_via_api()
    else:
        # Check if API is reachable
        try:
            import requests
            r = requests.get("http://localhost:8000/health", timeout=2)
            if r.status_code == 200:
                print("Detected running FastAPI server on http://localhost:8000. Running tests via API...")
                success = test_via_api()
            else:
                success = test_via_graph()
        except Exception:
            print("FastAPI server not reachable on localhost:8000. Running tests directly via LangGraph...")
            success = test_via_graph()

    print("\n" + "=" * 70)
    if success:
        print("🎉 ALL BENCHMARK VERIFICATION TESTS PASSED SUCCESSFULLY!")
        sys.exit(0)
    else:
        print("⚠️ SOME BENCHMARK TESTS FAILED OR ENCOUNTERED ERRORS.")
        sys.exit(1)
