#!/usr/bin/env python3
"""
Replay the 10 public conversation traces and measure Recall@10.

Hardcoded expected final shortlists from C1–C10 for local evaluation.

Usage:
    python scripts/eval_traces.py [--base-url http://localhost:8000]
"""
import sys
import argparse
import httpx

BASE_URL = "http://localhost:8000"

# Expected final shortlists (assessment names, lowercase) per trace
EXPECTED = {
    "C1": [
        "occupational personality questionnaire opq32r",
        "opq universal competency report 2.0",
        "opq leadership report",
    ],
    "C2": [
        "smart interview live coding",
        "linux programming (general)",
        "networking and implementation (new)",
        "shl verify interactive g+",
        "occupational personality questionnaire opq32r",
    ],
    "C3": [
        "svar spoken english (us) (new)",
        "contact center call simulation (new)",
        "entry level customer serv - retail & contact center",
        "customer service phone simulation",
    ],
    "C4": [
        "shl verify interactive – numerical reasoning",
        "financial accounting (new)",
        "basic statistics (new)",
        "graduate scenarios",
        "occupational personality questionnaire opq32r",
    ],
    "C5": [
        "global skills assessment",
        "global skills development report",
        "occupational personality questionnaire opq32r",
        "opq mq sales report",
        "sales transformation 2.0 - individual contributor",
    ],
    "C6": [
        "manufac. & indust. - safety & dependability 8.0",
        "workplace health and safety (new)",
    ],
    "C7": [
        "hipaa (security)",
        "medical terminology (new)",
        "microsoft word 365 - essentials (new)",
        "dependability and safety instrument (dsi)",
        "occupational personality questionnaire opq32r",
    ],
    "C8": [
        "microsoft excel 365 (new)",
        "microsoft word 365 (new)",
        "ms excel (new)",
        "ms word (new)",
        "occupational personality questionnaire opq32r",
    ],
    "C9": [
        "core java (advanced level) (new)",
        "spring (new)",
        "sql (new)",
        "amazon web services (aws) development (new)",
        "docker (new)",
        "shl verify interactive g+",
        "occupational personality questionnaire opq32r",
    ],
    "C10": [
        "shl verify interactive g+",
        "graduate scenarios",
    ],
}

# Minimal conversation starters (persona summaries) per trace
STARTERS = {
    "C1": "We need a solution for senior leadership — CXOs and directors, 15+ years experience. Selection, comparing candidates against a leadership benchmark.",
    "C2": "I'm hiring a senior Rust engineer for high-performance networking infrastructure. Please recommend assessments including cognitive tests.",
    "C3": "We're screening 500 entry-level contact centre agents. Inbound calls, customer service focus. English US. New simulation for volume, old solution for finalists.",
    "C4": "Hiring graduate financial analysts — final-year students, no work experience. Need numerical reasoning, finance knowledge, and situational judgement for graduates.",
    "C5": "We need to re-skill our sales organization as part of restructuring. Recommend a full audit stack including personality, skills, and sales-specific reports.",
    "C6": "Hiring plant operators for a chemical facility. Industrial context. Safety is the absolute top priority. We're industrial, so the 8.0 bundle fits.",
    "C7": "Hiring bilingual healthcare admin in South Texas. English for written work, Spanish for personality. Need HIPAA, medical terminology, Word skills, and dependability assessments.",
    "C8": "I need to assess admin assistants for Excel and Word daily use. Include both knowledge tests and simulations.",
    "C9": "Senior Full-Stack Engineer, backend-leaning. Core Java advanced, Spring, SQL, AWS, Docker. Senior IC. Keep Verify G+.",
    "C10": "Graduate management trainee scheme. Full battery: cognitive and situational judgement only. Drop OPQ.",
}


def recall_at_k(recommended: list, expected: list, k: int = 10) -> float:
    rec_names = {r["name"].lower() for r in recommended[:k]}
    hits = sum(1 for e in expected if e in rec_names)
    return hits / len(expected) if expected else 0.0


def run_trace(trace_id: str, starter: str, expected: list, base_url: str) -> float:
    messages = [{"role": "user", "content": starter}]
    final_recs = []

    for turn in range(4):  # max 4 rounds
        resp = httpx.post(f"{base_url}/chat", json={"messages": messages}, timeout=45)
        resp.raise_for_status()
        data = resp.json()
        reply = data.get("reply", "")
        recs = data.get("recommendations") or []
        eoc = data.get("end_of_conversation", False)

        messages.append({"role": "assistant", "content": reply})

        if recs:
            final_recs = recs

        if eoc or recs:
            break

        # Simulate user confirming / saying "yes go ahead"
        messages.append({"role": "user", "content": "Yes, go ahead and recommend."})

    score = recall_at_k(final_recs, expected)
    return score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=BASE_URL)
    args = parser.parse_args()

    print(f"Evaluating against {args.base_url}\n")
    scores = {}
    for trace_id, starter in STARTERS.items():
        expected = EXPECTED[trace_id]
        try:
            score = run_trace(trace_id, starter, expected, args.base_url)
            scores[trace_id] = score
            status = "✓" if score >= 0.5 else "✗"
            print(f"  {status} {trace_id}: Recall@10 = {score:.2f}  (expected {len(expected)} items)")
        except Exception as e:
            scores[trace_id] = 0.0
            print(f"  ✗ {trace_id}: ERROR — {e}")

    mean_recall = sum(scores.values()) / len(scores)
    print(f"\nMean Recall@10 across {len(scores)} traces: {mean_recall:.3f}")


if __name__ == "__main__":
    main()
