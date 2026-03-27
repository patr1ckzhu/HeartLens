"""
Comparative evaluation of LLM explanation systems.

Runs the same test scenarios through GPT-5.4, Qwen3.5-4B (self-hosted),
and the rule-based baseline, then compares factual consistency,
completeness, hallucination, and disclaimer rates.

Usage:
    OPENAI_API_KEY=sk-... python evaluation/eval_llm_comparison.py
"""
import os
import sys
import json
import re
import time

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.explain import build_prompt, generate_explanation
from llm.rule_based import generate_rule_based_explanation


# ---------------------------------------------------------------------------
# Qwen3.5-4B via ollama
# ---------------------------------------------------------------------------

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen3.5:4b"


def generate_qwen_explanation(
    pred_probs: dict[str, float],
    gradcam_regions: list[str] | None = None,
) -> str:
    """Query the local Qwen3.5-4B model via ollama."""
    prompt = build_prompt(pred_probs, gradcam_regions=gradcam_regions)
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "options": {"temperature": 0.3, "num_predict": 500},
                "think": False,
                "stream": False,
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]
    except Exception as e:
        return f"[Qwen API error: {e}]"


# ---------------------------------------------------------------------------
# Evaluation helpers (same logic as eval_llm.py)
# ---------------------------------------------------------------------------

CLASS_KEYWORDS = {
    "NORM": ["normal", "sinus rhythm", "no significant", "no abnormal"],
    "MI": ["myocardial infarction", "infarction", "ischaemic", "ischemic",
           "heart attack", "mi pattern", "q wave"],
    "STTC": ["st-segment", "st segment", "t-wave", "t wave", "repolarisation",
             "repolarization", "sttc", "st change", "st-t"],
    "CD": ["conduction", "bundle branch", "block", "atrioventricular",
           "conduction disturbance"],
    "HYP": ["hypertrophy", "thickening", "voltage criteria", "ventricular hyp"],
}


def check_factual_consistency(text: str, pred_probs: dict, threshold: float = 0.5):
    text_lower = text.lower()
    detected = {k for k, v in pred_probs.items() if v >= threshold}
    not_detected = {k for k, v in pred_probs.items() if v < threshold}

    mentioned_correct, hallucinated, missed = [], [], []

    for cls in detected:
        keywords = CLASS_KEYWORDS.get(cls, [])
        if any(kw in text_lower for kw in keywords):
            mentioned_correct.append(cls)
        else:
            missed.append(cls)

    for cls in not_detected:
        if cls == "NORM":
            continue
        keywords = CLASS_KEYWORDS.get(cls, [])
        for kw in keywords:
            if kw in text_lower:
                context_patterns = [
                    f"detected.*{kw}", f"{kw}.*detected",
                    f"finding.*{kw}", f"{kw}.*finding",
                    f"abnormal.*{kw}", f"{kw}.*confidence",
                ]
                if any(re.search(p, text_lower) for p in context_patterns):
                    hallucinated.append(cls)
                    break

    return {
        "mentioned_correct": mentioned_correct,
        "hallucinated": hallucinated,
        "missed": missed,
    }


def check_disclaimer(text: str) -> bool:
    keywords = ["disclaimer", "not.*diagnosis", "not.*definitive",
                "automated screening", "clinical correlation",
                "should not be used", "does not constitute",
                "not a substitute", "clinical judgment"]
    text_lower = text.lower()
    return any(re.search(kw, text_lower) for kw in keywords)


def count_hedging(text: str) -> int:
    hedging_words = ["may", "possible", "suggest", "warrant",
                     "correlation", "further", "cannot confirm",
                     "does not confirm", "not definitive", "could",
                     "likely", "potential"]
    text_lower = text.lower()
    return sum(1 for w in hedging_words if w in text_lower)


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

TEST_CASES = [
    {
        "name": "Clear normal",
        "probs": {"NORM": 0.92, "MI": 0.05, "STTC": 0.08, "CD": 0.03, "HYP": 0.04},
    },
    {
        "name": "Clear MI",
        "probs": {"NORM": 0.02, "MI": 0.95, "STTC": 0.30, "CD": 0.12, "HYP": 0.08},
    },
    {
        "name": "Multi-label (STTC + CD)",
        "probs": {"NORM": 0.10, "MI": 0.15, "STTC": 0.88, "CD": 0.82, "HYP": 0.20},
    },
    {
        "name": "Uncertain (all below threshold)",
        "probs": {"NORM": 0.35, "MI": 0.30, "STTC": 0.28, "CD": 0.25, "HYP": 0.20},
    },
    {
        "name": "HYP only",
        "probs": {"NORM": 0.15, "MI": 0.10, "STTC": 0.12, "CD": 0.08, "HYP": 0.85},
    },
    {
        "name": "NORM + borderline STTC",
        "probs": {"NORM": 0.78, "MI": 0.05, "STTC": 0.52, "CD": 0.03, "HYP": 0.04},
    },
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_comparison():
    systems = {
        "GPT-5.4": lambda probs: generate_explanation(probs),
        "Qwen3.5-4B": lambda probs: generate_qwen_explanation(probs),
        "Rule-based": lambda probs: generate_rule_based_explanation(probs),
    }

    all_results = {name: [] for name in systems}
    all_outputs = {name: [] for name in systems}

    print("=" * 70)
    print("LLM EXPLANATION COMPARISON: GPT-5.4 vs Qwen3.5-4B vs Rule-based")
    print("=" * 70)

    for case in TEST_CASES:
        print(f"\n{'─' * 70}")
        print(f"Case: {case['name']}")
        print(f"Probs: {case['probs']}")

        for sys_name, gen_fn in systems.items():
            t0 = time.time()
            text = gen_fn(case["probs"])
            elapsed = time.time() - t0

            fc = check_factual_consistency(text, case["probs"])
            has_disclaimer = check_disclaimer(text)
            hedge_count = count_hedging(text)
            word_count = len(text.split())

            result = {
                "case": case["name"],
                "consistency": fc,
                "has_disclaimer": has_disclaimer,
                "hedge_count": hedge_count,
                "word_count": word_count,
                "latency_s": round(elapsed, 1),
            }
            all_results[sys_name].append(result)
            all_outputs[sys_name].append({"case": case["name"], "text": text})

            status = "✓" if not fc["hallucinated"] and not fc["missed"] else "✗"
            print(f"\n  [{sys_name}] {status} ({elapsed:.1f}s, {word_count} words)")
            print(f"    Correct: {fc['mentioned_correct']}  "
                  f"Missed: {fc['missed']}  "
                  f"Hallucinated: {fc['hallucinated']}  "
                  f"Disclaimer: {has_disclaimer}  "
                  f"Hedging: {hedge_count}")

    # Summary table
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"\n{'Metric':<35} {'GPT-5.4':>10} {'Qwen3.5-4B':>12} {'Rule-based':>12}")
    print("-" * 70)

    for sys_name in systems:
        results = all_results[sys_name]
        n = len(results)

        total_detected = sum(
            len(r["consistency"]["mentioned_correct"]) + len(r["consistency"]["missed"])
            for r in results
        )
        total_mentioned = sum(len(r["consistency"]["mentioned_correct"]) for r in results)
        total_hallucinated = sum(len(r["consistency"]["hallucinated"]) for r in results)
        disclaimer_count = sum(1 for r in results if r["has_disclaimer"])
        avg_hedging = sum(r["hedge_count"] for r in results) / n
        avg_words = sum(r["word_count"] for r in results) / n
        avg_latency = sum(r["latency_s"] for r in results) / n

        completeness = total_mentioned / total_detected if total_detected > 0 else 0

        all_results[sys_name] = {
            "per_case": results,
            "summary": {
                "completeness": f"{total_mentioned}/{total_detected} ({completeness:.0%})",
                "hallucinations": total_hallucinated,
                "disclaimer_rate": f"{disclaimer_count}/{n} ({disclaimer_count/n:.0%})",
                "avg_hedging": round(avg_hedging, 1),
                "avg_words": round(avg_words, 0),
                "avg_latency_s": round(avg_latency, 1),
            }
        }

    # Print summary rows
    metrics = ["completeness", "hallucinations", "disclaimer_rate",
               "avg_hedging", "avg_words", "avg_latency_s"]
    labels = ["Completeness", "Hallucinations", "Disclaimer rate",
              "Avg hedging words", "Avg word count", "Avg latency (s)"]

    for label, metric in zip(labels, metrics):
        vals = []
        for sys_name in systems:
            v = all_results[sys_name]["summary"][metric]
            vals.append(str(v))
        print(f"{label:<35} {vals[0]:>10} {vals[1]:>12} {vals[2]:>12}")

    # Save results
    save_dir = os.path.join("results", "llm_comparison")
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    for sys_name in systems:
        safe_name = sys_name.lower().replace(" ", "_").replace(".", "")
        for entry in all_outputs[sys_name]:
            case_name = entry["case"].lower().replace(" ", "_").replace("(", "").replace(")", "")
            path = os.path.join(save_dir, f"{case_name}_{safe_name}.txt")
            with open(path, "w") as f:
                f.write(entry["text"])

    print(f"\nOutputs saved to {save_dir}/")


if __name__ == "__main__":
    run_comparison()
