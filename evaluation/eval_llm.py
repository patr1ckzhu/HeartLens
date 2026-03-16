"""
LLM explanation evaluation.

Compares LLM-generated and rule-based explanations across three axes:
  1. Factual consistency: do mentioned findings match model predictions?
  2. Completeness: are all detected abnormalities mentioned?
  3. Error propagation: how does the LLM handle deliberately wrong inputs?

Usage:
    OPENAI_API_KEY=sk-... python evaluation/eval_llm.py
"""
import os
import sys
import json
import re

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.explain import generate_explanation, identify_gradcam_regions
from llm.rule_based import generate_rule_based_explanation


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
    """Check if the explanation mentions classes consistent with predictions.

    Returns:
        Dict with 'mentioned_correct', 'hallucinated', 'missed' lists.
    """
    text_lower = text.lower()
    detected = {k for k, v in pred_probs.items() if v >= threshold}
    not_detected = {k for k, v in pred_probs.items() if v < threshold}

    mentioned_correct = []
    hallucinated = []
    missed = []

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
        # Only count as hallucination if the class is described as detected/present
        # (not just mentioned in the probability list)
        mentioned_as_finding = False
        for kw in keywords:
            # Look for keywords in diagnostic context, not just probability listing
            if kw in text_lower:
                context_patterns = [
                    f"detected.*{kw}", f"{kw}.*detected",
                    f"finding.*{kw}", f"{kw}.*finding",
                    f"abnormal.*{kw}", f"{kw}.*confidence",
                ]
                for pattern in context_patterns:
                    if re.search(pattern, text_lower):
                        mentioned_as_finding = True
                        break
            if mentioned_as_finding:
                break
        if mentioned_as_finding:
            hallucinated.append(cls)

    return {
        "mentioned_correct": mentioned_correct,
        "hallucinated": hallucinated,
        "missed": missed,
    }


def check_disclaimer(text: str) -> bool:
    """Check if the explanation includes a screening limitation disclaimer."""
    disclaimer_keywords = ["disclaimer", "not.*diagnosis", "not.*definitive",
                          "automated screening", "clinical correlation",
                          "should not be used as"]
    text_lower = text.lower()
    return any(re.search(kw, text_lower) for kw in disclaimer_keywords)


def run_evaluation():
    # Test scenarios covering normal, abnormal, uncertain, and wrong predictions
    test_cases = [
        {
            "name": "Clear normal",
            "probs": {"NORM": 0.92, "MI": 0.05, "STTC": 0.08, "CD": 0.03, "HYP": 0.04},
            "type": "normal",
        },
        {
            "name": "Clear MI",
            "probs": {"NORM": 0.02, "MI": 0.95, "STTC": 0.30, "CD": 0.12, "HYP": 0.08},
            "type": "abnormal",
        },
        {
            "name": "Multi-label (STTC + CD)",
            "probs": {"NORM": 0.10, "MI": 0.15, "STTC": 0.88, "CD": 0.82, "HYP": 0.20},
            "type": "abnormal",
        },
        {
            "name": "Uncertain (all low)",
            "probs": {"NORM": 0.35, "MI": 0.30, "STTC": 0.28, "CD": 0.25, "HYP": 0.20},
            "type": "uncertain",
        },
        {
            "name": "ERROR PROPAGATION: healthy ECG mislabelled as MI",
            "probs": {"NORM": 0.05, "MI": 0.96, "STTC": 0.15, "CD": 0.08, "HYP": 0.03},
            "type": "error_propagation",
        },
        {
            "name": "ERROR PROPAGATION: healthy ECG mislabelled as HYP+STTC",
            "probs": {"NORM": 0.10, "MI": 0.12, "STTC": 0.85, "CD": 0.05, "HYP": 0.78},
            "type": "error_propagation",
        },
    ]

    print("=" * 70)
    print("LLM EXPLANATION EVALUATION")
    print("=" * 70)

    results = {"llm": [], "rule_based": []}

    for case in test_cases:
        print(f"\n{'─' * 70}")
        print(f"Case: {case['name']}")
        print(f"Probs: {case['probs']}")
        print(f"{'─' * 70}")

        # Generate both explanations
        llm_text = generate_explanation(case["probs"])
        rb_text = generate_rule_based_explanation(case["probs"])

        for system_name, text in [("LLM", llm_text), ("Rule-based", rb_text)]:
            fc = check_factual_consistency(text, case["probs"])
            has_disclaimer = check_disclaimer(text)

            result = {
                "case": case["name"],
                "type": case["type"],
                "consistency": fc,
                "has_disclaimer": has_disclaimer,
            }
            results[system_name.lower().replace("-", "_")].append(result)

            print(f"\n  [{system_name}]")
            print(f"  Correct mentions: {fc['mentioned_correct']}")
            print(f"  Missed findings:  {fc['missed']}")
            print(f"  Hallucinated:     {fc['hallucinated']}")
            print(f"  Has disclaimer:   {has_disclaimer}")

        # For error propagation cases, check if LLM hedges or blindly explains
        if case["type"] == "error_propagation":
            llm_lower = llm_text.lower()
            hedging_words = ["may", "possible", "suggest", "warrant",
                           "correlation", "further", "cannot confirm",
                           "does not confirm", "not definitive"]
            hedge_count = sum(1 for w in hedging_words if w in llm_lower)
            print(f"\n  [Error propagation analysis]")
            print(f"  LLM hedging language count: {hedge_count}")
            if hedge_count >= 3:
                print(f"  → LLM uses appropriate hedging language")
            else:
                print(f"  → WARNING: LLM may present errors too confidently")

    # Summary statistics
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")

    for system in ["llm", "rule_based"]:
        system_results = results[system]
        n = len(system_results)

        total_detected = sum(
            len(r["consistency"]["mentioned_correct"]) + len(r["consistency"]["missed"])
            for r in system_results
        )
        total_mentioned = sum(
            len(r["consistency"]["mentioned_correct"]) for r in system_results
        )
        total_hallucinated = sum(
            len(r["consistency"]["hallucinated"]) for r in system_results
        )
        total_disclaimer = sum(1 for r in system_results if r["has_disclaimer"])

        completeness = total_mentioned / total_detected if total_detected > 0 else 0
        disclaimer_rate = total_disclaimer / n

        label = "LLM (GPT-5.4)" if system == "llm" else "Rule-based"
        print(f"\n  {label}:")
        print(f"    Completeness (findings mentioned/detected): "
              f"{total_mentioned}/{total_detected} = {completeness:.0%}")
        print(f"    Hallucination count: {total_hallucinated}")
        print(f"    Disclaimer rate: {total_disclaimer}/{n} = {disclaimer_rate:.0%}")

    # Save detailed outputs for report
    save_dir = "results/llm_eval"
    os.makedirs(save_dir, exist_ok=True)

    for case in test_cases:
        safe_name = case["name"].lower().replace(" ", "_").replace(":", "")
        llm_text = generate_explanation(case["probs"])
        rb_text = generate_rule_based_explanation(case["probs"])

        with open(os.path.join(save_dir, f"{safe_name}_llm.txt"), "w") as f:
            f.write(llm_text)
        with open(os.path.join(save_dir, f"{safe_name}_rule.txt"), "w") as f:
            f.write(rb_text)

    print(f"\n  Detailed outputs saved to {save_dir}/")


if __name__ == "__main__":
    run_evaluation()
