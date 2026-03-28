"""
Scaled LLM explanation evaluation (50 synthetic scenarios).

Generates diverse prediction probability combinations and evaluates
all systems on completeness, hallucination, and disclaimer metrics
with 95% confidence intervals via bootstrap resampling.

Usage:
    OPENAI_API_KEY=sk-... python evaluation/eval_llm_scaled.py
"""
import os
import sys
import json
import re
import time
import itertools

import numpy as np
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.explain import build_prompt, generate_explanation
from llm.rule_based import generate_rule_based_explanation


OLLAMA_URL = "http://localhost:11434/api/chat"
CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]

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


# ---------------------------------------------------------------------------
# Test case generation
# ---------------------------------------------------------------------------

def generate_test_cases(n: int = 50, seed: int = 42) -> list[dict]:
    """Generate diverse synthetic prediction scenarios.

    Covers: single-class detections at varying confidence, multi-label
    combinations, borderline cases near threshold, all-negative, and
    edge cases with conflicting signals (e.g. NORM + abnormality).
    """
    rng = np.random.RandomState(seed)
    cases = []

    # Category 1: Single-class high confidence (5 classes × 2 levels)
    for cls in CLASSES:
        for conf in [0.85, 0.95]:
            probs = {c: round(rng.uniform(0.02, 0.15), 2) for c in CLASSES}
            probs[cls] = conf
            cases.append({
                "name": f"{cls}_high_{conf}",
                "probs": probs,
                "category": "single_high",
            })

    # Category 2: Single-class borderline (just above/below 0.5)
    for cls in CLASSES:
        for conf in [0.48, 0.52]:
            probs = {c: round(rng.uniform(0.05, 0.20), 2) for c in CLASSES}
            probs[cls] = conf
            tag = "above" if conf > 0.5 else "below"
            cases.append({
                "name": f"{cls}_borderline_{tag}",
                "probs": probs,
                "category": "borderline",
            })

    # Category 3: Multi-label pairs (all 2-class combinations)
    for cls_a, cls_b in itertools.combinations(["MI", "STTC", "CD", "HYP"], 2):
        probs = {c: round(rng.uniform(0.02, 0.15), 2) for c in CLASSES}
        probs[cls_a] = round(rng.uniform(0.70, 0.95), 2)
        probs[cls_b] = round(rng.uniform(0.60, 0.90), 2)
        probs["NORM"] = round(rng.uniform(0.02, 0.10), 2)
        cases.append({
            "name": f"multi_{cls_a}_{cls_b}",
            "probs": probs,
            "category": "multi_label",
        })

    # Category 4: NORM + one borderline abnormality
    for cls in ["MI", "STTC", "CD", "HYP"]:
        probs = {c: round(rng.uniform(0.02, 0.10), 2) for c in CLASSES}
        probs["NORM"] = round(rng.uniform(0.70, 0.90), 2)
        probs[cls] = round(rng.uniform(0.50, 0.60), 2)
        cases.append({
            "name": f"norm_plus_borderline_{cls}",
            "probs": probs,
            "category": "norm_plus_borderline",
        })

    # Category 5: All below threshold (uncertain)
    for i in range(4):
        probs = {c: round(rng.uniform(0.10, 0.45), 2) for c in CLASSES}
        cases.append({
            "name": f"uncertain_{i}",
            "probs": probs,
            "category": "uncertain",
        })

    # Shuffle and truncate to n
    rng.shuffle(cases)
    cases = cases[:n]

    return cases


# ---------------------------------------------------------------------------
# Model backends
# ---------------------------------------------------------------------------

def generate_ollama(probs: dict, model: str) -> str:
    prompt = build_prompt(probs)
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "options": {"temperature": 0.3, "num_predict": 500},
                "think": False,
                "stream": False,
            },
            timeout=180,
        )
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            return f"[Ollama error: {data['error']}]"
        return data["message"]["content"]
    except Exception as e:
        return f"[Ollama error: {e}]"


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

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
        "correct": mentioned_correct,
        "hallucinated": hallucinated,
        "missed": missed,
    }


def check_disclaimer(text: str) -> bool:
    keywords = ["disclaimer", "not.*diagnosis", "not.*definitive",
                "automated screening", "clinical correlation",
                "should not be used", "does not constitute",
                "clinical judgment", "not a substitute"]
    text_lower = text.lower()
    return any(re.search(kw, text_lower) for kw in keywords)


def bootstrap_ci(values, n_boot=2000, seed=42):
    """Compute mean and 95% CI via bootstrap."""
    rng = np.random.RandomState(seed)
    arr = np.array(values, dtype=float)
    means = []
    for _ in range(n_boot):
        sample = rng.choice(arr, size=len(arr), replace=True)
        means.append(np.mean(sample))
    means = np.array(means)
    return np.mean(arr), np.percentile(means, 2.5), np.percentile(means, 97.5)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cases = generate_test_cases(n=50)
    print(f"Generated {len(cases)} test cases")
    print(f"  Categories: {dict(zip(*np.unique([c['category'] for c in cases], return_counts=True)))}")

    systems = {
        "GPT-5.4": lambda p: generate_explanation(p),
        "Qwen3.5-4B": lambda p: generate_ollama(p, "qwen3.5:4b"),
        "Qwen3.5-2B": lambda p: generate_ollama(p, "qwen3.5:2b"),
        "Qwen3.5-0.8B": lambda p: generate_ollama(p, "qwen3.5:0.8b"),
        "Rule-based": lambda p: generate_rule_based_explanation(p),
    }

    # Collect per-case results
    results = {name: [] for name in systems}

    for i, case in enumerate(cases):
        print(f"\r[{i+1}/{len(cases)}] {case['name']:<40}", end="", flush=True)

        for sys_name, gen_fn in systems.items():
            t0 = time.time()
            text = gen_fn(case["probs"])
            elapsed = time.time() - t0

            fc = check_factual_consistency(text, case["probs"])
            n_detected = len(fc["correct"]) + len(fc["missed"])

            results[sys_name].append({
                "case": case["name"],
                "category": case["category"],
                "n_detected": n_detected,
                "n_correct": len(fc["correct"]),
                "n_hallucinated": len(fc["hallucinated"]),
                "n_missed": len(fc["missed"]),
                "has_disclaimer": check_disclaimer(text),
                "word_count": len(text.split()),
                "latency_s": round(elapsed, 2),
                "hallucinated_classes": fc["hallucinated"],
            })

    print("\n")

    # Compute summary statistics with CIs
    print("=" * 90)
    print(f"SCALED LLM EVALUATION: {len(cases)} synthetic scenarios")
    print("=" * 90)

    header = f"{'Metric':<32}"
    for sys_name in systems:
        header += f"  {sys_name:>14}"
    print(f"\n{header}")
    print("-" * len(header))

    summary = {}

    for sys_name in systems:
        r = results[sys_name]
        n = len(r)

        # Completeness: fraction of detected classes correctly mentioned
        completeness_per_case = []
        for row in r:
            if row["n_detected"] > 0:
                completeness_per_case.append(row["n_correct"] / row["n_detected"])
            else:
                completeness_per_case.append(1.0)  # nothing to detect = complete

        comp_mean, comp_lo, comp_hi = bootstrap_ci(completeness_per_case)

        # Hallucination rate: fraction of cases with any hallucination
        halluc_flags = [1 if row["n_hallucinated"] > 0 else 0 for row in r]
        halluc_mean, halluc_lo, halluc_hi = bootstrap_ci(halluc_flags)

        # Total hallucination count
        total_halluc = sum(row["n_hallucinated"] for row in r)

        # Disclaimer rate
        disc_flags = [1 if row["has_disclaimer"] else 0 for row in r]
        disc_mean, disc_lo, disc_hi = bootstrap_ci(disc_flags)

        # Latency
        latencies = [row["latency_s"] for row in r]
        lat_mean = np.mean(latencies)

        # Word count
        wc_mean = np.mean([row["word_count"] for row in r])

        summary[sys_name] = {
            "completeness": f"{comp_mean:.0%} ({comp_lo:.0%}-{comp_hi:.0%})",
            "halluc_rate": f"{halluc_mean:.0%} ({halluc_lo:.0%}-{halluc_hi:.0%})",
            "halluc_total": total_halluc,
            "disclaimer": f"{disc_mean:.0%} ({disc_lo:.0%}-{disc_hi:.0%})",
            "latency": f"{lat_mean:.1f}",
            "words": f"{wc_mean:.0f}",
        }

    metrics = ["completeness", "halluc_rate", "halluc_total", "disclaimer", "words", "latency"]
    labels = [
        "Completeness (95% CI)",
        "Hallucination rate (95% CI)",
        "Total hallucinated findings",
        "Disclaimer rate (95% CI)",
        "Avg word count",
        "Avg latency (s)",
    ]
    for label, metric in zip(labels, metrics):
        row = f"{label:<32}"
        for sys_name in systems:
            row += f"  {str(summary[sys_name][metric]):>14}"
        print(row)

    # Breakdown by category
    print(f"\n{'=' * 90}")
    print("HALLUCINATION BREAKDOWN BY SCENARIO CATEGORY")
    print(f"{'=' * 90}")

    categories = sorted(set(c["category"] for c in cases))
    cat_header = f"{'Category':<24}"
    for sys_name in systems:
        cat_header += f"  {sys_name:>14}"
    print(f"\n{cat_header}")
    print("-" * len(cat_header))

    for cat in categories:
        row_str = f"{cat:<24}"
        for sys_name in systems:
            cat_results = [r for r in results[sys_name] if r["category"] == cat]
            cat_n = len(cat_results)
            cat_halluc = sum(1 for r in cat_results if r["n_hallucinated"] > 0)
            row_str += f"  {cat_halluc}/{cat_n}:>14"
            row_str = row_str[:-len(f"{cat_halluc}/{cat_n}:>14")]
            row_str += f"  {f'{cat_halluc}/{cat_n}':>14}"
        print(row_str)

    # Save
    save_dir = os.path.join("results", "llm_comparison")
    os.makedirs(save_dir, exist_ok=True)

    out = {
        "n_cases": len(cases),
        "summary": summary,
        "per_system": {
            sys_name: {
                "per_case": results[sys_name],
            }
            for sys_name in systems
        },
    }
    out_path = os.path.join(save_dir, "scaled_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
