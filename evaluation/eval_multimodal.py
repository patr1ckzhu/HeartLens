"""
Multimodal LLM evaluation for ECG Grad-CAM interpretation.

Compares GPT-5.4, Qwen3.5-4B, Qwen3.5-2B, and Qwen3.5-0.8B on their
ability to interpret Grad-CAM attention heatmaps directly from images.

Usage:
    OPENAI_API_KEY=sk-... python evaluation/eval_multimodal.py
"""
import os
import sys
import base64
import json
import time
import re

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OLLAMA_URL = "http://localhost:11434/api/chat"
GRADCAM_DIR = "results/figures"

# What each class's Grad-CAM should plausibly attend to
CLASS_EXPECTED_FEATURES = {
    "NORM": ["qrs", "rhythm", "regular", "heartbeat", "depolarization"],
    "MI": ["q wave", "st segment", "st-segment", "qrs", "ischaemic", "ischemic", "infarct"],
    "STTC": ["st segment", "st-segment", "t wave", "t-wave", "repolarization", "repolarisation"],
    "CD": ["qrs", "conduction", "bundle branch", "block", "pr interval", "depolarization", "widened"],
    "HYP": ["qrs", "amplitude", "voltage", "high-amplitude", "hypertrophy", "tall"],
}


def build_multimodal_prompt(cls: str) -> str:
    return (
        "You are a cardiology AI assistant. This image shows an ECG waveform "
        "(Lead I) with a Grad-CAM attention heatmap overlay from an automated "
        "deep learning classifier. The top panel shows the ECG signal "
        "colour-coded by model attention intensity (red = high attention, "
        "blue = low attention). The bottom panel shows the raw attention "
        "magnitude over time.\n\n"
        f"The automated classifier detected: {cls} with high confidence.\n\n"
        "Based on what you observe in the image:\n"
        "1. Describe which specific ECG waveform features the model is "
        "attending to (e.g. QRS complex, ST segment, T wave, P wave).\n"
        f"2. Are the attention patterns clinically plausible for {cls}?\n"
        "3. Provide a 2-3 sentence clinical interpretation.\n"
        "4. Include a brief disclaimer about automated screening limitations.\n\n"
        "IMPORTANT: ONLY discuss the detected class listed above. Do NOT "
        "speculate about other conditions not mentioned."
    )


# ---------------------------------------------------------------------------
# Model backends
# ---------------------------------------------------------------------------

def query_ollama_multimodal(prompt: str, img_b64: str, model: str) -> str:
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt, "images": [img_b64]}],
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


def query_gpt_multimodal(prompt: str, img_b64: str) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return "[OPENAI_API_KEY not set]"
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-5.4",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{img_b64}",
                    }},
                ],
            }],
            temperature=0.3,
            max_completion_tokens=500,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[GPT error: {e}]"


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def check_mentions_expected_features(text: str, cls: str) -> tuple[list, int]:
    """Check if the explanation mentions clinically expected features."""
    text_lower = text.lower()
    expected = CLASS_EXPECTED_FEATURES.get(cls, [])
    mentioned = [kw for kw in expected if kw in text_lower]
    return mentioned, len(mentioned)


def check_hallucination(text: str, cls: str) -> list:
    """Check if the model discusses other classes not in the prompt."""
    text_lower = text.lower()
    other_classes = {
        "NORM": [],  # Normal doesn't hallucinate other conditions easily
        "MI": ["hypertrophy", "conduction disturbance", "bundle branch"],
        "STTC": ["myocardial infarction", "heart attack", "hypertrophy", "bundle branch"],
        "CD": ["myocardial infarction", "heart attack", "hypertrophy", "st-t change"],
        "HYP": ["myocardial infarction", "heart attack", "conduction disturbance", "bundle branch"],
    }
    hallucinated = []
    for kw in other_classes.get(cls, []):
        # Only count as hallucination if presented as a finding, not as differential
        patterns = [
            f"detected.*{kw}", f"{kw}.*detected",
            f"suggests.*{kw}", f"{kw}.*present",
            f"evidence of.*{kw}",
        ]
        if any(re.search(p, text_lower) for p in patterns):
            hallucinated.append(kw)
    return hallucinated


def check_disclaimer(text: str) -> bool:
    keywords = ["disclaimer", "not.*diagnosis", "not.*definitive",
                "automated screening", "clinical correlation",
                "should not be used", "does not constitute",
                "clinical judgment", "not a substitute"]
    text_lower = text.lower()
    return any(re.search(kw, text_lower) for kw in keywords)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    systems = {
        "GPT-5.4": lambda prompt, img: query_gpt_multimodal(prompt, img),
        "Qwen3.5-4B": lambda prompt, img: query_ollama_multimodal(prompt, img, "qwen3.5:4b"),
        "Qwen3.5-2B": lambda prompt, img: query_ollama_multimodal(prompt, img, "qwen3.5:2b"),
        "Qwen3.5-0.8B": lambda prompt, img: query_ollama_multimodal(prompt, img, "qwen3.5:0.8b"),
    }

    classes = ["NORM", "MI", "STTC", "CD", "HYP"]
    all_results = {name: [] for name in systems}

    print("=" * 80)
    print("MULTIMODAL EVALUATION: Grad-CAM Image Interpretation")
    print("=" * 80)

    for cls in classes:
        img_path = os.path.join(GRADCAM_DIR, f"gradcam_{cls}_0.png")
        if not os.path.exists(img_path):
            print(f"  [SKIP] {img_path} not found")
            continue

        with open(img_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()

        prompt = build_multimodal_prompt(cls)

        print(f"\n{'─' * 80}")
        print(f"CLASS: {cls}")

        for sys_name, query_fn in systems.items():
            t0 = time.time()
            text = query_fn(prompt, img_b64)
            elapsed = time.time() - t0

            mentioned, feat_count = check_mentions_expected_features(text, cls)
            hallucinated = check_hallucination(text, cls)
            has_disclaimer = check_disclaimer(text)
            word_count = len(text.split())

            result = {
                "class": cls,
                "features_mentioned": mentioned,
                "feature_count": feat_count,
                "hallucinated": hallucinated,
                "has_disclaimer": has_disclaimer,
                "word_count": word_count,
                "latency_s": round(elapsed, 1),
            }
            all_results[sys_name].append(result)

            status = "✓" if not hallucinated else "✗"
            print(f"\n  [{sys_name}] {status} ({elapsed:.1f}s, {word_count} words)")
            print(f"    Features: {mentioned}")
            print(f"    Hallucinated: {hallucinated}")
            print(f"    Disclaimer: {has_disclaimer}")

    # Summary
    print(f"\n{'=' * 80}")
    print("MULTIMODAL SUMMARY")
    print(f"{'=' * 80}")

    header = f"{'Metric':<32}"
    for sys_name in systems:
        header += f" {sys_name:>13}"
    print(f"\n{header}")
    print("-" * len(header))

    summary = {}
    for sys_name in systems:
        results = all_results[sys_name]
        n = len(results)
        total_features = sum(r["feature_count"] for r in results)
        total_halluc = sum(len(r["hallucinated"]) for r in results)
        disclaimer_count = sum(1 for r in results if r["has_disclaimer"])
        avg_words = sum(r["word_count"] for r in results) / n if n else 0
        avg_latency = sum(r["latency_s"] for r in results) / n if n else 0

        summary[sys_name] = {
            "avg_features_mentioned": round(total_features / n, 1) if n else 0,
            "total_hallucinations": total_halluc,
            "disclaimer_rate": f"{disclaimer_count}/{n} ({disclaimer_count*100//n}%)" if n else "N/A",
            "avg_words": round(avg_words),
            "avg_latency_s": round(avg_latency, 1),
        }

    metrics = ["avg_features_mentioned", "total_hallucinations",
               "disclaimer_rate", "avg_words", "avg_latency_s"]
    labels = ["Avg features mentioned/class", "Total hallucinations",
              "Disclaimer rate", "Avg word count", "Avg latency (s)"]

    for label, metric in zip(labels, metrics):
        row = f"{label:<32}"
        for sys_name in systems:
            row += f" {str(summary[sys_name][metric]):>13}"
        print(row)

    # Save
    save_dir = os.path.join("results", "llm_comparison")
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "multimodal_results.json")
    with open(out_path, "w") as f:
        json.dump({"per_model": all_results, "summary": summary}, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
