"""
LLM-powered ECG explanation module.

Generates clinician-friendly natural language explanations of model
predictions by formatting structured diagnostic information into a
prompt and querying an LLM via an OpenAI-compatible API.
"""
import os
import json
from typing import Optional

import numpy as np


# Class descriptions grounded in clinical cardiology
CLASS_DESCRIPTIONS = {
    "NORM": "Normal sinus rhythm with no significant abnormalities",
    "MI": "Myocardial infarction (heart attack) pattern, indicating possible "
          "ischaemic damage to the heart muscle",
    "STTC": "ST-segment or T-wave changes, often associated with myocardial "
            "ischaemia, electrolyte imbalances, or drug effects",
    "CD": "Conduction disturbance, such as bundle branch block or "
          "atrioventricular block, indicating abnormal electrical propagation",
    "HYP": "Ventricular hypertrophy, suggesting thickening of the heart "
           "muscle walls, commonly related to chronic hypertension",
}


def build_prompt(
    pred_probs: dict[str, float],
    threshold: float = 0.5,
    ecg_stats: Optional[dict] = None,
    gradcam_regions: Optional[list[str]] = None,
) -> str:
    """Build a structured prompt for the LLM from model outputs.

    Args:
        pred_probs: Dict mapping class names to predicted probabilities.
        threshold: Probability threshold for considering a class positive.
        ecg_stats: Optional signal statistics (heart rate, duration, etc.).
        gradcam_regions: Optional list of descriptions of regions the model
            attended to (from Grad-CAM analysis).

    Returns:
        Formatted prompt string.
    """
    detected = {k: v for k, v in pred_probs.items() if v >= threshold}
    below = {k: v for k, v in pred_probs.items() if v < threshold}

    lines = [
        "You are a cardiology AI assistant. Based on the automated ECG "
        "analysis results below, provide a clear and concise clinical "
        "interpretation suitable for a healthcare professional. Do not "
        "provide a definitive diagnosis; instead frame findings as "
        "observations that warrant further clinical evaluation.",
        "",
        "## Automated Analysis Results",
        "",
    ]

    if detected:
        lines.append("**Detected abnormalities:**")
        for cls, prob in sorted(detected.items(), key=lambda x: -x[1]):
            desc = CLASS_DESCRIPTIONS.get(cls, cls)
            lines.append(f"- {cls} (confidence: {prob:.1%}): {desc}")
    else:
        lines.append("**No abnormalities detected above threshold.**")

    lines.append("")
    lines.append("**All class probabilities:**")
    for cls, prob in sorted(pred_probs.items(), key=lambda x: -x[1]):
        lines.append(f"- {cls}: {prob:.1%}")

    if ecg_stats:
        lines.append("")
        lines.append("**Signal statistics:**")
        for k, v in ecg_stats.items():
            lines.append(f"- {k}: {v}")

    if gradcam_regions:
        lines.append("")
        lines.append("**Model attention regions (from Grad-CAM):**")
        for region in gradcam_regions:
            lines.append(f"- {region}")

    lines.extend([
        "",
        "## Instructions",
        "1. Summarise the key findings in 2-3 sentences.",
        "2. Explain what each detected abnormality means clinically.",
        "3. If Grad-CAM attention regions are provided, comment on whether "
        "the model is focusing on clinically expected waveform features.",
        "4. Suggest appropriate follow-up actions.",
        "5. Include a brief disclaimer about automated screening limitations.",
    ])

    return "\n".join(lines)


def identify_gradcam_regions(
    cam: np.ndarray,
    fs: float = 500.0,
    attention_threshold: float = 0.6,
) -> list[str]:
    """Convert Grad-CAM heatmap into human-readable region descriptions.

    Identifies contiguous high-attention segments and describes their
    temporal location and approximate ECG waveform correspondence.

    Args:
        cam: Grad-CAM heatmap, shape (time,).
        fs: Sampling frequency.
        attention_threshold: Minimum attention value to consider.

    Returns:
        List of region descriptions.
    """
    high_attention = cam >= attention_threshold
    regions = []

    # Find contiguous segments above threshold
    in_region = False
    start = 0
    for i in range(len(high_attention)):
        if high_attention[i] and not in_region:
            start = i
            in_region = True
        elif not high_attention[i] and in_region:
            end = i
            t_start = start / fs
            t_end = end / fs
            duration_ms = (end - start) / fs * 1000
            peak_attention = cam[start:end].max()

            desc = (
                f"High attention at {t_start:.2f}-{t_end:.2f}s "
                f"(duration: {duration_ms:.0f}ms, "
                f"peak intensity: {peak_attention:.2f})"
            )
            regions.append(desc)
            in_region = False

    # Handle region that extends to end of signal
    if in_region:
        t_start = start / fs
        t_end = len(cam) / fs
        peak_attention = cam[start:].max()
        desc = (
            f"High attention at {t_start:.2f}-{t_end:.2f}s "
            f"(peak intensity: {peak_attention:.2f})"
        )
        regions.append(desc)

    return regions


def generate_explanation(
    pred_probs: dict[str, float],
    cam: Optional[np.ndarray] = None,
    fs: float = 500.0,
    ecg_stats: Optional[dict] = None,
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
    base_url: Optional[str] = None,
) -> str:
    """Generate a natural language explanation of the ECG analysis.

    Uses an OpenAI-compatible API. Set the OPENAI_API_KEY environment
    variable or pass api_key directly. The base_url parameter allows
    pointing to alternative providers.

    Args:
        pred_probs: Dict mapping class names to predicted probabilities.
        cam: Optional Grad-CAM heatmap for attention region analysis.
        fs: Sampling frequency of the ECG signal.
        ecg_stats: Optional signal statistics.
        api_key: API key (falls back to OPENAI_API_KEY env var).
        model: LLM model identifier.
        base_url: Optional base URL for API-compatible providers.

    Returns:
        Generated explanation text.
    """
    gradcam_regions = None
    if cam is not None:
        gradcam_regions = identify_gradcam_regions(cam, fs)

    prompt = build_prompt(pred_probs, ecg_stats=ecg_stats,
                          gradcam_regions=gradcam_regions)

    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        # Return the prompt itself as a fallback when no API key is set,
        # so the system remains functional for demo and testing
        return (
            "[LLM API key not configured. Showing raw analysis prompt.]\n\n"
            + prompt
        )

    try:
        from openai import OpenAI
        client = OpenAI(api_key=key, base_url=base_url)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500,
        )
        return response.choices[0].message.content
    except ImportError:
        return (
            "[openai package not installed. Run: pip install openai]\n\n"
            + prompt
        )
    except Exception as e:
        return f"[LLM API error: {e}]\n\n" + prompt
