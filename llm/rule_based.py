"""
Rule-based ECG explanation baseline.

Generates template-driven explanations from model predictions using
deterministic if/else logic. This serves as a comparison to the
LLM-based explanation module to justify the added complexity of
natural language generation.
"""
import numpy as np
from typing import Optional


CLASS_TEMPLATES = {
    "NORM": {
        "finding": "normal sinus rhythm",
        "detail": "No significant abnormalities detected in the ECG waveform.",
        "followup": "Routine follow-up as clinically indicated.",
    },
    "MI": {
        "finding": "myocardial infarction pattern",
        "detail": "Features suggestive of ischaemic injury to the myocardium, "
                  "such as pathological Q waves or ST-segment changes.",
        "followup": "Urgent cardiology referral recommended. Consider serial "
                    "troponin measurements and 12-lead ECG comparison.",
    },
    "STTC": {
        "finding": "ST-segment/T-wave changes",
        "detail": "Repolarisation abnormalities detected, which may indicate "
                  "myocardial ischaemia, electrolyte imbalance, or drug effect.",
        "followup": "Correlate with clinical symptoms. Consider repeat ECG "
                    "and electrolyte panel.",
    },
    "CD": {
        "finding": "conduction disturbance",
        "detail": "Abnormal electrical conduction pattern, possibly a bundle "
                  "branch block or atrioventricular conduction delay.",
        "followup": "Consider echocardiography to assess structural correlates. "
                    "Cardiology review if symptomatic.",
    },
    "HYP": {
        "finding": "ventricular hypertrophy",
        "detail": "Voltage criteria or repolarisation pattern suggestive of "
                  "increased myocardial wall thickness.",
        "followup": "Blood pressure assessment and echocardiography recommended "
                    "to evaluate chamber dimensions.",
    },
}

DISCLAIMER = (
    "DISCLAIMER: This is an automated screening result and does not "
    "constitute a medical diagnosis. Clinical correlation and specialist "
    "review are required."
)


def generate_rule_based_explanation(
    pred_probs: dict[str, float],
    threshold: float = 0.5,
    gradcam_regions: Optional[list[str]] = None,
) -> str:
    """Generate a template-based explanation from model predictions.

    Args:
        pred_probs: Dict mapping class names to predicted probabilities.
        threshold: Decision threshold for flagging abnormalities.
        gradcam_regions: Optional Grad-CAM region descriptions (ignored
            by rule-based system, included for interface compatibility).

    Returns:
        Formatted explanation string.
    """
    detected = {k: v for k, v in pred_probs.items() if v >= threshold}
    lines = ["ECG SCREENING REPORT", "=" * 40, ""]

    if not detected:
        lines.append("No abnormalities detected above reporting threshold.")
        lines.append("")
        lines.append("All class probabilities are below 0.5, suggesting a "
                      "likely normal recording. However, borderline values "
                      "may warrant clinical review.")
    elif len(detected) == 1 and "NORM" in detected:
        t = CLASS_TEMPLATES["NORM"]
        lines.append(f"Finding: {t['finding'].capitalize()} "
                      f"(confidence: {detected['NORM']:.0%})")
        lines.append(f"Detail: {t['detail']}")
        lines.append(f"Follow-up: {t['followup']}")
    else:
        abnormals = {k: v for k, v in detected.items() if k != "NORM"}
        if abnormals:
            lines.append("ABNORMALITIES DETECTED:")
            lines.append("")
            for cls, prob in sorted(abnormals.items(), key=lambda x: -x[1]):
                t = CLASS_TEMPLATES[cls]
                lines.append(f"- {t['finding'].capitalize()} "
                              f"(confidence: {prob:.0%})")
                lines.append(f"  Detail: {t['detail']}")
                lines.append(f"  Follow-up: {t['followup']}")
                lines.append("")

    # Probability summary
    lines.append("")
    lines.append("Class probabilities:")
    for cls, prob in sorted(pred_probs.items(), key=lambda x: -x[1]):
        lines.append(f"  {cls}: {prob:.1%}")

    lines.extend(["", DISCLAIMER])
    return "\n".join(lines)
