"""
Test Qwen3.5-9B with our ECG explanation prompt (no fine-tuning).
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("Loading Qwen3.5-9B-NVFP4 (pre-quantized for Blackwell)...")
model_name = "AxionML/Qwen3.5-9B-NVFP4"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    trust_remote_code=True,
)
print(f"Model loaded. GPU memory: {torch.cuda.memory_allocated()/1e9:.1f} GB")

# Test cases matching our eval_llm.py scenarios
test_cases = [
    {
        "name": "Clear normal",
        "detected": "- NORM (confidence: 92.0%): Normal sinus rhythm with no significant abnormalities",
    },
    {
        "name": "Clear MI",
        "detected": "- MI (confidence: 95.0%): Myocardial infarction pattern, indicating possible ischaemic damage to the heart muscle",
    },
    {
        "name": "Multi-label STTC+CD",
        "detected": (
            "- STTC (confidence: 88.0%): ST-segment or T-wave changes, often associated with myocardial ischaemia\n"
            "- CD (confidence: 82.0%): Conduction disturbance, such as bundle branch block"
        ),
    },
    {
        "name": "Uncertain (none detected)",
        "detected": None,
    },
]

PROMPT_TEMPLATE = """You are a cardiology AI assistant. Based on the automated ECG analysis results below, provide a clear and concise clinical interpretation.

## Automated Analysis Results

{findings}

## Instructions
1. Summarise the key findings in 2-3 sentences.
2. ONLY discuss abnormalities listed above as 'Detected abnormalities'. Do NOT mention, speculate about, or discuss any class that was not detected. If no abnormalities were detected, simply state the recording appears normal.
3. Suggest appropriate follow-up actions.
4. ALWAYS include a brief disclaimer about automated screening limitations."""

for case in test_cases:
    if case["detected"]:
        findings = f"**Detected abnormalities:**\n{case['detected']}"
    else:
        findings = "**No abnormalities detected above threshold.**"

    prompt = PROMPT_TEMPLATE.format(findings=findings)
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=300, temperature=0.3,
            do_sample=True, top_p=0.9,
        )

    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    print(f"\n{'='*60}")
    print(f"Case: {case['name']}")
    print(f"{'='*60}")
    print(response)
    print()
