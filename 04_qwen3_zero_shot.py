"""
Step 4: Qwen3-VL Zero-Shot Evaluation

No training — just prompt the model with image + headline and see if it classifies correctly.
This tests whether a pretrained VLM can match your task-specific models out of the box.
"""

import torch
import json
import os
import pickle
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm
from PIL import Image

import config
from data_utils import load_data, get_label_encoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================================
# Load label encoder (to get class names)
# ============================================================
with open(f"{config.SAVE_DIR}/label_encoder.pkl", 'rb') as f:
    le = pickle.load(f)

CATEGORIES = list(le.classes_)
print(f"Categories ({len(CATEGORIES)}): {CATEGORIES}")

# ============================================================
# Load Qwen3-VL
# ============================================================
print(f"\nLoading {config.QWEN3_MODEL_NAME}...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    config.QWEN3_MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(config.QWEN3_MODEL_NAME)
model.eval()
print("Model loaded!")

# ============================================================
# Load test/dev data
# ============================================================
# Use dev set for evaluation (or test set if you prefer)
df = load_data(config.DEV_JSON, config.IMAGE_DIR)
df, _ = get_label_encoder(df)

# Use a subset if the full set is too slow (each sample takes a few seconds)
MAX_SAMPLES = 200  # Set to None to run on full dev set
if MAX_SAMPLES and len(df) > MAX_SAMPLES:
    df = df.sample(n=MAX_SAMPLES, random_state=config.RANDOM_SEED).reset_index(drop=True)
    print(f"Using {MAX_SAMPLES} samples for evaluation")

# ============================================================
# Build prompt
# ============================================================
def build_prompt(headline):
    category_str = ", ".join(CATEGORIES)
    return (
        f"You are a news article classifier. Given the news image and headline below, "
        f"classify the article into exactly one of these categories: {category_str}.\n\n"
        f"Headline: \"{headline}\"\n\n"
        f"Respond with ONLY the category name. No explanation, no punctuation, just the category."
    )

# ============================================================
# Run inference
# ============================================================
def predict(image_path, headline):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": Image.open(image_path).convert("RGB")},
                {"type": "text", "text": build_prompt(headline)},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,  # greedy for consistency
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip().lower()

    return output


def match_to_category(prediction):
    """Match the model's free-text output to one of our known categories."""
    prediction = prediction.strip().lower()

    # Exact match
    for cat in CATEGORIES:
        if prediction == cat.lower():
            return cat

    # Partial match (prediction contains category name)
    for cat in CATEGORIES:
        if cat.lower() in prediction:
            return cat

    # No match — return raw prediction
    return prediction


# ============================================================
# Evaluate
# ============================================================
y_true = []
y_pred = []
raw_preds = []

for idx in tqdm(range(len(df)), desc="Qwen3 Zero-Shot"):
    row = df.iloc[idx]
    img_path = os.path.join(config.IMAGE_DIR, row['image_filename'])
    true_label = row['section']

    try:
        raw_pred = predict(img_path, row['headline'])
        matched_pred = match_to_category(raw_pred)
    except Exception as e:
        print(f"  Error on sample {idx}: {e}")
        raw_pred = "error"
        matched_pred = "unknown"

    y_true.append(true_label)
    y_pred.append(matched_pred)
    raw_preds.append(raw_pred)

# ============================================================
# Results
# ============================================================
# Only evaluate on samples where we got a valid category match
valid_mask = [p in CATEGORIES for p in y_pred]
valid_true = [t for t, v in zip(y_true, valid_mask) if v]
valid_pred = [p for p, v in zip(y_pred, valid_mask) if v]

match_rate = sum(valid_mask) / len(valid_mask) * 100
acc = accuracy_score(valid_true, valid_pred) if valid_true else 0

print(f"\n{'='*50}")
print(f"Qwen3-VL Zero-Shot Results")
print(f"{'='*50}")
print(f"Total samples: {len(y_true)}")
print(f"Category match rate: {match_rate:.1f}% ({sum(valid_mask)}/{len(valid_mask)})")
print(f"Accuracy (on matched): {acc:.4f} ({acc*100:.1f}%)")

if valid_true:
    # Filter to only categories that appear in predictions
    report = classification_report(valid_true, valid_pred, zero_division=0)
    print(f"\nClassification Report:")
    print(report)

# Show some unmatched predictions
unmatched = [(t, r) for t, r, v in zip(y_true, raw_preds, valid_mask) if not v]
if unmatched:
    print(f"\nUnmatched predictions (first 10):")
    for true, raw in unmatched[:10]:
        print(f"  True: {true:20} | Predicted: {raw}")

# Save results
results = {
    "model": config.QWEN3_MODEL_NAME,
    "mode": "zero-shot",
    "total_samples": len(y_true),
    "match_rate": match_rate,
    "accuracy": acc,
    "predictions": [
        {"true": t, "predicted": p, "raw": r}
        for t, p, r in zip(y_true, y_pred, raw_preds)
    ],
}
with open(os.path.join(config.RESULTS_DIR, "qwen3_zero_shot_results.json"), 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to {config.RESULTS_DIR}/qwen3_zero_shot_results.json")
