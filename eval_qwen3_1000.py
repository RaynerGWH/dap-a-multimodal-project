"""
Evaluate both Qwen3-VL zero-shot and fine-tuned on 1000 samples.
Run after starting the pod and installing deps.
"""

import torch
import json
import os
import pickle
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm
import config
from data_utils import load_data, get_label_encoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

NUM_SAMPLES = 1000

# Load label encoder
with open(f"{config.SAVE_DIR}/label_encoder.pkl", 'rb') as f:
    le = pickle.load(f)
CATEGORIES = list(le.classes_)
category_str = ", ".join(CATEGORIES)

# Load dev data
dev_df = load_data(config.DEV_JSON, config.IMAGE_DIR)
dev_df, _ = get_label_encoder(dev_df)
dev_df = dev_df.sample(n=NUM_SAMPLES, random_state=config.RANDOM_SEED).reset_index(drop=True)
print(f"Evaluating on {len(dev_df)} samples")

# Shared functions
def predict(model, processor, image_path, headline):
    messages = [{"role": "user", "content": [
        {"type": "image", "image": Image.open(image_path).convert("RGB")},
        {"type": "text", "text": f"Classify this news article into one of: {category_str}.\nHeadline: \"{headline}\"\nRespond with ONLY the category name."},
    ]}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out)]
    return processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip().lower()

def match_to_category(pred):
    pred = pred.strip().lower()
    for cat in CATEGORIES:
        if pred == cat.lower():
            return cat
    for cat in CATEGORIES:
        if cat.lower() in pred:
            return cat
    return pred

def evaluate_model(model, processor, label, dev_df):
    y_true, y_pred, raw_preds = [], [], []
    for idx in tqdm(range(len(dev_df)), desc=f"{label} eval"):
        row = dev_df.iloc[idx]
        img_path = os.path.join(config.IMAGE_DIR, row['image_filename'])
        try:
            raw = predict(model, processor, img_path, row['headline'])
            matched = match_to_category(raw)
        except Exception as e:
            print(f"  Error on sample {idx}: {e}")
            raw, matched = "error", "unknown"
        y_true.append(row['section'])
        y_pred.append(matched)
        raw_preds.append(raw)

    valid_mask = [p in CATEGORIES for p in y_pred]
    valid_true = [t for t, v in zip(y_true, valid_mask) if v]
    valid_pred = [p for p, v in zip(y_pred, valid_mask) if v]
    acc = accuracy_score(valid_true, valid_pred) if valid_true else 0

    print(f"\n{'='*50}")
    print(f"{label} Results")
    print(f"{'='*50}")
    print(f"Total samples: {len(y_true)}")
    print(f"Match rate: {sum(valid_mask)/len(valid_mask)*100:.1f}%")
    print(f"Accuracy: {acc:.4f} ({acc*100:.1f}%)")

    if valid_true:
        report = classification_report(valid_true, valid_pred, zero_division=0)
        print(f"\nClassification Report:")
        print(report)
    else:
        report = "No valid predictions"

    return {
        "model": label,
        "total_samples": len(y_true),
        "match_rate": sum(valid_mask)/len(valid_mask)*100,
        "accuracy": acc,
        "report": report,
        "predictions": [{"true": t, "predicted": p, "raw": r} for t, p, r in zip(y_true, y_pred, raw_preds)]
    }

# ============================================================
# 1. Zero-shot evaluation
# ============================================================
print("\nLoading Qwen3-VL base model...")
base_model = Qwen3VLForConditionalGeneration.from_pretrained(
    config.QWEN3_MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto",
)
base_model.eval()
processor = AutoProcessor.from_pretrained(config.QWEN3_MODEL_NAME)

zs_results = evaluate_model(base_model, processor, "Qwen3-VL Zero-Shot", dev_df)

with open(f"{config.RESULTS_DIR}/qwen3_zero_shot_1000.json", 'w') as f:
    json.dump(zs_results, f, indent=2)

# Free memory before loading fine-tuned
del base_model
torch.cuda.empty_cache()

# ============================================================
# 2. Fine-tuned evaluation
# ============================================================
print("\nLoading Qwen3-VL fine-tuned model...")
base_model = Qwen3VLForConditionalGeneration.from_pretrained(
    config.QWEN3_MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto",
)
ft_model = PeftModel.from_pretrained(base_model, config.QWEN3_FINETUNE_DIR)
ft_model.eval()
processor = AutoProcessor.from_pretrained(config.QWEN3_MODEL_NAME)

ft_results = evaluate_model(ft_model, processor, "Qwen3-VL Fine-Tuned", dev_df)

with open(f"{config.RESULTS_DIR}/qwen3_finetuned_1000.json", 'w') as f:
    json.dump(ft_results, f, indent=2)

# ============================================================
# Summary
# ============================================================
print(f"\n{'='*50}")
print("SUMMARY")
print(f"{'='*50}")
print(f"Zero-shot:  {zs_results['accuracy']*100:.1f}% ({NUM_SAMPLES} samples)")
print(f"Fine-tuned: {ft_results['accuracy']*100:.1f}% ({NUM_SAMPLES} samples)")
print(f"\nResults saved to {config.RESULTS_DIR}/")
