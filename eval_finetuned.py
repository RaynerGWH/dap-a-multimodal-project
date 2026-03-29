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

with open(f"{config.SAVE_DIR}/label_encoder.pkl", 'rb') as f:
    le = pickle.load(f)
CATEGORIES = list(le.classes_)
category_str = ", ".join(CATEGORIES)

print("Loading base model + LoRA adapter...")
base_model = Qwen3VLForConditionalGeneration.from_pretrained(
    config.QWEN3_MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto",
)
model = PeftModel.from_pretrained(base_model, config.QWEN3_FINETUNE_DIR)
model.eval()
processor = AutoProcessor.from_pretrained(config.QWEN3_MODEL_NAME)

dev_df = load_data(config.DEV_JSON, config.IMAGE_DIR)
dev_df, _ = get_label_encoder(dev_df)
dev_df = dev_df.sample(n=200, random_state=config.RANDOM_SEED).reset_index(drop=True)

def predict(image_path, headline):
    messages = [{"role": "user", "content": [
        {"type": "image", "image": Image.open(image_path).convert("RGB")},
        {"type": "text", "text": f"Classify this news article into one of: {category_str}.\nHeadline: \"{headline}\"\nRespond with ONLY the category name."},
    ]}]
    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
    inputs = inputs.to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out)]
    return processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip().lower()

def match_to_category(pred):
    pred = pred.strip().lower()
    for cat in CATEGORIES:
        if pred == cat.lower(): return cat
    for cat in CATEGORIES:
        if cat.lower() in pred: return cat
    return pred

y_true, y_pred, raw_preds = [], [], []
for idx in tqdm(range(len(dev_df)), desc="Fine-tuned eval"):
    row = dev_df.iloc[idx]
    img_path = os.path.join(config.IMAGE_DIR, row['image_filename'])
    try:
        raw = predict(img_path, row['headline'])
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
print(f"Qwen3-VL Fine-Tuned Results")
print(f"{'='*50}")
print(f"Total samples: {len(y_true)}")
print(f"Match rate: {sum(valid_mask)/len(valid_mask)*100:.1f}%")
print(f"Accuracy: {acc:.4f} ({acc*100:.1f}%)")

if valid_true:
    print("\nClassification Report:")
    print(classification_report(valid_true, valid_pred, zero_division=0))

results = {
    "model": "Qwen3-VL-2B + LoRA",
    "mode": "fine-tuned",
    "accuracy": acc,
    "match_rate": sum(valid_mask)/len(valid_mask)*100,
    "predictions": [{"true": t, "predicted": p, "raw": r} for t, p, r in zip(y_true, y_pred, raw_preds)]
}
with open(f"{config.RESULTS_DIR}/qwen3_finetuned_results.json", 'w') as f:
    json.dump(results, f, indent=2)
print(f"Results saved to {config.RESULTS_DIR}/qwen3_finetuned_results.json")
