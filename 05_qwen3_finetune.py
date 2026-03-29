"""
Step 5: Fine-tune Qwen3-VL with LoRA on your dataset, then evaluate.

Uses QLoRA (4-bit quantization + LoRA) to fit on a single GPU.
After fine-tuning, evaluates on the same dev set as zero-shot for comparison.
"""

import torch
import json
import os
import pickle
import copy
from PIL import Image
from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training
from datasets import Dataset as HFDataset
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm

import config
from data_utils import load_data, get_label_encoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================================
# Load label encoder
# ============================================================
with open(f"{config.SAVE_DIR}/label_encoder.pkl", 'rb') as f:
    le = pickle.load(f)

CATEGORIES = list(le.classes_)

# ============================================================
# Load model with 4-bit quantization (QLoRA)
# ============================================================
print(f"\nLoading {config.QWEN3_MODEL_NAME} with 4-bit quantization...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = Qwen3VLForConditionalGeneration.from_pretrained(
    config.QWEN3_MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

processor = AutoProcessor.from_pretrained(config.QWEN3_MODEL_NAME)

# Prepare for QLoRA
model = prepare_model_for_kbit_training(model)
model.gradient_checkpointing_enable()

# ============================================================
# Apply LoRA
# ============================================================
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ============================================================
# Prepare training data
# ============================================================
print("\nPreparing training data...")
df = load_data(config.TRAIN_JSON, config.IMAGE_DIR)
df, _ = get_label_encoder(df)

# Limit training data if needed (LoRA converges fast, don't need everything)
MAX_TRAIN = 2000  # Set to None for full dataset
if MAX_TRAIN and len(df) > MAX_TRAIN:
    df = df.sample(n=MAX_TRAIN, random_state=config.RANDOM_SEED).reset_index(drop=True)
    print(f"Using {MAX_TRAIN} training samples")

category_str = ", ".join(CATEGORIES)


def format_for_training(row):
    """Format a single sample as a conversation for the VLM."""
    img_path = os.path.join(config.IMAGE_DIR, row['image_filename'])
    image = Image.open(img_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {
                    "type": "text",
                    "text": (
                        f"Classify this news article into one of: {category_str}.\n"
                        f"Headline: \"{row['headline']}\"\n"
                        f"Respond with ONLY the category name."
                    ),
                },
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": row['section']}],
        },
    ]

    return messages


class Qwen3Dataset(torch.utils.data.Dataset):
    """Dataset that formats samples for Qwen3-VL fine-tuning."""

    def __init__(self, df, processor, max_length=512):
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        messages = format_for_training(row)

        # Process with the chat template
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
        )

        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)

        # For causal LM, labels = input_ids (shifted internally by the model)
        labels = input_ids.clone()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def collate_fn(batch):
    """Pad batch to same length."""
    max_len = max(item["input_ids"].size(0) for item in batch)

    input_ids = []
    attention_masks = []
    labels = []

    for item in batch:
        pad_len = max_len - item["input_ids"].size(0)
        input_ids.append(
            torch.cat([item["input_ids"], torch.full((pad_len,), processor.tokenizer.pad_token_id or 0)])
        )
        attention_masks.append(
            torch.cat([item["attention_mask"], torch.zeros(pad_len, dtype=torch.long)])
        )
        label = item["labels"]
        labels.append(
            torch.cat([label, torch.full((pad_len,), -100)])  # -100 = ignore in loss
        )

    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_masks),
        "labels": torch.stack(labels),
    }


# ============================================================
# Train
# ============================================================
print("\nCreating dataset...")
train_dataset = Qwen3Dataset(df, processor)

training_args = TrainingArguments(
    output_dir=config.QWEN3_FINETUNE_DIR,
    num_train_epochs=config.QWEN3_FINETUNE_EPOCHS,
    per_device_train_batch_size=config.QWEN3_FINETUNE_BATCH_SIZE,
    gradient_accumulation_steps=4,
    learning_rate=config.QWEN3_FINETUNE_LR,
    warmup_ratio=0.1,
    logging_steps=10,
    save_strategy="epoch",
    bf16=True,
    dataloader_pin_memory=False,
    remove_unused_columns=False,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=collate_fn,
)

print("\nStarting fine-tuning...")
trainer.train()

# Save LoRA adapter
trainer.save_model(config.QWEN3_FINETUNE_DIR)
processor.save_pretrained(config.QWEN3_FINETUNE_DIR)
print(f"\nFine-tuned model saved to {config.QWEN3_FINETUNE_DIR}")

# ============================================================
# Evaluate fine-tuned model
# ============================================================
print("\n" + "="*50)
print("Evaluating fine-tuned Qwen3-VL...")
print("="*50)

# Reload for inference (clean state)
del model, trainer
torch.cuda.empty_cache()

# Load base + LoRA adapter
base_model = Qwen3VLForConditionalGeneration.from_pretrained(
    config.QWEN3_MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model = PeftModel.from_pretrained(base_model, config.QWEN3_FINETUNE_DIR)
model.eval()

processor = AutoProcessor.from_pretrained(config.QWEN3_MODEL_NAME)

# Load dev set
dev_df = load_data(config.DEV_JSON, config.IMAGE_DIR)
dev_df, _ = get_label_encoder(dev_df)

MAX_EVAL = 200  # Match zero-shot evaluation size
if MAX_EVAL and len(dev_df) > MAX_EVAL:
    dev_df = dev_df.sample(n=MAX_EVAL, random_state=config.RANDOM_SEED).reset_index(drop=True)


def predict(image_path, headline):
    """Same inference function as zero-shot."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": Image.open(image_path).convert("RGB")},
                {
                    "type": "text",
                    "text": (
                        f"Classify this news article into one of: {category_str}.\n"
                        f"Headline: \"{headline}\"\n"
                        f"Respond with ONLY the category name."
                    ),
                },
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=20, do_sample=False)

    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, generated_ids)]
    return processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip().lower()


def match_to_category(prediction):
    prediction = prediction.strip().lower()
    for cat in CATEGORIES:
        if prediction == cat.lower():
            return cat
    for cat in CATEGORIES:
        if cat.lower() in prediction:
            return cat
    return prediction


y_true, y_pred, raw_preds = [], [], []

for idx in tqdm(range(len(dev_df)), desc="Fine-tuned inference"):
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

# Results
valid_mask = [p in CATEGORIES for p in y_pred]
valid_true = [t for t, v in zip(y_true, valid_mask) if v]
valid_pred = [p for p, v in zip(y_pred, valid_mask) if v]

match_rate = sum(valid_mask) / len(valid_mask) * 100
acc = accuracy_score(valid_true, valid_pred) if valid_true else 0

print(f"\n{'='*50}")
print(f"Qwen3-VL Fine-Tuned Results")
print(f"{'='*50}")
print(f"Total samples: {len(y_true)}")
print(f"Category match rate: {match_rate:.1f}%")
print(f"Accuracy (on matched): {acc:.4f} ({acc*100:.1f}%)")

if valid_true:
    report = classification_report(valid_true, valid_pred, zero_division=0)
    print(f"\nClassification Report:")
    print(report)

# Save
results = {
    "model": f"{config.QWEN3_MODEL_NAME} + LoRA",
    "mode": "fine-tuned",
    "total_samples": len(y_true),
    "match_rate": match_rate,
    "accuracy": acc,
    "predictions": [
        {"true": t, "predicted": p, "raw": r}
        for t, p, r in zip(y_true, y_pred, raw_preds)
    ],
}
with open(os.path.join(config.RESULTS_DIR, "qwen3_finetuned_results.json"), 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to {config.RESULTS_DIR}/qwen3_finetuned_results.json")
