"""
Step 5 (fixed): Fine-tune Qwen3-VL with LoRA
The key fix: properly mask the prompt tokens in labels so the model
only learns to predict the category, not reconstruct the entire prompt.
"""

import torch
import json
import os
import pickle
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from data_utils import load_data, get_label_encoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load label encoder
with open(f"{config.SAVE_DIR}/label_encoder.pkl", 'rb') as f:
    le = pickle.load(f)
CATEGORIES = list(le.classes_)
category_str = ", ".join(CATEGORIES)

# Load model
print(f"\nLoading {config.QWEN3_MODEL_NAME}...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    config.QWEN3_MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
processor = AutoProcessor.from_pretrained(config.QWEN3_MODEL_NAME)

# LoRA
model.gradient_checkpointing_enable()
model.enable_input_require_grads()

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Load data
print("\nPreparing training data...")
df = load_data(config.TRAIN_JSON, config.IMAGE_DIR)
df, _ = get_label_encoder(df)

MAX_TRAIN = 2000
if MAX_TRAIN and len(df) > MAX_TRAIN:
    df = df.sample(n=MAX_TRAIN, random_state=config.RANDOM_SEED).reset_index(drop=True)
    print(f"Using {MAX_TRAIN} training samples")


def make_prompt(headline):
    return (
        f"Classify this news article into one of: {category_str}.\n"
        f"Headline: \"{headline}\"\n"
        f"Respond with ONLY the category name."
    )


# Training loop (manual, not Trainer — more control over labels)
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=2e-5,
    weight_decay=0.01,
)

NUM_EPOCHS = 3
BATCH_SIZE = 1  # VLM with images, one at a time
GRAD_ACCUM = 8  # effective batch size = 8
LOG_EVERY = 50

model.train()
global_step = 0
total_loss = 0

for epoch in range(NUM_EPOCHS):
    print(f"\n{'='*50}")
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
    print(f"{'='*50}")

    # Shuffle
    epoch_df = df.sample(frac=1, random_state=config.RANDOM_SEED + epoch).reset_index(drop=True)
    epoch_loss = 0
    epoch_steps = 0

    for idx in tqdm(range(len(epoch_df)), desc=f"Epoch {epoch+1}"):
        row = epoch_df.iloc[idx]
        img_path = os.path.join(config.IMAGE_DIR, row['image_filename'])
        headline = row['headline']
        label = row['section']

        try:
            image = Image.open(img_path).convert("RGB")

            # Build the PROMPT part only (user message)
            prompt_messages = [{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": make_prompt(headline)},
            ]}]

            # Get prompt tokens (without generation prompt, we add assistant response manually)
            prompt_inputs = processor.apply_chat_template(
                prompt_messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            prompt_len = prompt_inputs["input_ids"].shape[1]

            # Build FULL conversation (prompt + answer)
            full_messages = [
                {"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": make_prompt(headline)},
                ]},
                {"role": "assistant", "content": [
                    {"type": "text", "text": label},
                ]},
            ]

            full_inputs = processor.apply_chat_template(
                full_messages,
                tokenize=True,
                add_generation_prompt=False,
                return_dict=True,
                return_tensors="pt",
            )

            input_ids = full_inputs["input_ids"].to(device)
            attention_mask = full_inputs["attention_mask"].to(device)

            # Create labels: mask prompt tokens with -100, only train on the answer
            labels = input_ids.clone()
            labels[:, :prompt_len] = -100

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss / GRAD_ACCUM
            loss.backward()

            epoch_loss += outputs.loss.item()
            total_loss += outputs.loss.item()
            epoch_steps += 1
            global_step += 1

            # Gradient accumulation step
            if (idx + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            if global_step % LOG_EVERY == 0:
                avg = total_loss / global_step
                print(f"  Step {global_step} | Loss: {outputs.loss.item():.4f} | Avg: {avg:.4f}")

        except Exception as e:
            print(f"  Skip sample {idx}: {e}")
            continue

    avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
    print(f"  Epoch {epoch+1} avg loss: {avg_epoch_loss:.4f}")

# Save
model.save_pretrained(config.QWEN3_FINETUNE_DIR)
processor.save_pretrained(config.QWEN3_FINETUNE_DIR)
print(f"\nModel saved to {config.QWEN3_FINETUNE_DIR}")
print("Now run: python eval_finetuned.py")
