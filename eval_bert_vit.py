import torch
import pickle
from torch.utils.data import DataLoader
from transformers import (
    BertForSequenceClassification, BertTokenizer,
    ViTForImageClassification, ViTImageProcessor
)
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm

import config
from data_utils import load_data, get_label_encoder, get_train_val_split, TextDataset, ImageDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load label encoder
with open(f"{config.SAVE_DIR}/label_encoder.pkl", 'rb') as f:
    le = pickle.load(f)

# Load data
df = load_data(config.TRAIN_JSON, config.IMAGE_DIR)
df, _ = get_label_encoder(df)
_, val_df = get_train_val_split(df)

# ============================================================
# BERT Evaluation
# ============================================================
print("=" * 50)
print("BERT Evaluation")
print("=" * 50)

tokenizer = BertTokenizer.from_pretrained(config.BERT_SAVE_DIR)
bert_model = BertForSequenceClassification.from_pretrained(config.BERT_SAVE_DIR)
bert_model.to(device)
bert_model.eval()

val_text_dataset = TextDataset(val_df, tokenizer, max_length=config.MAX_TEXT_LENGTH)
val_text_loader = DataLoader(val_text_dataset, batch_size=config.BERT_BATCH_SIZE, shuffle=False)

bert_true, bert_pred = [], []
with torch.no_grad():
    for batch in tqdm(val_text_loader, desc="BERT inference"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label']

        outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1).cpu()
        bert_true.extend(labels.tolist())
        bert_pred.extend(preds.tolist())

bert_acc = accuracy_score(bert_true, bert_pred)
print(f"\nBERT Accuracy: {bert_acc:.4f} ({bert_acc*100:.1f}%)")
print("\nClassification Report:")
print(classification_report(bert_true, bert_pred, target_names=le.classes_, zero_division=0))

del bert_model
torch.cuda.empty_cache()

# ============================================================
# ViT Evaluation
# ============================================================
print("=" * 50)
print("ViT Evaluation")
print("=" * 50)

processor = ViTImageProcessor.from_pretrained(config.VIT_SAVE_DIR)
vit_model = ViTForImageClassification.from_pretrained(config.VIT_SAVE_DIR)
vit_model.to(device)
vit_model.eval()

val_img_dataset = ImageDataset(val_df, config.IMAGE_DIR, processor)
val_img_loader = DataLoader(val_img_dataset, batch_size=config.VIT_BATCH_SIZE, shuffle=False, num_workers=4)

vit_true, vit_pred = [], []
with torch.no_grad():
    for batch in tqdm(val_img_loader, desc="ViT inference"):
        pixel_values = batch['pixel_values'].to(device)
        labels = batch['label']

        outputs = vit_model(pixel_values=pixel_values)
        preds = torch.argmax(outputs.logits, dim=1).cpu()
        vit_true.extend(labels.tolist())
        vit_pred.extend(preds.tolist())

vit_acc = accuracy_score(vit_true, vit_pred)
print(f"\nViT Accuracy: {vit_acc:.4f} ({vit_acc*100:.1f}%)")
print("\nClassification Report:")
print(classification_report(vit_true, vit_pred, target_names=le.classes_, zero_division=0))

# ============================================================
# Save results
# ============================================================
import json

results = {
    "bert": {
        "accuracy": bert_acc,
        "report": classification_report(bert_true, bert_pred, target_names=le.classes_, output_dict=True, zero_division=0)
    },
    "vit": {
        "accuracy": vit_acc,
        "report": classification_report(vit_true, vit_pred, target_names=le.classes_, output_dict=True, zero_division=0)
    }
}

with open(f"{config.RESULTS_DIR}/bert_vit_results.json", 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to {config.RESULTS_DIR}/bert_vit_results.json")
