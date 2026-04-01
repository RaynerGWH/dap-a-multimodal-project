"""
Evaluate BERT, ViT, and cross-attention with per-sample predictions saved.
Needed for the category merging analysis.
"""

import torch
import pickle
import json
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    BertForSequenceClassification, BertTokenizer,
    ViTForImageClassification, ViTImageProcessor
)
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm

import config
from data_utils import load_data, get_label_encoder, get_train_val_split, TextDataset, ImageDataset, MultimodalDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open(f"{config.SAVE_DIR}/label_encoder.pkl", 'rb') as f:
    le = pickle.load(f)

df = load_data(config.TRAIN_JSON, config.IMAGE_DIR)
df, _ = get_label_encoder(df)
_, val_df = get_train_val_split(df)


# ============================================================
# Cross-attention model definition
# ============================================================
class CrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim=768, num_heads=8, num_classes=24, dropout=0.3):
        super().__init__()
        self.text_to_image_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.image_to_text_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.text_norm = nn.LayerNorm(embed_dim)
        self.image_norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, 512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    def forward(self, text_seq, image_seq, text_mask=None):
        text_attended, _ = self.text_to_image_attn(query=text_seq, key=image_seq, value=image_seq)
        text_attended = self.text_norm(text_attended + text_seq)
        image_attended, _ = self.image_to_text_attn(
            query=image_seq, key=text_seq, value=text_seq,
            key_padding_mask=(~text_mask.bool()) if text_mask is not None else None
        )
        image_attended = self.image_norm(image_attended + image_seq)
        text_pooled = text_attended.mean(dim=1)
        image_pooled = image_attended.mean(dim=1)
        combined = torch.cat([text_pooled, image_pooled], dim=1)
        return self.classifier(combined)


# ============================================================
# BERT
# ============================================================
print("=" * 50)
print("BERT Evaluation (with per-sample predictions)")
print("=" * 50)

tokenizer = BertTokenizer.from_pretrained(config.BERT_SAVE_DIR)
bert_model = BertForSequenceClassification.from_pretrained(config.BERT_SAVE_DIR).to(device).eval()

val_text = TextDataset(val_df, tokenizer, max_length=config.MAX_TEXT_LENGTH)
val_text_loader = DataLoader(val_text, batch_size=config.BERT_BATCH_SIZE, shuffle=False)

bert_true, bert_pred = [], []
with torch.no_grad():
    for batch in tqdm(val_text_loader, desc="BERT"):
        outputs = bert_model(
            input_ids=batch['input_ids'].to(device),
            attention_mask=batch['attention_mask'].to(device)
        )
        preds = torch.argmax(outputs.logits, dim=1).cpu().tolist()
        bert_true.extend(batch['label'].tolist())
        bert_pred.extend(preds)

bert_true_labels = [le.inverse_transform([t])[0] for t in bert_true]
bert_pred_labels = [le.inverse_transform([p])[0] for p in bert_pred]
bert_acc = accuracy_score(bert_true_labels, bert_pred_labels)
print(f"BERT Accuracy: {bert_acc:.4f} ({bert_acc*100:.1f}%)")

json.dump({
    "model": "BERT", "accuracy": bert_acc,
    "predictions": [{"true": t, "predicted": p} for t, p in zip(bert_true_labels, bert_pred_labels)]
}, open(f"{config.RESULTS_DIR}/bert_predictions.json", 'w'), indent=2)

del bert_model
torch.cuda.empty_cache()

# ============================================================
# ViT
# ============================================================
print("\n" + "=" * 50)
print("ViT Evaluation (with per-sample predictions)")
print("=" * 50)

processor = ViTImageProcessor.from_pretrained(config.VIT_SAVE_DIR)
vit_model = ViTForImageClassification.from_pretrained(config.VIT_SAVE_DIR).to(device).eval()

val_img = ImageDataset(val_df, config.IMAGE_DIR, processor)
val_img_loader = DataLoader(val_img, batch_size=config.VIT_BATCH_SIZE, shuffle=False, num_workers=4)

vit_true, vit_pred = [], []
with torch.no_grad():
    for batch in tqdm(val_img_loader, desc="ViT"):
        outputs = vit_model(pixel_values=batch['pixel_values'].to(device))
        preds = torch.argmax(outputs.logits, dim=1).cpu().tolist()
        vit_true.extend(batch['label'].tolist())
        vit_pred.extend(preds)

vit_true_labels = [le.inverse_transform([t])[0] for t in vit_true]
vit_pred_labels = [le.inverse_transform([p])[0] for p in vit_pred]
vit_acc = accuracy_score(vit_true_labels, vit_pred_labels)
print(f"ViT Accuracy: {vit_acc:.4f} ({vit_acc*100:.1f}%)")

json.dump({
    "model": "ViT", "accuracy": vit_acc,
    "predictions": [{"true": t, "predicted": p} for t, p in zip(vit_true_labels, vit_pred_labels)]
}, open(f"{config.RESULTS_DIR}/vit_predictions.json", 'w'), indent=2)

del vit_model
torch.cuda.empty_cache()

# ============================================================
# Cross-attention
# ============================================================
print("\n" + "=" * 50)
print("Cross-attention Evaluation (with per-sample predictions)")
print("=" * 50)

bert_model = BertForSequenceClassification.from_pretrained(config.BERT_SAVE_DIR).to(device).eval()
vit_model = ViTForImageClassification.from_pretrained(config.VIT_SAVE_DIR).to(device).eval()

cross_attn = CrossAttentionFusion(num_classes=config.NUM_CLASSES)
cross_attn.load_state_dict(torch.load(config.CROSS_ATTN_SAVE_PATH, map_location=device, weights_only=True))
cross_attn.to(device).eval()

val_mm = MultimodalDataset(val_df, config.IMAGE_DIR, tokenizer, processor)
val_mm_loader = DataLoader(val_mm, batch_size=8, shuffle=False, num_workers=4)

ca_true, ca_pred = [], []
with torch.no_grad():
    for batch in tqdm(val_mm_loader, desc="Cross-attention"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        pixel_values = batch['pixel_values'].to(device)

        bert_out = bert_model.bert(input_ids=input_ids, attention_mask=attention_mask)
        vit_out = vit_model.vit(pixel_values=pixel_values)

        logits = cross_attn(bert_out.last_hidden_state, vit_out.last_hidden_state, text_mask=attention_mask)
        preds = torch.argmax(logits, dim=1).cpu().tolist()
        ca_true.extend(batch['label'].tolist())
        ca_pred.extend(preds)

ca_true_labels = [le.inverse_transform([t])[0] for t in ca_true]
ca_pred_labels = [le.inverse_transform([p])[0] for p in ca_pred]
ca_acc = accuracy_score(ca_true_labels, ca_pred_labels)
print(f"Cross-attention Accuracy: {ca_acc:.4f} ({ca_acc*100:.1f}%)")

json.dump({
    "model": "Cross-attention", "accuracy": ca_acc,
    "predictions": [{"true": t, "predicted": p} for t, p in zip(ca_true_labels, ca_pred_labels)]
}, open(f"{config.RESULTS_DIR}/cross_attention_predictions.json", 'w'), indent=2)

print(f"\nAll predictions saved to {config.RESULTS_DIR}/")
print("Now run: python merge_categories.py")
