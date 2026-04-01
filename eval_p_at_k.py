"""
Compute P@1, P@2, P@3 for BERT, ViT, and cross-attention.
P@k = the correct label is in the model's top-k predictions.

If P@3 is much higher than P@1, it proves the model understands
the content but the overlapping categories cause "wrong" predictions.
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
from tqdm import tqdm

import config
from data_utils import load_data, get_label_encoder, get_train_val_split, TextDataset, ImageDataset, MultimodalDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open(f"{config.SAVE_DIR}/label_encoder.pkl", 'rb') as f:
    le = pickle.load(f)

df = load_data(config.TRAIN_JSON, config.IMAGE_DIR)
df, _ = get_label_encoder(df)
_, val_df = get_train_val_split(df)


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
        return self.classifier(torch.cat([text_pooled, image_pooled], dim=1))


def compute_p_at_k(all_logits, all_labels, k_values=[1, 2, 3]):
    """Compute P@k for each k value."""
    results = {}
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    for k in k_values:
        top_k = torch.topk(all_logits, k, dim=1).indices
        correct = 0
        for i in range(len(all_labels)):
            if all_labels[i] in top_k[i]:
                correct += 1
        results[f"P@{k}"] = correct / len(all_labels)

    return results


def compute_per_class_p_at_k(all_logits, all_labels, k_values=[1, 2, 3]):
    """Compute P@k per class — shows which classes benefit most from P@3."""
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    per_class = {}
    for cls_idx in range(len(le.classes_)):
        mask = all_labels == cls_idx
        if mask.sum() == 0:
            continue
        cls_logits = all_logits[mask]
        cls_labels = all_labels[mask]
        cls_name = le.classes_[cls_idx]
        per_class[cls_name] = {}
        for k in k_values:
            top_k = torch.topk(cls_logits, k, dim=1).indices
            correct = sum(1 for i in range(len(cls_labels)) if cls_labels[i] in top_k[i])
            per_class[cls_name][f"P@{k}"] = correct / len(cls_labels)

    return per_class


# ============================================================
# BERT
# ============================================================
print("=" * 60)
print("BERT P@k")
print("=" * 60)

tokenizer = BertTokenizer.from_pretrained(config.BERT_SAVE_DIR)
bert_model = BertForSequenceClassification.from_pretrained(config.BERT_SAVE_DIR).to(device).eval()

val_text = TextDataset(val_df, tokenizer, max_length=config.MAX_TEXT_LENGTH)
loader = DataLoader(val_text, batch_size=config.BERT_BATCH_SIZE, shuffle=False)

logits_list, labels_list = [], []
with torch.no_grad():
    for batch in tqdm(loader, desc="BERT"):
        out = bert_model(input_ids=batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device))
        logits_list.append(out.logits.cpu())
        labels_list.append(batch['label'])

bert_pk = compute_p_at_k(logits_list, labels_list)
bert_per_class = compute_per_class_p_at_k(logits_list, labels_list)
print(f"  P@1: {bert_pk['P@1']*100:.1f}%  P@2: {bert_pk['P@2']*100:.1f}%  P@3: {bert_pk['P@3']*100:.1f}%")

del bert_model
torch.cuda.empty_cache()

# ============================================================
# ViT
# ============================================================
print("\n" + "=" * 60)
print("ViT P@k")
print("=" * 60)

processor = ViTImageProcessor.from_pretrained(config.VIT_SAVE_DIR)
vit_model = ViTForImageClassification.from_pretrained(config.VIT_SAVE_DIR).to(device).eval()

val_img = ImageDataset(val_df, config.IMAGE_DIR, processor)
loader = DataLoader(val_img, batch_size=config.VIT_BATCH_SIZE, shuffle=False, num_workers=4)

logits_list, labels_list = [], []
with torch.no_grad():
    for batch in tqdm(loader, desc="ViT"):
        out = vit_model(pixel_values=batch['pixel_values'].to(device))
        logits_list.append(out.logits.cpu())
        labels_list.append(batch['label'])

vit_pk = compute_p_at_k(logits_list, labels_list)
vit_per_class = compute_per_class_p_at_k(logits_list, labels_list)
print(f"  P@1: {vit_pk['P@1']*100:.1f}%  P@2: {vit_pk['P@2']*100:.1f}%  P@3: {vit_pk['P@3']*100:.1f}%")

del vit_model

# ============================================================
# Cross-attention
# ============================================================
print("\n" + "=" * 60)
print("Cross-attention P@k")
print("=" * 60)

bert_model = BertForSequenceClassification.from_pretrained(config.BERT_SAVE_DIR).to(device).eval()
vit_model = ViTForImageClassification.from_pretrained(config.VIT_SAVE_DIR).to(device).eval()
cross_attn = CrossAttentionFusion(num_classes=config.NUM_CLASSES)
cross_attn.load_state_dict(torch.load(config.CROSS_ATTN_SAVE_PATH, map_location=device, weights_only=True))
cross_attn.to(device).eval()

val_mm = MultimodalDataset(val_df, config.IMAGE_DIR, tokenizer, processor)
loader = DataLoader(val_mm, batch_size=8, shuffle=False, num_workers=4)

logits_list, labels_list = [], []
with torch.no_grad():
    for batch in tqdm(loader, desc="Cross-attention"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        pixel_values = batch['pixel_values'].to(device)
        bert_out = bert_model.bert(input_ids=input_ids, attention_mask=attention_mask)
        vit_out = vit_model.vit(pixel_values=pixel_values)
        logits = cross_attn(bert_out.last_hidden_state, vit_out.last_hidden_state, text_mask=attention_mask)
        logits_list.append(logits.cpu())
        labels_list.append(batch['label'])

ca_pk = compute_p_at_k(logits_list, labels_list)
ca_per_class = compute_per_class_p_at_k(logits_list, labels_list)
print(f"  P@1: {ca_pk['P@1']*100:.1f}%  P@2: {ca_pk['P@2']*100:.1f}%  P@3: {ca_pk['P@3']*100:.1f}%")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("P@k SUMMARY")
print("=" * 60)
print(f"\n{'Model':<25} {'P@1':<10} {'P@2':<10} {'P@3':<10} {'P@1→P@3 gain':<15}")
print("-" * 70)
for name, pk in [("BERT", bert_pk), ("ViT", vit_pk), ("Cross-attention", ca_pk)]:
    gain = pk['P@3'] - pk['P@1']
    print(f"{name:<25} {pk['P@1']*100:.1f}%{'':>4} {pk['P@2']*100:.1f}%{'':>4} {pk['P@3']*100:.1f}%{'':>4} +{gain*100:.1f}%")

# ============================================================
# Per-class P@k (show biggest gainers)
# ============================================================
print("\n" + "=" * 60)
print("PER-CLASS P@k (classes with biggest P@1→P@3 improvement)")
print("=" * 60)

print(f"\n{'Class':<20} {'Model':<18} {'P@1':<8} {'P@2':<8} {'P@3':<8} {'Gain':<8}")
print("-" * 70)

for model_name, per_class in [("BERT", bert_per_class), ("ViT", vit_per_class), ("Cross-attn", ca_per_class)]:
    # Sort by P@1→P@3 gain
    gains = [(cls, data['P@3'] - data['P@1']) for cls, data in per_class.items()]
    gains.sort(key=lambda x: -x[1])

    for cls, gain in gains[:5]:  # top 5 gainers per model
        d = per_class[cls]
        print(f"{cls:<20} {model_name:<18} {d['P@1']*100:.1f}%{'':>2} {d['P@2']*100:.1f}%{'':>2} {d['P@3']*100:.1f}%{'':>2} +{gain*100:.1f}%")
    print()

# Save results
results = {
    "summary": {
        "BERT": bert_pk, "ViT": vit_pk, "Cross-attention": ca_pk,
    },
    "per_class": {
        "BERT": bert_per_class, "ViT": vit_per_class, "Cross-attention": ca_per_class,
    }
}
with open(f"{config.RESULTS_DIR}/p_at_k_results.json", 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to {config.RESULTS_DIR}/p_at_k_results.json")
