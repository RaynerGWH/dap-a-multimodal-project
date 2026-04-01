"""
Merge overlapping categories and recompute metrics for all models.
This shows what accuracy would look like if NYT had cleaner category boundaries.

Merge groups:
- Style + Fashion & Style → "Fashion/Style"
- Well + Health → "Health/Wellness"
- Economy + Global Business + Your Money → "Business/Finance"
- Books + Movies + Television → "Entertainment/Reviews"
- Media + Opinion → "Media/Opinion"

Run locally — just needs the result JSON files, no GPU needed.
"""

import json
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from collections import Counter

# ============================================================
# Define merge mapping
# ============================================================
MERGE_MAP = {
    # Fashion/Style
    "Style": "Fashion/Style",
    "Fashion & Style": "Fashion/Style",
    # Health/Wellness
    "Well": "Health/Wellness",
    "Health": "Health/Wellness",
    # Business/Finance
    "Economy": "Business/Finance",
    "Global Business": "Business/Finance",
    "Your Money": "Business/Finance",
    # Entertainment/Reviews
    "Books": "Entertainment/Reviews",
    "Movies": "Entertainment/Reviews",
    "Television": "Entertainment/Reviews",
    # Media/Opinion
    "Media": "Media/Opinion",
    "Opinion": "Media/Opinion",
}

def merge_label(label):
    return MERGE_MAP.get(label, label)

def compute_merged_metrics(y_true, y_pred, model_name):
    """Merge overlapping categories and recompute metrics."""
    merged_true = [merge_label(t) for t in y_true]
    merged_pred = [merge_label(p) for p in y_pred]
    
    acc = accuracy_score(merged_true, merged_pred)
    
    # Get sorted unique labels
    labels = sorted(set(merged_true + merged_pred))
    
    report = classification_report(
        merged_true, merged_pred,
        labels=labels,
        zero_division=0
    )
    
    report_dict = classification_report(
        merged_true, merged_pred,
        labels=labels,
        output_dict=True,
        zero_division=0
    )
    
    return acc, report, report_dict

# ============================================================
# Original categories (24) → Merged categories (17)
# ============================================================
print("=" * 60)
print("CATEGORY MERGING")
print("=" * 60)
print(f"\nMerge groups:")
print(f"  Style + Fashion & Style → Fashion/Style")
print(f"  Well + Health → Health/Wellness")
print(f"  Economy + Global Business + Your Money → Business/Finance")
print(f"  Books + Movies + Television → Entertainment/Reviews")
print(f"  Media + Opinion → Media/Opinion")
print(f"\n  24 original classes → 17 merged classes")

# ============================================================
# Process each model's results
# ============================================================

# --- BERT ---
print("\n" + "=" * 60)
print("BERT (9798 samples) — MERGED")
print("=" * 60)

try:
    with open("./results/bert_predictions.json", 'r') as f:
        bert = json.load(f)
    y_true = [p['true'] for p in bert['predictions']]
    y_pred = [p['predicted'] for p in bert['predictions']]
    orig_acc = accuracy_score(y_true, y_pred)
    print(f"\nOriginal accuracy (24 classes): {orig_acc:.4f} ({orig_acc*100:.1f}%)")
    acc, report, _ = compute_merged_metrics(y_true, y_pred, "BERT")
    print(f"Merged accuracy (17 classes):   {acc:.4f} ({acc*100:.1f}%)")
    print(f"Improvement: +{(acc - orig_acc)*100:.1f}%")
    print(f"\n{report}")
    bert_orig, bert_merged = orig_acc, acc
except Exception as e:
    print(f"Error: {e} — run eval_with_predictions.py first")
    bert_orig, bert_merged = 0.7356, None

# --- ViT ---
print("\n" + "=" * 60)
print("ViT (9798 samples) — MERGED")
print("=" * 60)

try:
    with open("./results/vit_predictions.json", 'r') as f:
        vit = json.load(f)
    y_true = [p['true'] for p in vit['predictions']]
    y_pred = [p['predicted'] for p in vit['predictions']]
    orig_acc = accuracy_score(y_true, y_pred)
    print(f"\nOriginal accuracy (24 classes): {orig_acc:.4f} ({orig_acc*100:.1f}%)")
    acc, report, _ = compute_merged_metrics(y_true, y_pred, "ViT")
    print(f"Merged accuracy (17 classes):   {acc:.4f} ({acc*100:.1f}%)")
    print(f"Improvement: +{(acc - orig_acc)*100:.1f}%")
    print(f"\n{report}")
    vit_orig, vit_merged = orig_acc, acc
except Exception as e:
    print(f"Error: {e} — run eval_with_predictions.py first")
    vit_orig, vit_merged = 0.5389, None

# --- Cross-attention ---
print("\n" + "=" * 60)
print("CROSS-ATTENTION (9798 samples) — MERGED")
print("=" * 60)

try:
    with open("./results/cross_attention_predictions.json", 'r') as f:
        ca = json.load(f)
    y_true = [p['true'] for p in ca['predictions']]
    y_pred = [p['predicted'] for p in ca['predictions']]
    orig_acc = accuracy_score(y_true, y_pred)
    print(f"\nOriginal accuracy (24 classes): {orig_acc:.4f} ({orig_acc*100:.1f}%)")
    acc, report, _ = compute_merged_metrics(y_true, y_pred, "Cross-attn")
    print(f"Merged accuracy (17 classes):   {acc:.4f} ({acc*100:.1f}%)")
    print(f"Improvement: +{(acc - orig_acc)*100:.1f}%")
    print(f"\n{report}")
    ca_orig, ca_merged = orig_acc, acc
except Exception as e:
    print(f"Error: {e} — run eval_with_predictions.py first")
    ca_orig, ca_merged = 0.7892, None

# --- Qwen3 Zero-Shot (1000 samples) ---
print("\n" + "=" * 60)
print("QWEN3-VL ZERO-SHOT (1000 samples) — MERGED")
print("=" * 60)

try:
    with open("./results/qwen3_zero_shot_1000.json", 'r') as f:
        zs = json.load(f)
    
    y_true = [p['true'] for p in zs['predictions'] if p['predicted'] != 'error' and p['predicted'] != 'unknown']
    y_pred = [p['predicted'] for p in zs['predictions'] if p['predicted'] != 'error' and p['predicted'] != 'unknown']
    
    # Filter to valid categories only
    all_cats = set(MERGE_MAP.keys()) | set(MERGE_MAP.values()) | {
        "Art & Design", "Automobiles", "Dance", "Education", "Food",
        "Music", "Real Estate", "Science", "Sports", "Technology",
        "Theater", "Travel"
    }
    valid = [(t, p) for t, p in zip(y_true, y_pred) if p in all_cats]
    if valid:
        y_true_v, y_pred_v = zip(*valid)
    else:
        y_true_v, y_pred_v = y_true, y_pred
    
    # Original accuracy
    orig_acc = accuracy_score(y_true_v, y_pred_v)
    print(f"\nOriginal accuracy (24 classes): {orig_acc:.4f} ({orig_acc*100:.1f}%)")
    
    # Merged accuracy
    acc, report, _ = compute_merged_metrics(list(y_true_v), list(y_pred_v), "Qwen3 ZS")
    print(f"Merged accuracy (17 classes):   {acc:.4f} ({acc*100:.1f}%)")
    print(f"Improvement: +{(acc - orig_acc)*100:.1f}%")
    print(f"\n{report}")
except Exception as e:
    print(f"Error: {e}")

# --- Qwen3 Fine-Tuned (1000 samples) ---
print("\n" + "=" * 60)
print("QWEN3-VL FINE-TUNED (1000 samples) — MERGED")
print("=" * 60)

try:
    with open("./results/qwen3_finetuned_1000.json", 'r') as f:
        ft = json.load(f)
    
    y_true = [p['true'] for p in ft['predictions'] if p['predicted'] != 'error' and p['predicted'] != 'unknown']
    y_pred = [p['predicted'] for p in ft['predictions'] if p['predicted'] != 'error' and p['predicted'] != 'unknown']
    
    valid = [(t, p) for t, p in zip(y_true, y_pred) if p in all_cats]
    if valid:
        y_true_v, y_pred_v = zip(*valid)
    else:
        y_true_v, y_pred_v = y_true, y_pred
    
    orig_acc = accuracy_score(y_true_v, y_pred_v)
    print(f"\nOriginal accuracy (24 classes): {orig_acc:.4f} ({orig_acc*100:.1f}%)")
    
    acc, report, _ = compute_merged_metrics(list(y_true_v), list(y_pred_v), "Qwen3 FT")
    print(f"Merged accuracy (17 classes):   {acc:.4f} ({acc*100:.1f}%)")
    print(f"Improvement: +{(acc - orig_acc)*100:.1f}%")
    print(f"\n{report}")
except Exception as e:
    print(f"Error: {e}")

# ============================================================
# Summary table
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY: ORIGINAL vs MERGED ACCURACY")
print("=" * 60)
print(f"\n{'Model':<30} {'Original (24)':<15} {'Merged (17)':<15} {'Improvement':<12}")
print("-" * 72)

try:
    print(f"{'ViT (image only)':<30} {vit_orig*100:.1f}%{'':>8} {vit_merged*100:.1f}%{'':>8} +{(vit_merged-vit_orig)*100:.1f}%")
except:
    print(f"{'ViT (image only)':<30} 53.9%{'':>8} {'(re-run needed)':<15}")

try:
    print(f"{'BERT (text only)':<30} {bert_orig*100:.1f}%{'':>8} {bert_merged*100:.1f}%{'':>8} +{(bert_merged-bert_orig)*100:.1f}%")
except:
    print(f"{'BERT (text only)':<30} 73.6%{'':>8} {'(re-run needed)':<15}")

try:
    print(f"{'Cross-attention fusion':<30} {ca_orig*100:.1f}%{'':>8} {ca_merged*100:.1f}%{'':>8} +{(ca_merged-ca_orig)*100:.1f}%")
except:
    print(f"{'Cross-attention fusion':<30} 78.9%{'':>8} {'(re-run needed)':<15}")

# Print Qwen3 results with merged
try:
    with open("./results/qwen3_zero_shot_1000.json", 'r') as f:
        zs = json.load(f)
    y_true = [p['true'] for p in zs['predictions'] if p['predicted'] != 'error' and p['predicted'] != 'unknown']
    y_pred = [p['predicted'] for p in zs['predictions'] if p['predicted'] != 'error' and p['predicted'] != 'unknown']
    valid = [(t, p) for t, p in zip(y_true, y_pred) if p in all_cats]
    if valid: y_true_v, y_pred_v = zip(*valid)
    else: y_true_v, y_pred_v = y_true, y_pred
    zs_orig = accuracy_score(y_true_v, y_pred_v)
    zs_merged, _, _ = compute_merged_metrics(list(y_true_v), list(y_pred_v), "")
    print(f"{'Qwen3-VL zero-shot':<30} {zs_orig*100:.1f}%{'':>8} {zs_merged*100:.1f}%{'':>8} +{(zs_merged-zs_orig)*100:.1f}%")
except:
    pass

try:
    with open("./results/qwen3_finetuned_1000.json", 'r') as f:
        ft = json.load(f)
    y_true = [p['true'] for p in ft['predictions'] if p['predicted'] != 'error' and p['predicted'] != 'unknown']
    y_pred = [p['predicted'] for p in ft['predictions'] if p['predicted'] != 'error' and p['predicted'] != 'unknown']
    valid = [(t, p) for t, p in zip(y_true, y_pred) if p in all_cats]
    if valid: y_true_v, y_pred_v = zip(*valid)
    else: y_true_v, y_pred_v = y_true, y_pred
    ft_orig = accuracy_score(y_true_v, y_pred_v)
    ft_merged, _, _ = compute_merged_metrics(list(y_true_v), list(y_pred_v), "")
    print(f"{'Qwen3-VL fine-tuned':<30} {ft_orig*100:.1f}%{'':>8} {ft_merged*100:.1f}%{'':>8} +{(ft_merged-ft_orig)*100:.1f}%")
except:
    pass

print(f"\nNote: BERT, ViT, and cross-attention need per-sample predictions")
print(f"saved to compute merged metrics. Re-run eval scripts with prediction saving.")
