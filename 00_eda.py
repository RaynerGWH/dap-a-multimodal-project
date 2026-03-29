"""
Step 0.5: Exploratory Data Analysis

Run this AFTER downloading the dataset to understand what you're working with.
Also computes class weights for handling imbalance — saved for use by training scripts.
"""

import json
import os
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder
import torch
import pickle

import config

# ============================================================
# Load dataset
# ============================================================
print("Loading dataset...")

for name, path in [("train", config.TRAIN_JSON), ("dev", config.DEV_JSON), ("test", config.TEST_JSON)]:
    if os.path.exists(path):
        with open(path, 'r') as f:
            data = json.load(f)
        print(f"  {name}: {len(data)} samples")
    else:
        print(f"  {name}: FILE NOT FOUND at {path}")

# Work with train set for analysis
with open(config.TRAIN_JSON, 'r') as f:
    train_data = json.load(f)
df = pd.DataFrame(train_data)

print(f"\nTotal train samples: {len(df)}")
print(f"Columns: {list(df.columns)}")

# ============================================================
# Check images
# ============================================================
if os.path.exists(config.IMAGE_DIR):
    available_images = set(os.listdir(config.IMAGE_DIR))
    df['image_filename'] = df['image_id'] + '.jpg'
    df_with_images = df[df['image_filename'].isin(available_images)]
    print(f"\nImages available: {len(available_images)}")
    print(f"Samples with matching images: {len(df_with_images)} / {len(df)}")
    df = df_with_images
else:
    print(f"\nWARNING: Image directory not found at {config.IMAGE_DIR}")
    print("Check config.py IMAGE_DIR path")

# ============================================================
# Class distribution
# ============================================================
print(f"\n{'='*60}")
print("CLASS DISTRIBUTION")
print(f"{'='*60}")

section_counts = df['section'].value_counts()
print(f"\nNumber of unique classes: {len(section_counts)}")
print(f"\nClass counts (sorted):")
print("-" * 45)
for section, count in section_counts.items():
    pct = count / len(df) * 100
    bar = "█" * int(pct)
    print(f"  {section:25s} {count:5d}  ({pct:5.1f}%) {bar}")

print(f"\n  {'Total':25s} {len(df):5d}")
print(f"\n  Largest class:  {section_counts.index[0]} ({section_counts.iloc[0]})")
print(f"  Smallest class: {section_counts.index[-1]} ({section_counts.iloc[-1]})")
print(f"  Imbalance ratio: {section_counts.iloc[0] / section_counts.iloc[-1]:.1f}x")

# ============================================================
# Headline stats
# ============================================================
print(f"\n{'='*60}")
print("HEADLINE STATISTICS")
print(f"{'='*60}")

df['headline_len'] = df['headline'].str.len()
df['headline_words'] = df['headline'].str.split().str.len()

print(f"  Char length:  mean={df['headline_len'].mean():.0f}, "
      f"min={df['headline_len'].min()}, max={df['headline_len'].max()}")
print(f"  Word count:   mean={df['headline_words'].mean():.1f}, "
      f"min={df['headline_words'].min()}, max={df['headline_words'].max()}")

# ============================================================
# Compute class weights (for handling imbalance)
# ============================================================
print(f"\n{'='*60}")
print("CLASS WEIGHTS (for weighted CrossEntropyLoss)")
print(f"{'='*60}")

le = LabelEncoder()
labels = le.fit_transform(df['section'])

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels),
    y=labels
)

print(f"\nClasses and their weights:")
for cls_name, weight in zip(le.classes_, class_weights):
    print(f"  {cls_name:25s}  weight={weight:.3f}")

# Save class weights and label encoder
weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
torch.save(weights_tensor, os.path.join(config.SAVE_DIR, "class_weights.pt"))

with open(os.path.join(config.SAVE_DIR, "label_encoder.pkl"), 'wb') as f:
    pickle.dump(le, f)

print(f"\nSaved class_weights.pt and label_encoder.pkl to {config.SAVE_DIR}/")
print(f"Number of classes: {len(le.classes_)}")

# Update config if class count differs
if len(le.classes_) != config.NUM_CLASSES:
    print(f"\n⚠️  WARNING: config.py says NUM_CLASSES={config.NUM_CLASSES} "
          f"but dataset has {len(le.classes_)} classes!")
    print(f"   Update NUM_CLASSES in config.py to {len(le.classes_)}")

# ============================================================
# Sample data
# ============================================================
print(f"\n{'='*60}")
print("SAMPLE DATA (first 5)")
print(f"{'='*60}")
for i, row in df.head(5).iterrows():
    print(f"\n  [{i}] Section: {row['section']}")
    print(f"      Headline: {row['headline'][:80]}...")
    print(f"      Image ID: {row['image_id']}")
