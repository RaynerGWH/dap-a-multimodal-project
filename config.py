"""
Shared configuration for all training scripts.
Edit paths here if your data is in a different location.
"""

import os

# ============================================================
# DATA PATHS — edit these to match your setup
# ============================================================

DATA_DIR = "./data"
IMAGE_DIR = os.path.join(DATA_DIR, "imgs")  # folder containing {image_id}.jpg

TRAIN_JSON = os.path.join(DATA_DIR, "nytimes_train.json")
TEST_JSON = os.path.join(DATA_DIR, "nytimes_test.json")
DEV_JSON = os.path.join(DATA_DIR, "nytimes_dev.json")

# ============================================================
# MODEL SAVE PATHS
# ============================================================

SAVE_DIR = "./saved_models"
BERT_SAVE_DIR = os.path.join(SAVE_DIR, "bert_news_classifier")
VIT_SAVE_DIR = os.path.join(SAVE_DIR, "vit_news_classifier")
CROSS_ATTN_SAVE_PATH = os.path.join(SAVE_DIR, "cross_attention_best.pt")
EMBEDDINGS_DIR = os.path.join(SAVE_DIR, "embeddings")
QWEN3_FINETUNE_DIR = os.path.join(SAVE_DIR, "qwen3_finetuned")

RESULTS_DIR = "./results"

# ============================================================
# MODEL CONFIG
# ============================================================

BERT_MODEL_NAME = "bert-base-uncased"
VIT_MODEL_NAME = "google/vit-base-patch16-224"
QWEN3_MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"  # small enough for LoRA on single GPU

NUM_CLASSES = 24  # NYTimes sections
RANDOM_SEED = 42
MAX_TEXT_LENGTH = 32

# ============================================================
# TRAINING CONFIG
# ============================================================

BERT_EPOCHS = 5
BERT_BATCH_SIZE = 32
BERT_LR = 2e-5

VIT_EPOCHS = 5
VIT_BATCH_SIZE = 16
VIT_LR = 2e-5

CROSS_ATTN_EPOCHS = 15
CROSS_ATTN_BATCH_SIZE = 32
CROSS_ATTN_LR = 1e-3

QWEN3_FINETUNE_EPOCHS = 3
QWEN3_FINETUNE_BATCH_SIZE = 2
QWEN3_FINETUNE_LR = 2e-4

# ============================================================
# Create directories
# ============================================================

for d in [SAVE_DIR, BERT_SAVE_DIR, VIT_SAVE_DIR, EMBEDDINGS_DIR, QWEN3_FINETUNE_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)
