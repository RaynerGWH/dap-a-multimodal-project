#!/bin/bash
# ============================================================
# RunPod Environment Setup
# Run this ONCE when you first connect to the pod
# ============================================================

echo "=== Installing dependencies ==="

pip install --upgrade pip

# Core ML
pip install torch torchvision torchaudio
pip install transformers>=4.57.0 accelerate bitsandbytes
pip install peft datasets

# Data + utils
pip install pandas scikit-learn pillow tqdm

# Qwen3-VL specific
pip install qwen-vl-utils==0.0.14

# Flash attention (speeds up Qwen3, optional but recommended)
pip install flash-attn --no-build-isolation 2>/dev/null || echo "Flash attention install failed (non-critical, continuing...)"

# For downloading from Google Drive
pip install gdown

# HuggingFace login (needed to download models)
pip install huggingface_hub
echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "1. Download your dataset: python download_data.py"
echo "2. Train BERT: python 01_train_bert.py"
echo "3. Train ViT: python 02_train_vit.py"
echo "4. Train cross-attention: python 03_cross_attention.py"
echo "5. Qwen3 zero-shot: python 04_qwen3_zero_shot.py"
echo "6. Qwen3 fine-tune: python 05_qwen3_finetune.py"
