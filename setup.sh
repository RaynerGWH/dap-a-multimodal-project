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