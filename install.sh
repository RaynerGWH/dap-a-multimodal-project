#!/bin/bash
echo "=== Installing dependencies ==="
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu124
pip install git+https://github.com/huggingface/transformers.git
pip install peft accelerate scikit-learn pillow tqdm pandas qwen-vl-utils==0.0.14
pip install fastapi uvicorn python-multipart
echo "=== Done ==="
