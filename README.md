# News Headline Classification: Cross-Attention vs Qwen3-VL

## Overview

3 models compared on NYTimes news classification (image + headline → category):

| Model | Type | Training |
|-------|------|----------|
| **Cross-Attention Fusion** | BERT + ViT with cross-attention | Trained from scratch |
| **Qwen3-VL Zero-Shot** | Pretrained VLM | No training |
| **Qwen3-VL Fine-Tuned** | Pretrained VLM + LoRA | Adapted to dataset |

---

## Setup: RunPod

### 1. Create Pod
1. Go to [runpod.io](https://runpod.io) → sign up → add $10-15 credits
2. **Pods** → **Deploy** → Select **GPU Cloud**
3. Pick **A100 40GB** (or RTX 4090 24GB for budget)
4. Template: **RunPod PyTorch 2.4**
5. Volume: **50 GB**
6. Deploy

### 2. Connect VSCode
1. In RunPod, click pod → **Connect** → copy SSH command
2. Add your SSH public key in RunPod **Settings** → **SSH Keys**
   ```bash
   # Generate key if you don't have one (on YOUR machine)
   ssh-keygen -t ed25519
   cat ~/.ssh/id_ed25519.pub  # copy this to RunPod
   ```
3. In VSCode: install **Remote - SSH** extension
4. `Ctrl+Shift+P` → **Remote-SSH: Add New SSH Host** → paste SSH command
5. `Ctrl+Shift+P` → **Remote-SSH: Connect to Host** → select your pod

### 3. Upload Project Files
On your local machine:
```bash
# Upload all project files to the pod
scp -P <port> -i ~/.ssh/id_ed25519 *.py setup.sh root@<ip>:/workspace/
```

Or just copy-paste the files via VSCode's file explorer.

---

## Run Order

### Step 0: Install dependencies
```bash
cd /workspace
chmod +x setup.sh
./setup.sh
```

### Step 1: Download dataset
```bash
python download_data.py
```

After downloading and extracting, check the structure. You may need to edit `config.py` paths if the ZIP extracts into a subfolder. It should look like:
```
data/
├── imgs/              # folder of .jpg images
├── nytimes_train.json
├── nytimes_test.json
├── nytimes_dev.json
└── ...
```

### Step 2: Run EDA (important — computes class weights)
```bash
python 00_eda.py
```
This shows class distribution, checks for imbalance, and saves `class_weights.pt` used by all training scripts.

### Step 3: Train BERT (~20-30 min)
```bash
python 01_train_bert.py
```

### Step 4: Train ViT (~30-45 min)
```bash
python 02_train_vit.py
```

### Step 5: Train Cross-Attention (~20 min)
```bash
python 03_cross_attention.py
```

### Step 6: Qwen3 Zero-Shot (~30-60 min on 200 samples)
```bash
python 04_qwen3_zero_shot.py
```

### Step 7: Qwen3 Fine-Tune + Evaluate (~2-3 hrs)
```bash
python 05_qwen3_finetune.py
```

### Step 8: Download results
```bash
# From your LOCAL machine
scp -P <port> -i ~/.ssh/id_ed25519 -r root@<ip>:/workspace/results/ ./results/
```

**STOP THE POD** when done to stop billing!

---

## Output

All results are saved in `results/`:
- `cross_attention_results.json`
- `qwen3_zero_shot_results.json`
- `qwen3_finetuned_results.json`

Each contains accuracy, classification report, and per-sample predictions.

---

## Estimated Time & Cost

| Step | Time | GPU Cost (~$1.50/hr) |
|------|------|---------------------|
| BERT training | 20-30 min | $0.75 |
| ViT training | 30-45 min | $1.00 |
| Cross-attention | 20 min | $0.50 |
| Qwen3 zero-shot | 30-60 min | $1.25 |
| Qwen3 fine-tune | 2-3 hrs | $4.50 |
| **Total** | **~4-5 hrs** | **~$8** |

---

## Troubleshooting

- **OOM (Out of Memory)**: Reduce batch sizes in `config.py`
- **Model download slow**: Run `export HF_HUB_ENABLE_HF_TRANSFER=1` before scripts
- **Flash attention fails**: Non-critical, remove `attn_implementation="flash_attention_2"` if it causes issues
- **Qwen3 outputs weird text**: The `match_to_category()` function handles fuzzy matching. Check `raw_preds` in results JSON
- **gdown auth error**: Make sure your Google Drive link is set to "Anyone with the link can view"
