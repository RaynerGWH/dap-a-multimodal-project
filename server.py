"""
FastAPI backend for multimodal news classification demo.
Loads 3 models: Cross-attention, Qwen3-VL zero-shot, Qwen3-VL fine-tuned.
Run on RunPod with: uvicorn server:app --host 0.0.0.0 --port 8000
"""

import torch
import pickle
import os
import json
import base64
import io
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from transformers import (
    BertForSequenceClassification, BertTokenizer,
    ViTForImageClassification, ViTImageProcessor,
    Qwen3VLForConditionalGeneration, AutoProcessor,
)
from peft import PeftModel
import torch.nn as nn

import config

app = FastAPI(title="Multimodal News Classifier")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================
# Global model references
# ============================================================
models = {}


# ============================================================
# Cross-attention model definition (must match training)
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
# Load all models on startup
# ============================================================
@app.on_event("startup")
async def load_models():
    print("Loading models...")

    # Label encoder
    with open(f"{config.SAVE_DIR}/label_encoder.pkl", 'rb') as f:
        models['le'] = pickle.load(f)
    models['categories'] = list(models['le'].classes_)

    # --- BERT ---
    print("  Loading BERT...")
    models['bert_tokenizer'] = BertTokenizer.from_pretrained(config.BERT_SAVE_DIR)
    models['bert'] = BertForSequenceClassification.from_pretrained(config.BERT_SAVE_DIR).to(device).eval()

    # --- ViT ---
    print("  Loading ViT...")
    models['vit_processor'] = ViTImageProcessor.from_pretrained(config.VIT_SAVE_DIR)
    models['vit'] = ViTForImageClassification.from_pretrained(config.VIT_SAVE_DIR).to(device).eval()

    # --- Cross-attention ---
    print("  Loading cross-attention...")
    cross_attn = CrossAttentionFusion(num_classes=config.NUM_CLASSES)
    cross_attn.load_state_dict(torch.load(config.CROSS_ATTN_SAVE_PATH, map_location=device, weights_only=True))
    cross_attn.to(device).eval()
    models['cross_attn'] = cross_attn

    # --- Qwen3-VL (base for zero-shot) ---
    print("  Loading Qwen3-VL base...")
    models['qwen_processor'] = AutoProcessor.from_pretrained(config.QWEN3_MODEL_NAME)
    models['qwen_base'] = Qwen3VLForConditionalGeneration.from_pretrained(
        config.QWEN3_MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto",
    )
    models['qwen_base'].eval()

    # --- Qwen3-VL fine-tuned ---
    print("  Loading Qwen3-VL fine-tuned...")
    qwen_ft_base = Qwen3VLForConditionalGeneration.from_pretrained(
        config.QWEN3_MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto",
    )
    models['qwen_finetuned'] = PeftModel.from_pretrained(qwen_ft_base, config.QWEN3_FINETUNE_DIR)
    models['qwen_finetuned'].eval()

    print("All models loaded!")


# ============================================================
# Helper functions
# ============================================================
def get_probs(logits):
    """Convert logits to probability dict."""
    probs = torch.softmax(logits, dim=-1).squeeze().cpu().tolist()
    return {cat: round(p, 4) for cat, p in zip(models['categories'], probs)}


def qwen_predict(model, image, headline):
    """Run Qwen3-VL inference."""
    category_str = ", ".join(models['categories'])
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": f"Classify this news article into one of: {category_str}.\nHeadline: \"{headline}\"\nRespond with ONLY the category name."},
    ]}]
    processor = models['qwen_processor']
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out)]
    raw = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

    # Match to known category
    matched = raw.lower()
    for cat in models['categories']:
        if matched == cat.lower():
            return cat, raw
    for cat in models['categories']:
        if cat.lower() in matched:
            return cat, raw
    return raw, raw


# ============================================================
# Endpoints
# ============================================================
@app.get("/health")
async def health():
    return {"status": "ok", "models_loaded": list(models.keys())}


@app.post("/predict")
async def predict(file: UploadFile = File(...), headline: str = Form(...)):
    # Read image
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    results = {}

    # --- Cross-attention ---
    try:
        tokenizer = models['bert_tokenizer']
        vit_proc = models['vit_processor']

        # BERT embeddings
        encoding = tokenizer(headline, max_length=32, padding='max_length', truncation=True, return_tensors='pt')
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        with torch.no_grad():
            bert_out = models['bert'].bert(input_ids=input_ids, attention_mask=attention_mask)
            bert_seq = bert_out.last_hidden_state

            # ViT embeddings
            pixel_values = vit_proc(images=image, return_tensors="pt")['pixel_values'].to(device)
            vit_out = models['vit'].vit(pixel_values=pixel_values)
            vit_seq = vit_out.last_hidden_state

            # Cross-attention prediction
            logits = models['cross_attn'](bert_seq, vit_seq, text_mask=attention_mask)
            probs = get_probs(logits)
            pred_idx = torch.argmax(logits, dim=-1).item()
            pred_label = models['le'].inverse_transform([pred_idx])[0]

        results['cross_attention'] = {
            'prediction': pred_label,
            'probabilities': probs,
        }
    except Exception as e:
        results['cross_attention'] = {'prediction': 'error', 'error': str(e)}

    # --- Qwen3 zero-shot ---
    try:
        pred, raw = qwen_predict(models['qwen_base'], image, headline)
        results['qwen3_zero_shot'] = {
            'prediction': pred,
            'raw_output': raw,
        }
    except Exception as e:
        results['qwen3_zero_shot'] = {'prediction': 'error', 'error': str(e)}

    # --- Qwen3 fine-tuned ---
    try:
        pred, raw = qwen_predict(models['qwen_finetuned'], image, headline)
        results['qwen3_finetuned'] = {
            'prediction': pred,
            'raw_output': raw,
        }
    except Exception as e:
        results['qwen3_finetuned'] = {'prediction': 'error', 'error': str(e)}

    return results


@app.get("/categories")
async def categories():
    return {"categories": models.get('categories', [])}
