"""
Step 3: Cross-Attention Fusion Model

This script:
1. Loads trained BERT and ViT models
2. Extracts FULL sequence embeddings (not just CLS — needed for cross-attention)
3. Trains a cross-attention fusion classifier
4. Evaluates on validation set

Cross-attention: text tokens attend to image patches, and image patches attend to text tokens.
This lets the model learn which parts of the image are relevant to which words.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import (
    BertForSequenceClassification, BertTokenizer,
    ViTForImageClassification, ViTImageProcessor
)
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm
import json
import os
import pickle

import config
from data_utils import load_data, get_label_encoder, get_train_val_split, MultimodalDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ============================================================
# 1. CROSS-ATTENTION MODEL ARCHITECTURE
# ============================================================

class CrossAttentionFusion(nn.Module):
    """
    Cross-attention fusion between text (BERT) and image (ViT) sequences.

    Unlike early fusion (concatenate CLS tokens) or late fusion (separate classifiers + combine),
    cross-attention lets text tokens directly attend to image patches and vice versa.
    This allows the model to learn fine-grained text-image relationships.
    """
    def __init__(self, embed_dim=768, num_heads=8, num_classes=24, dropout=0.3):
        super().__init__()

        # Text tokens attend to image patches (Query=text, Key/Value=image)
        self.text_to_image_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        # Image patches attend to text tokens (Query=image, Key/Value=text)
        self.image_to_text_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )

        # Layer norms
        self.text_norm = nn.LayerNorm(embed_dim)
        self.image_norm = nn.LayerNorm(embed_dim)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, text_seq, image_seq, text_mask=None):
        """
        Args:
            text_seq:  [batch, text_len, 768]  — BERT token embeddings
            image_seq: [batch, num_patches, 768] — ViT patch embeddings
            text_mask: [batch, text_len] — attention mask for text padding
        """
        # Cross-attention: text attends to image
        text_attended, _ = self.text_to_image_attn(
            query=text_seq, key=image_seq, value=image_seq
        )
        text_attended = self.text_norm(text_attended + text_seq)  # residual

        # Cross-attention: image attends to text
        image_attended, _ = self.image_to_text_attn(
            query=image_seq, key=text_seq, value=text_seq,
            key_padding_mask=(~text_mask.bool()) if text_mask is not None else None
        )
        image_attended = self.image_norm(image_attended + image_seq)  # residual

        # Pool: mean over sequence dimension
        text_pooled = text_attended.mean(dim=1)    # [batch, 768]
        image_pooled = image_attended.mean(dim=1)  # [batch, 768]

        # Combine and classify
        combined = torch.cat([text_pooled, image_pooled], dim=1)  # [batch, 1536]
        return self.classifier(combined)


# ============================================================
# 2. EXTRACT FULL SEQUENCE EMBEDDINGS
# ============================================================

def extract_sequence_embeddings(bert_model, vit_model, dataloader, device):
    """
    Extract full sequence embeddings (not just CLS) from BERT and ViT.
    BERT: [batch, seq_len, 768]
    ViT:  [batch, num_patches+1, 768]  (197 for ViT-base with 224x224 images)
    """
    all_bert_seq = []
    all_vit_seq = []
    all_masks = []
    all_labels = []

    bert_model.eval()
    vit_model.eval()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting sequence embeddings"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            pixel_values = batch['pixel_values'].to(device)

            # BERT: full sequence output [batch, seq_len, 768]
            bert_out = bert_model.bert(input_ids=input_ids, attention_mask=attention_mask)
            bert_seq = bert_out.last_hidden_state

            # ViT: full sequence output [batch, 197, 768]
            vit_out = vit_model.vit(pixel_values=pixel_values)
            vit_seq = vit_out.last_hidden_state

            all_bert_seq.append(bert_seq.cpu())
            all_vit_seq.append(vit_seq.cpu())
            all_masks.append(attention_mask.cpu())
            all_labels.append(batch['label'])

    return (
        torch.cat(all_bert_seq, dim=0),
        torch.cat(all_vit_seq, dim=0),
        torch.cat(all_masks, dim=0),
        torch.cat(all_labels, dim=0),
    )


# ============================================================
# 3. MAIN
# ============================================================

if __name__ == "__main__":

    # --- Load data ---
    df = load_data(config.TRAIN_JSON, config.IMAGE_DIR)
    df, le = get_label_encoder(df)
    train_df, val_df = get_train_val_split(df)

    # --- Load trained models ---
    print("\nLoading trained BERT model...")
    bert_model = BertForSequenceClassification.from_pretrained(config.BERT_SAVE_DIR)
    bert_model.to(device)
    bert_model.eval()

    tokenizer = BertTokenizer.from_pretrained(config.BERT_SAVE_DIR)

    print("Loading trained ViT model...")
    vit_model = ViTForImageClassification.from_pretrained(config.VIT_SAVE_DIR)
    vit_model.to(device)
    vit_model.eval()

    processor = ViTImageProcessor.from_pretrained(config.VIT_SAVE_DIR)

    # --- Create multimodal dataloaders ---
    train_dataset = MultimodalDataset(train_df, config.IMAGE_DIR, tokenizer, processor)
    val_dataset = MultimodalDataset(val_df, config.IMAGE_DIR, tokenizer, processor)

    # Smaller batch size for embedding extraction (full sequences use more memory)
    extract_loader_train = DataLoader(train_dataset, batch_size=8, shuffle=False, num_workers=4)
    extract_loader_val = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

    # --- Extract or load embeddings ---
    train_emb_path = os.path.join(config.EMBEDDINGS_DIR, "train_seq_embeddings.pt")
    val_emb_path = os.path.join(config.EMBEDDINGS_DIR, "val_seq_embeddings.pt")

    if os.path.exists(train_emb_path) and os.path.exists(val_emb_path):
        print("\nLoading cached embeddings...")
        train_data = torch.load(train_emb_path)
        val_data = torch.load(val_emb_path)
    else:
        print("\nExtracting train embeddings...")
        train_bert, train_vit, train_masks, train_labels = extract_sequence_embeddings(
            bert_model, vit_model, extract_loader_train, device
        )
        print(f"  BERT: {train_bert.shape}, ViT: {train_vit.shape}")

        print("Extracting val embeddings...")
        val_bert, val_vit, val_masks, val_labels = extract_sequence_embeddings(
            bert_model, vit_model, extract_loader_val, device
        )
        print(f"  BERT: {val_bert.shape}, ViT: {val_vit.shape}")

        # Save
        train_data = {'bert': train_bert, 'vit': train_vit, 'masks': train_masks, 'labels': train_labels}
        val_data = {'bert': val_bert, 'vit': val_vit, 'masks': val_masks, 'labels': val_labels}
        torch.save(train_data, train_emb_path)
        torch.save(val_data, val_emb_path)
        print(f"Saved embeddings to {config.EMBEDDINGS_DIR}")

    # Free GPU memory from BERT/ViT
    del bert_model, vit_model
    torch.cuda.empty_cache()

    # --- Create training dataloaders from embeddings ---
    train_emb_dataset = TensorDataset(
        train_data['bert'], train_data['vit'], train_data['masks'], train_data['labels']
    )
    val_emb_dataset = TensorDataset(
        val_data['bert'], val_data['vit'], val_data['masks'], val_data['labels']
    )

    train_loader = DataLoader(train_emb_dataset, batch_size=config.CROSS_ATTN_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_emb_dataset, batch_size=config.CROSS_ATTN_BATCH_SIZE, shuffle=False)

    # --- Create cross-attention model ---
    model = CrossAttentionFusion(num_classes=config.NUM_CLASSES)
    model.to(device)
    print(f"\nCross-attention model params: {sum(p.numel() for p in model.parameters()):,}")
    print(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.CROSS_ATTN_LR)
    class_weights = torch.load(f"{config.SAVE_DIR}/class_weights.pt").to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # --- Train ---
    best_val_acc = 0

    for epoch in range(config.CROSS_ATTN_EPOCHS):
        # Train
        model.train()
        total_loss, correct, total = 0, 0, 0

        for bert_emb, vit_emb, masks, labels in tqdm(train_loader, desc=f"CrossAttn Epoch {epoch+1}"):
            bert_emb = bert_emb.to(device)
            vit_emb = vit_emb.to(device)
            masks = masks.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(bert_emb, vit_emb, text_mask=masks)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total

        # Evaluate
        model.eval()
        val_correct, val_total = 0, 0
        all_preds, all_true = [], []

        with torch.no_grad():
            for bert_emb, vit_emb, masks, labels in val_loader:
                bert_emb = bert_emb.to(device)
                vit_emb = vit_emb.to(device)
                masks = masks.to(device)
                labels = labels.to(device)

                outputs = model(bert_emb, vit_emb, text_mask=masks)
                preds = torch.argmax(outputs, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                all_preds.extend(preds.cpu().tolist())
                all_true.extend(labels.cpu().tolist())

        val_acc = val_correct / val_total
        print(f"  Train Loss: {total_loss/len(train_loader):.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), config.CROSS_ATTN_SAVE_PATH)
            print(f"  ✓ Saved best model (val_acc: {val_acc:.4f})")

    # --- Final evaluation ---
    print(f"\n{'='*50}")
    print(f"Cross-Attention Training Complete!")
    print(f"Best Val Accuracy: {best_val_acc:.4f}")
    print(f"{'='*50}")

    # Load best and get full report
    model.load_state_dict(torch.load(config.CROSS_ATTN_SAVE_PATH))
    model.eval()

    all_preds, all_true = [], []
    with torch.no_grad():
        for bert_emb, vit_emb, masks, labels in val_loader:
            outputs = model(bert_emb.to(device), vit_emb.to(device), masks.to(device))
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_true.extend(labels.tolist())

    report = classification_report(all_true, all_preds, target_names=le.classes_)
    print("\nClassification Report:")
    print(report)

    # Save results
    results = {
        "model": "cross_attention_fusion",
        "best_val_accuracy": best_val_acc,
        "report": report,
    }
    with open(os.path.join(config.RESULTS_DIR, "cross_attention_results.json"), 'w') as f:
        json.dump(results, f, indent=2)
