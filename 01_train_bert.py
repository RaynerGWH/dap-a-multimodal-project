"""
Step 1: Fine-tune BERT on news headlines for classification.
Saves the trained model to saved_models/bert_news_classifier/
"""

import torch
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, BertTokenizer
from tqdm import tqdm
import json

import config
from data_utils import load_data, get_label_encoder, get_train_val_split, TextDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================================
# Load and prepare data
# ============================================================
df = load_data(config.TRAIN_JSON, config.IMAGE_DIR)
df, le = get_label_encoder(df)
train_df, val_df = get_train_val_split(df)

# Save label encoder for later use
import pickle
with open(f"{config.SAVE_DIR}/label_encoder.pkl", 'wb') as f:
    pickle.dump(le, f)

# Load class weights (computed by 00_eda.py)
class_weights = torch.load(f"{config.SAVE_DIR}/class_weights.pt").to(device)
print(f"Using class weights for {len(class_weights)} classes")

# ============================================================
# Load model and tokenizer
# ============================================================
tokenizer = BertTokenizer.from_pretrained(config.BERT_MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(
    config.BERT_MODEL_NAME, num_labels=config.NUM_CLASSES
)
model.to(device)

# ============================================================
# Create data loaders
# ============================================================
train_dataset = TextDataset(train_df, tokenizer, max_length=config.MAX_TEXT_LENGTH)
val_dataset = TextDataset(val_df, tokenizer, max_length=config.MAX_TEXT_LENGTH)

train_loader = DataLoader(train_dataset, batch_size=config.BERT_BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.BERT_BATCH_SIZE, shuffle=False)

# ============================================================
# Training
# ============================================================
optimizer = torch.optim.AdamW(model.parameters(), lr=config.BERT_LR)
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

best_val_acc = 0

for epoch in range(config.BERT_EPOCHS):
    # Train
    model.train()
    total_loss, correct, total = 0, 0, 0

    for batch in tqdm(train_loader, desc=f"BERT Epoch {epoch+1}/{config.BERT_EPOCHS}"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predictions = torch.argmax(outputs.logits, dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total
    train_loss = total_loss / len(train_loader)

    # Evaluate
    model.eval()
    val_correct, val_total = 0, 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=1)
            val_correct += (predictions == labels).sum().item()
            val_total += labels.size(0)

    val_acc = val_correct / val_total

    print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        model.save_pretrained(config.BERT_SAVE_DIR)
        tokenizer.save_pretrained(config.BERT_SAVE_DIR)
        print(f"  ✓ Saved best model (val_acc: {val_acc:.4f})")

print(f"\nBERT training complete! Best val accuracy: {best_val_acc:.4f}")
