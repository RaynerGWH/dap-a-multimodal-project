"""
Step 2: Fine-tune ViT on news images for classification.
Saves the trained model to saved_models/vit_news_classifier/
"""

import torch
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification, ViTImageProcessor
from tqdm import tqdm

import config
from data_utils import load_data, get_label_encoder, get_train_val_split, ImageDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================================
# Load and prepare data
# ============================================================
df = load_data(config.TRAIN_JSON, config.IMAGE_DIR)
df, le = get_label_encoder(df)
train_df, val_df = get_train_val_split(df)

# ============================================================
# Load model and processor
# ============================================================
processor = ViTImageProcessor.from_pretrained(config.VIT_MODEL_NAME)
model = ViTForImageClassification.from_pretrained(
    config.VIT_MODEL_NAME,
    num_labels=config.NUM_CLASSES,
    ignore_mismatched_sizes=True  # classifier head size will differ
)
model.to(device)

# ============================================================
# Create data loaders
# ============================================================
train_dataset = ImageDataset(train_df, config.IMAGE_DIR, processor)
val_dataset = ImageDataset(val_df, config.IMAGE_DIR, processor)

train_loader = DataLoader(train_dataset, batch_size=config.VIT_BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=config.VIT_BATCH_SIZE, shuffle=False, num_workers=4)

# ============================================================
# Training
# ============================================================
# Load class weights (computed by 00_eda.py)
class_weights = torch.load(f"{config.SAVE_DIR}/class_weights.pt").to(device)
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

optimizer = torch.optim.AdamW(model.parameters(), lr=config.VIT_LR)

best_val_acc = 0

for epoch in range(config.VIT_EPOCHS):
    # Train
    model.train()
    total_loss, correct, total = 0, 0, 0

    for batch in tqdm(train_loader, desc=f"ViT Epoch {epoch+1}/{config.VIT_EPOCHS}"):
        pixel_values = batch['pixel_values'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(pixel_values=pixel_values)
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
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['label'].to(device)

            outputs = model(pixel_values=pixel_values)
            predictions = torch.argmax(outputs.logits, dim=1)
            val_correct += (predictions == labels).sum().item()
            val_total += labels.size(0)

    val_acc = val_correct / val_total

    print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        model.save_pretrained(config.VIT_SAVE_DIR)
        processor.save_pretrained(config.VIT_SAVE_DIR)
        print(f"  ✓ Saved best model (val_acc: {val_acc:.4f})")

print(f"\nViT training complete! Best val accuracy: {best_val_acc:.4f}")
