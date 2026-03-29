"""
Shared data loading utilities used by all training scripts.
"""

import json
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from PIL import Image
import torch
from torch.utils.data import Dataset
import config


def load_data(json_path, image_dir=None):
    """Load NYTimes JSON and filter to samples with available images."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df['image_filename'] = df['image_id'] + '.jpg'

    if image_dir:
        available = set(os.listdir(image_dir))
        before = len(df)
        df = df[df['image_filename'].isin(available)]
        print(f"Loaded {json_path}: {before} total, {len(df)} with images")

    return df


def get_label_encoder(df):
    """Create and fit a label encoder on the 'section' column."""
    le = LabelEncoder()
    df = df.copy()
    df['label'] = le.fit_transform(df['section'])
    print(f"Classes ({len(le.classes_)}): {list(le.classes_)}")
    return df, le


def get_train_val_split(df, test_size=0.2):
    """Split dataframe into train/val with stratification."""
    train_df, val_df = train_test_split(
        df, test_size=test_size, stratify=df['label'], random_state=config.RANDOM_SEED
    )
    print(f"Train: {len(train_df)}, Val: {len(val_df)}")
    return train_df, val_df


class TextDataset(Dataset):
    """Dataset for BERT text-only training."""
    def __init__(self, df, tokenizer, max_length=32):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        encoding = self.tokenizer(
            row['headline'], max_length=self.max_length,
            padding='max_length', truncation=True, return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(row['label'], dtype=torch.long)
        }


class ImageDataset(Dataset):
    """Dataset for ViT image-only training."""
    def __init__(self, df, image_dir, processor):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row['image_filename'])
        image = Image.open(img_path).convert('RGB')
        processed = self.processor(images=image, return_tensors="pt")
        return {
            'pixel_values': processed['pixel_values'].squeeze(0),
            'label': torch.tensor(row['label'], dtype=torch.long)
        }


class MultimodalDataset(Dataset):
    """Dataset for multimodal (text + image) models."""
    def __init__(self, df, image_dir, tokenizer, image_processor, max_length=32):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        encoding = self.tokenizer(
            row['headline'], max_length=self.max_length,
            padding='max_length', truncation=True, return_tensors='pt'
        )

        img_path = os.path.join(self.image_dir, row['image_filename'])
        image = Image.open(img_path).convert('RGB')
        processed_img = self.image_processor(images=image, return_tensors="pt")

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'pixel_values': processed_img['pixel_values'].squeeze(0),
            'label': torch.tensor(row['label'], dtype=torch.long)
        }
