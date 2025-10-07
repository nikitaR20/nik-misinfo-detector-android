import torch
from transformers import DistilBertTokenizer
from typing import Dict, List, Tuple
import pandas as pd
from torch.utils.data import Dataset

# Initialize tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Default label mapping
default_label_mapping = {
    "pants-fire": 0,
    "false": 0,
    "barely-true": 0,
    "half-true": 0,
    "mostly-true": 1,
    "true": 1,
    # Additional mappings for common cases
    "0": 0,
    "1": 1,
    0: 0,
    1: 1,
    "fake": 0,
    "real": 1,
    "misinformation": 0,
    "fact": 1
}

def preprocess_dataframe(
    df: pd.DataFrame,
    label_mapping: Dict = None,
    max_length: int = 256
) -> Tuple[Dict[str, List[int]], List[int]]:
    """
    Tokenizes text and maps labels to numbers.

    Args:
        df: DataFrame with columns ["label", "text"]
        label_mapping: Optional dict mapping labels to ints
        max_length: Max token length

    Returns:
        encodings: dict of tokenized text (input_ids, attention_mask)
        labels: list of integer labels
    """
    if label_mapping is None:
        label_mapping = default_label_mapping

    df = df.copy()
    
    # Convert labels to strings first for consistent mapping
    df["label"] = df["label"].astype(str).str.lower().str.strip()
    
    # Map labels to integers
    df["label"] = df["label"].map(label_mapping)
    
    # Drop any rows where mapping failed
    original_len = len(df)
    df = df.dropna(subset=["label"])
    if len(df) < original_len:
        print(f"⚠️  Dropped {original_len - len(df)} samples with unmapped labels")
    
    # Convert labels to int
    df["label"] = df["label"].astype(int)

    # Tokenize text
    texts = df["text"].tolist()
    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors=None  # Return lists, not tensors
    )

    labels = df["label"].tolist()

    return encodings, labels


class MisinformationDataset(Dataset):
    """PyTorch Dataset wrapper for misinformation detection"""
    
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.encodings['input_ids'][idx], dtype=torch.long),
            'attention_mask': torch.tensor(self.encodings['attention_mask'][idx], dtype=torch.long),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }