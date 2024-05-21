# utils/data_utils.py

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple

class TextDataset(Dataset):
    """Dataset class for text data."""
    
    def __init__(self, texts: List[str], labels: List[None] = None):
        """Initialize the dataset with texts and labels."""
        self.texts = texts
        self.labels = labels
        
    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Tuple[str, None]:
        """Return a single sample from the dataset."""
        return self.texts[idx], self.labels[idx]

def load_dataset(file_path: str) -> Tuple[List[str], List[None]]:
    """Load a dataset from a file."""
    texts = []
    labels = []
    with open(file_path, 'r') as file:
        for line in file:
            text = line.strip()
            texts.append(text)
            labels.append(None)  # no labels in your data files
    return texts, labels

def preprocess_data(texts: List[str], tokenizer) -> Tuple[List[int], List[int]]:
    """Preprocess the text data using the provided tokenizer."""
    input_ids = []
    attention_masks = []
    for text in texts:
        encoded_dict = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=128,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    return input_ids, attention_masks

def create_data_loader(dataset: Dataset, batch_size: int, shuffle: bool = True) -> DataLoader:
    """Create a data loader from a dataset."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)