# utils/evaluation_utils.py

import torch
from typing import Dict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(model, data_loader, device) -> Dict[str, float]:
    """Evaluate the model on the provided data loader."""
    model.eval()
    predictions = []
    true_labels = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids, attention_masks, labels = batch
            input_ids = input_ids.to(device)
            attention_masks = attention_masks.to(device)
            outputs = model(input_ids, attention_mask=attention_masks)
            _, preds = torch.max(outputs, dim=1)
            predictions.extend(preds.tolist())
            true_labels.extend(labels.tolist())
    return calculate_metrics(true_labels, predictions)

def calculate_metrics(true_labels, predictions) -> Dict[str, float]:
    """Calculate evaluation metrics."""
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='macro')
    recall = recall_score(true_labels, predictions, average='macro')
    f1 = f1_score(true_labels, predictions, average='macro')
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }