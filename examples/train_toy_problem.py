# examples/train_toy_problem.py

import torch
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer
import sys
import os

# Add the parent directory to the Python path
parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, parent_dir)

from models.trittention import Trittention
from config.cfgs import TrittentionConfig
from utils.data_utils import load_dataset, TextDataset, preprocess_data
from utils.evaluation_utils import evaluate_model

def main():
    # Set device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Load and preprocess data
    train_texts, _ = load_dataset("data/toy_problems/arithmetic_operations.txt")
    train_input_ids, train_attention_masks = preprocess_data(train_texts, tokenizer)
    train_dataset = TextDataset(train_input_ids, train_attention_masks)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # Define model and optimizer
    config = TrittentionConfig(hidden_size=128, num_attention_heads=4, num_hidden_layers=2)
    model = Trittention(config).to(device)
    optimizer = AdamW(model.parameters(), lr=1e-4)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            input_ids, attention_masks = batch
            input_ids = input_ids.to(device)
            attention_masks = attention_masks.to(device)

            outputs = model(input_ids, attention_mask=attention_masks)
            loss = torch.nn.functional.cross_entropy(outputs, torch.zeros_like(outputs))  # dummy labels

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    # Evaluate the trained model
    test_texts, _ = load_dataset("data/toy_problems/arithmetic_operations_test.txt")
    test_input_ids, test_attention_masks = preprocess_data(test_texts, tokenizer)
    test_dataset = TextDataset(test_input_ids, test_attention_masks)
    test_loader = DataLoader(test_dataset, batch_size=16)

    metrics = evaluate_model(model, test_loader, device)
    print("Test Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main()