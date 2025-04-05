"""
Text classification example using trittention.

This example demonstrates how to use Trittention-Transformer for text classification
tasks, comparing different attention mechanisms for effectiveness.
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

# Add parent directory to Python path
script_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(script_dir))

from config.cfgs import TrittentionConfig
from models import (
    Attention,
    Trittention,
    SparseTrittention,
    WindowedTrittention
)
from models.lightning_module import TrittentionLightningModule, TrittentionDataModule


class TextClassificationDataset(Dataset):
    """
    Dataset for text classification tasks.
    
    This dataset handles loading and preprocessing text data for classification.
    It converts text to sequences of token IDs and pairs them with class labels.
    
    Attributes:
        texts (list): List of text samples
        labels (list): List of corresponding labels
        vocab (dict): Mapping from tokens to IDs
        max_length (int): Maximum sequence length
    """
    
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        vocab: Optional[Dict[str, int]] = None,
        max_length: int = 128,
        build_vocab_from_data: bool = True
    ):
        """
        Initialize the dataset.
        
        Args:
            texts: List of text samples
            labels: List of corresponding class labels
            vocab: Vocabulary mapping (if None, will be built from data)
            max_length: Maximum sequence length
            build_vocab_from_data: Whether to build vocab from the provided texts
        """
        self.texts = texts
        self.labels = labels
        self.max_length = max_length
        
        assert len(texts) == len(labels), "Number of texts and labels must match"
        
        # Build or use provided vocabulary
        if vocab is None and build_vocab_from_data:
            self.vocab = self._build_vocabulary(texts)
        else:
            self.vocab = vocab or {}
        
        # Add special tokens if they don't exist
        self._ensure_special_tokens()
    
    def _build_vocabulary(self, texts: List[str]) -> Dict[str, int]:
        """
        Build vocabulary from texts.
        
        Args:
            texts: List of text samples
            
        Returns:
            Vocabulary mapping from tokens to IDs
        """
        # Start with special tokens
        vocab = {
            "<PAD>": 0,
            "<UNK>": 1,
            "<BOS>": 2,
            "<EOS>": 3
        }
        
        # Add tokens from texts
        token_id = len(vocab)
        for text in texts:
            for token in self._tokenize(text):
                if token not in vocab:
                    vocab[token] = token_id
                    token_id += 1
        
        return vocab
    
    def _ensure_special_tokens(self):
        """Ensure special tokens are in vocabulary."""
        special_tokens = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
        token_id = max(self.vocab.values()) + 1 if self.vocab else 0
        
        for token in special_tokens:
            if token not in self.vocab:
                self.vocab[token] = token_id
                token_id += 1
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text (simple whitespace tokenization).
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        # Simple tokenization by whitespace and punctuation
        # In a real application, you would use a proper tokenizer
        tokens = []
        for word in text.split():
            # Handle basic punctuation
            if word.endswith(('.', ',', '!', '?', ':', ';')):
                tokens.append(word[:-1])
                tokens.append(word[-1])
            else:
                tokens.append(word)
        
        return tokens
    
    def _convert_to_ids(self, tokens: List[str]) -> List[int]:
        """
        Convert tokens to IDs.
        
        Args:
            tokens: List of tokens
            
        Returns:
            List of token IDs
        """
        return [self.vocab.get(token, self.vocab["<UNK>"]) for token in tokens]
    
    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample by index.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (input_ids, label)
        """
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenize and convert to IDs
        tokens = self._tokenize(text)
        token_ids = self._convert_to_ids(tokens)
        
        # Add special tokens
        token_ids = [self.vocab["<BOS>"]] + token_ids + [self.vocab["<EOS>"]]
        
        # Truncate or pad sequence
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
        else:
            token_ids = token_ids + [self.vocab["<PAD>"]] * (self.max_length - len(token_ids))
        
        # Convert to tensors
        input_ids = torch.tensor(token_ids, dtype=torch.long)
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return input_ids, label_tensor
    
    @classmethod
    def from_csv(
        cls,
        file_path: str,
        text_column: str = "text",
        label_column: str = "label",
        max_length: int = 128
    ) -> "TextClassificationDataset":
        """
        Create dataset from CSV file.
        
        Args:
            file_path: Path to CSV file
            text_column: Column name for text
            label_column: Column name for labels
            max_length: Maximum sequence length
            
        Returns:
            TextClassificationDataset
        """
        df = pd.read_csv(file_path)
        
        # Ensure columns exist
        if text_column not in df.columns:
            raise ValueError(f"Text column '{text_column}' not found in CSV file")
        if label_column not in df.columns:
            raise ValueError(f"Label column '{label_column}' not found in CSV file")
        
        # Extract texts and labels
        texts = df[text_column].tolist()
        
        # Convert labels to integers if they're not already
        if not pd.api.types.is_numeric_dtype(df[label_column]):
            # Map unique labels to integers
            label_map = {label: i for i, label in enumerate(df[label_column].unique())}
            labels = [label_map[label] for label in df[label_column]]
        else:
            labels = df[label_column].tolist()
        
        return cls(texts, labels, max_length=max_length)


class TextClassificationModel(pl.LightningModule):
    """
    Text classification model using attention mechanisms.
    
    This model embeds input tokens, passes them through an attention layer,
    and makes classification predictions.
    
    Attributes:
        vocab_size (int): Size of the vocabulary
        embedding_dim (int): Dimension of token embeddings
        num_classes (int): Number of output classes
        attention (nn.Module): Attention mechanism module
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_size: int,
        num_classes: int,
        attention_type: str = "standard",
        num_attention_heads: int = 4,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.01
    ):
        """
        Initialize the model.
        
        Args:
            vocab_size: Size of the vocabulary
            embedding_dim: Dimension of token embeddings
            hidden_size: Size of hidden layers
            num_classes: Number of output classes
            attention_type: Type of attention mechanism to use
            num_attention_heads: Number of attention heads
            dropout: Dropout probability
            learning_rate: Learning rate for optimization
            weight_decay: Weight decay for regularization
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(2048, embedding_dim)  # Max position embedding
        
        # Create attention mechanism
        config = TrittentionConfig(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_probs_dropout_prob=dropout,
            hidden_dropout_prob=dropout
        )
        
        # Set attention-specific parameters
        if attention_type == "sparse":
            config.sparsity_threshold = 0.1
        elif attention_type in ["local", "windowed"]:
            config.window_size = 128
        
        # Initialize attention mechanism
        if attention_type == "standard":
            self.attention = Attention(config)
        elif attention_type == "trittention":
            self.attention = Trittention(config)
        elif attention_type == "sparse":
            self.attention = SparseTrittention(config)
        elif attention_type in ["windowed", "local"]:
            self.attention = WindowedTrittention(config)
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")
        
        # Output layers
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)
        
        # Save hyperparameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the model.
        
        Args:
            input_ids: Input token IDs of shape [batch_size, seq_length]
            
        Returns:
            Logits of shape [batch_size, num_classes]
        """
        # Get sequence length
        seq_length = input_ids.size(1)
        
        # Create position IDs
        position_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(0)
        
        # Embedding
        token_embeddings = self.embedding(input_ids)  # [batch_size, seq_length, embedding_dim]
        position_embeddings = self.position_embedding(position_ids)  # [1, seq_length, embedding_dim]
        
        # Combine embeddings
        embeddings = token_embeddings + position_embeddings
        
        # Apply attention
        attention_output = self.attention(embeddings)
        
        # Apply layer normalization and dropout
        hidden_states = self.layer_norm(attention_output)
        hidden_states = self.dropout(hidden_states)
        
        # Pool hidden states (use [CLS] token or average)
        pooled_output = hidden_states.mean(dim=1)  # [batch_size, hidden_size]
        
        # Classification
        logits = self.classifier(pooled_output)  # [batch_size, num_classes]
        
        return logits
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Training step.
        
        Args:
            batch: Tuple of (input_ids, labels)
            batch_idx: Index of the batch
            
        Returns:
            Loss tensor
        """
        input_ids, labels = batch
        logits = self(input_ids)
        
        loss = F.cross_entropy(logits, labels)
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Validation step.
        
        Args:
            batch: Tuple of (input_ids, labels)
            batch_idx: Index of the batch
            
        Returns:
            Dictionary of validation metrics
        """
        input_ids, labels = batch
        logits = self(input_ids)
        
        loss = F.cross_entropy(logits, labels)
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        
        # Log metrics
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True)
        
        return {'val_loss': loss, 'val_acc': acc, 'preds': preds, 'labels': labels}
    
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Test step.
        
        Args:
            batch: Tuple of (input_ids, labels)
            batch_idx: Index of the batch
            
        Returns:
            Dictionary of test metrics
        """
        input_ids, labels = batch
        logits = self(input_ids)
        
        # Calculate metrics
        preds = torch.argmax(logits, dim=1)
        loss = F.cross_entropy(logits, labels)
        acc = (preds == labels).float().mean()
        
        # Log metrics
        self.log('test_loss', loss, on_epoch=True)
        self.log('test_acc', acc, on_epoch=True)
        
        return {'test_loss': loss, 'test_acc': acc, 'preds': preds, 'labels': labels}
    
    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configure optimizers.
        
        Returns:
            Optimizer
        """
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
    
    def on_test_epoch_end(self) -> None:
        """Calculate and log test metrics at the end of the epoch."""
        preds = torch.cat([x['preds'] for x in self.test_step_outputs])
        labels = torch.cat([x['labels'] for x in self.test_step_outputs])
        
        # Convert to numpy for sklearn metrics
        preds_np = preds.cpu().numpy()
        labels_np = labels.cpu().numpy()
        
        # Calculate metrics
        acc = accuracy_score(labels_np, preds_np)
        precision = precision_score(labels_np, preds_np, average='macro')
        recall = recall_score(labels_np, preds_np, average='macro')
        f1 = f1_score(labels_np, preds_np, average='macro')
        
        # Log metrics
        self.log('test_accuracy', acc)
        self.log('test_precision', precision)
        self.log('test_recall', recall)
        self.log('test_f1', f1)
        
        print(f"\nTest Results:")
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")


def load_and_prepare_data(
    data_path: str,
    text_column: str = "text",
    label_column: str = "label",
    max_length: int = 128,
    batch_size: int = 32,
    num_workers: int = 4,
    val_split: float = 0.2,
    test_split: float = 0.1
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int], int]:
    """
    Load and prepare data for text classification.
    
    Args:
        data_path: Path to data file (CSV)
        text_column: Column name for text
        label_column: Column name for labels
        max_length: Maximum sequence length
        batch_size: Batch size for dataloaders
        num_workers: Number of dataloader workers
        val_split: Validation split ratio
        test_split: Test split ratio
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader, vocab, num_classes)
    """
    # Load dataset
    dataset = TextClassificationDataset.from_csv(
        data_path,
        text_column=text_column,
        label_column=label_column,
        max_length=max_length
    )
    
    # Get vocabulary and number of classes
    vocab = dataset.vocab
    num_classes = max(dataset.labels) + 1
    
    # Split dataset
    train_ratio = 1.0 - val_split - test_split
    train_size = int(train_ratio * len(dataset))
    val_size = int(val_split * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader, vocab, num_classes


def train_and_evaluate(
    data_path: str,
    attention_type: str = "standard",
    embedding_dim: int = 128,
    hidden_size: int = 256,
    num_attention_heads: int = 4,
    max_length: int = 128,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.01,
    dropout: float = 0.1,
    max_epochs: int = 10,
    patience: int = 3,
    save_dir: str = "./results/text_classification",
    use_gpu: bool = True
) -> pl.LightningModule:
    """
    Train and evaluate text classification model.
    
    Args:
        data_path: Path to data file (CSV)
        attention_type: Type of attention mechanism to use
        embedding_dim: Dimension of token embeddings
        hidden_size: Size of hidden layers
        num_attention_heads: Number of attention heads
        max_length: Maximum sequence length
        batch_size: Batch size for training
        learning_rate: Learning rate for optimization
        weight_decay: Weight decay for regularization
        dropout: Dropout probability
        max_epochs: Maximum number of training epochs
        patience: Patience for early stopping
        save_dir: Directory to save results
        use_gpu: Whether to use GPU if available
        
    Returns:
        Trained model
    """
    # Set up save directory
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and prepare data
    train_loader, val_loader, test_loader, vocab, num_classes = load_and_prepare_data(
        data_path=data_path,
        max_length=max_length,
        batch_size=batch_size
    )
    
    # Create model
    model = TextClassificationModel(
        vocab_size=len(vocab),
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        num_classes=num_classes,
        attention_type=attention_type,
        num_attention_heads=num_attention_heads,
        dropout=dropout,
        learning_rate=learning_rate,
        weight_decay=weight_decay
    )
    
    # Set up callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=save_dir / "checkpoints",
            filename=f"{attention_type}_model",
            monitor="val_acc",
            mode="max",
            save_top_k=1
        ),
        EarlyStopping(
            monitor="val_acc",
            mode="max",
            patience=patience
        )
    ]
    
    # Set up logger
    logger = TensorBoardLogger(
        save_dir=save_dir / "logs",
        name=attention_type
    )
    
    # Set up trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=callbacks,
        logger=logger,
        accelerator="gpu" if use_gpu and torch.cuda.is_available() else "cpu",
        devices=1,
        deterministic=True
    )
    
    # Train model
    trainer.fit(model, train_loader, val_loader)
    
    # Test model
    trainer.test(model, test_loader)
    
    return model


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Text classification with different attention mechanisms")
    
    # Data parameters
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to data file (CSV)")
    parser.add_argument("--text_column", type=str, default="text",
                        help="Column name for text")
    parser.add_argument("--label_column", type=str, default="label",
                        help="Column name for labels")
    
    # Model parameters
    parser.add_argument("--attention_type", type=str, default="standard",
                        choices=["standard", "trittention", "sparse", "windowed"],
                        help="Type of attention mechanism to use")
    parser.add_argument("--embedding_dim", type=int, default=128,
                        help="Dimension of token embeddings")
    parser.add_argument("--hidden_size", type=int, default=256,
                        help="Size of hidden layers")
    parser.add_argument("--num_attention_heads", type=int, default=4,
                        help="Number of attention heads")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout probability")
    
    # Training parameters
    parser.add_argument("--max_length", type=int, default=128,
                        help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="Learning rate for optimization")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay for regularization")
    parser.add_argument("--max_epochs", type=int, default=10,
                        help="Maximum number of training epochs")
    parser.add_argument("--patience", type=int, default=3,
                        help="Patience for early stopping")
    
    # Output parameters
    parser.add_argument("--save_dir", type=str, default="./results/text_classification",
                        help="Directory to save results")
    parser.add_argument("--gpu", action="store_true",
                        help="Use GPU if available")
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    print(f"Training text classification model with {args.attention_type} attention")
    print(f"Data path: {args.data_path}")
    print(f"Model parameters: embedding_dim={args.embedding_dim}, hidden_size={args.hidden_size}, "
          f"num_attention_heads={args.num_attention_heads}")
    
    # Train and evaluate model
    model = train_and_evaluate(
        data_path=args.data_path,
        attention_type=args.attention_type,
        embedding_dim=args.embedding_dim,
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_attention_heads,
        max_length=args.max_length,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        max_epochs=args.max_epochs,
        patience=args.patience,
        save_dir=args.save_dir,
        use_gpu=args.gpu
    )


if __name__ == "__main__":
    main()
