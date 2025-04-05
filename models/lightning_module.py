"""
PyTorch Lightning integration for Trittention models.

This module provides Lightning module implementations for easy training and
evaluation of various attention mechanisms with PyTorch Lightning.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader, Dataset

from models.attention import Attention
from models.trittention import Trittention
from models.trittention_cube import TrittentionCube
from models.local_trittention import LocalTrittention
from models.mixed_attention import MixedAttention
from models.sparse_trittention import SparseTrittention, WindowedTrittention


class TrittentionSequenceModel(nn.Module):
    """
    Enhanced base model class for sequence processing with trittention mechanisms.
    
    This model supports both raw feature processing and token embeddings.
    
    Attributes:
        embedding: Optional embedding layer for token-based inputs.
        attention: Attention mechanism module.
        input_projection: Linear projection for input features.
        output_projection: Linear projection for output features.
        layer_norm: Layer normalization.
        dropout: Dropout layer.
    """
    
    def __init__(
        self,
        attention_mechanism: nn.Module,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dropout_prob: float = 0.1,
        vocab_size: Optional[int] = None,
        embedding_dim: Optional[int] = None
    ):
        """
        Initialize the sequence model.
        
        Args:
            attention_mechanism: Attention module to use.
            input_size: Size of input features.
            hidden_size: Size of hidden layers.
            output_size: Size of output features.
            dropout_prob: Dropout probability.
            vocab_size: Size of vocabulary (for token-based inputs).
            embedding_dim: Size of token embeddings (if vocab_size is provided).
        """
        super().__init__()
        
        # Print initialization information in debug mode
        # print(f"Initializing TrittentionSequenceModel with:")
        # print(f"  input_size={input_size}, hidden_size={hidden_size}, output_size={output_size}")
        # print(f"  vocab_size={vocab_size}, embedding_dim={embedding_dim}")
        
        # Set up embedding layer if processing tokens
        self.use_embedding = vocab_size is not None
        if self.use_embedding:
            embedding_dim = embedding_dim or hidden_size // 2
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            # Input size becomes embedding_dim
            input_projection_in_features = embedding_dim
        else:
            self.embedding = None
            input_projection_in_features = input_size
        
        # Core model components
        self.attention = attention_mechanism
        self.input_projection = nn.Linear(input_projection_in_features, hidden_size)
        self.output_projection = nn.Linear(hidden_size, output_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
    
    def forward(
        self, 
        x: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for the model.
        
        Args:
            x: Input tensor that can be:
               - Token indices of shape [batch_size, seq_length] (for embedding)
               - Features of shape [batch_size, seq_length, input_size]
            attention_mask: Optional attention mask
            
        Returns:
            Output tensor of shape [batch_size, seq_length, output_size] or [batch_size, output_size]
        """
        batch_size = x.size(0)
        
        # Convert input to float if it's not already and not using embeddings
        if not self.use_embedding and x.dtype != torch.float32:
            x = x.float()
        
        # Apply embedding if using token inputs
        if self.use_embedding:
            # Ensure input is long type for embedding
            if x.dtype != torch.long:
                x = x.long()
            # Apply embedding [batch_size, seq_length] -> [batch_size, seq_length, embedding_dim]
            x = self.embedding(x)
        
        # Project input to hidden size
        hidden_states = self.input_projection(x)
        
        # Apply attention
        attention_output = self.attention(hidden_states, attention_mask)
        
        # Apply layer norm and residual connection
        normalized_output = self.layer_norm(attention_output + hidden_states)
        
        # Project to output size
        output = self.output_projection(self.dropout(normalized_output))
        
        return output


class TrittentionLightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for training and evaluating attention mechanisms.
    
    This module integrates with PyTorch Lightning for simplified training,
    evaluation, and experiment tracking.
    
    Attributes:
        model: The neural network model.
        learning_rate: Learning rate for optimization.
        weight_decay: Weight decay for regularization.
        optimizer_type: Type of optimizer to use.
        scheduler_type: Type of learning rate scheduler to use.
        save_attention_patterns: Whether to save attention patterns during validation.
    """
    
    def __init__(
        self,
        attention_type: str,
        config: Any,
        input_size: int,
        hidden_size: int,
        output_size: int,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.01,
        optimizer_type: str = 'adam',
        scheduler_type: Optional[str] = 'cosine',
        save_attention_patterns: bool = False,
        vocab_size: Optional[int] = None,  # For token-based datasets
        embedding_dim: Optional[int] = None  # For token-based datasets
    ):
        """
        Initialize the Lightning module.
        
        Args:
            attention_type: Type of attention mechanism to use.
                            Options: 'standard', 'trittention', 'trittention_cube',
                                    'local', 'mixed', 'sparse', 'windowed'
            config: Configuration object for the attention mechanism.
            input_size: Size of input features.
            hidden_size: Size of hidden layers.
            output_size: Size of output features.
            learning_rate: Learning rate for optimization.
            weight_decay: Weight decay for regularization.
            optimizer_type: Type of optimizer ('adam', 'adamw', or 'sgd').
            scheduler_type: Type of scheduler (None, 'cosine', 'linear', 'step').
            save_attention_patterns: Whether to save attention patterns.
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Create the attention mechanism
        self.attention_type = attention_type
        attention_mechanism = self._create_attention_mechanism(attention_type, config)
        
        # Debug model dimensions if needed
        # print(f"\nCreating model with input_size={input_size}, hidden_size={hidden_size}, output_size={output_size}")
        
        # Create the model
        self.model = TrittentionSequenceModel(
            attention_mechanism=attention_mechanism,
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            dropout_prob=getattr(config, 'hidden_dropout_prob', 0.1),
            vocab_size=vocab_size,
            embedding_dim=embedding_dim
        )
        
        # Save hyperparameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_type = optimizer_type
        self.scheduler_type = scheduler_type
        self.save_attention_patterns = save_attention_patterns
        
        # For storing metrics
        self.validation_step_outputs = []
        self.test_step_outputs = []
    
    def _create_attention_mechanism(self, attention_type: str, config: Any) -> nn.Module:
        """
        Create the specified attention mechanism.
        
        Args:
            attention_type: Type of attention mechanism.
            config: Configuration object.
            
        Returns:
            Instantiated attention mechanism module.
            
        Raises:
            ValueError: If the attention type is not recognized.
        """
        match attention_type.lower():
            case 'standard':
                return Attention(config)
            case 'trittention':
                return Trittention(config)
            case 'trittention_cube':
                return TrittentionCube(config)
            case 'local':
                return LocalTrittention(config)
            case 'mixed':
                return MixedAttention(config)
            case 'sparse':
                return SparseTrittention(config)
            case 'windowed':
                return WindowedTrittention(config)
            case _:
                raise ValueError(f"Unknown attention type: {attention_type}")
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for the Lightning module.
        
        Args:
            x: Input tensor
            attention_mask: Optional attention mask
            
        Returns:
            Model output
        """
        return self.model(x, attention_mask)
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Training step.
        
        Args:
            batch: Tuple of (inputs, targets)
            batch_idx: Index of the batch
            
        Returns:
            Loss tensor
        """
        inputs, targets = batch
        outputs = self(inputs)
        
        loss = F.mse_loss(outputs, targets)
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Validation step.
        
        Args:
            batch: Tuple of (inputs, targets)
            batch_idx: Index of the batch
            
        Returns:
            Dictionary of validation metrics
        """
        inputs, targets = batch
        
        # Print validation batch shapes only in debug mode
        # print(f"Validation batch shapes: inputs={inputs.shape}, targets={targets.shape}")
        
        # Handle different tensor shapes for ArithmeticDataset
        if len(inputs.shape) == 2 and inputs.shape[1] != self.model.input_projection.in_features:
            # Convert token sequences to embeddings or reshape as needed
            # This is for ArithmeticDataset where inputs are token sequences
            if hasattr(self.model.input_projection, 'in_features') and self.model.input_projection.in_features == inputs.shape[1]:
                # If input_size matches sequence length, we can use it directly
                pass
            elif hasattr(self.model.input_projection, 'in_features') and self.model.input_projection.in_features == 1:
                # Add a feature dimension
                inputs = inputs.unsqueeze(-1)
            else:
                # Create an embedding layer on the fly
                embedding_dim = self.model.input_projection.in_features
                vocab_size = max(inputs.max().item() + 1, 256)  # Ensure we cover all token IDs
                embedded_inputs = torch.nn.functional.one_hot(inputs.long(), num_classes=vocab_size).float()
                # Project to the expected input size
                inputs = embedded_inputs @ torch.randn(vocab_size, embedding_dim, device=inputs.device)
        
        outputs = self(inputs)
        
        loss = F.mse_loss(outputs, targets)
        
        # Calculate additional metrics
        mse = F.mse_loss(outputs, targets, reduction='none').mean(dim=1)
        mae = F.l1_loss(outputs, targets, reduction='none').mean(dim=1)
        
        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # Save outputs for epoch end processing
        metrics = {
            'val_loss': loss,
            'val_mse': mse.mean(),
            'val_mae': mae.mean(),
            'outputs': outputs.detach(),
            'targets': targets.detach()
        }
        
        # Save attention patterns if enabled
        if self.save_attention_patterns and hasattr(self.model.attention, 'get_attention_weights'):
            attention_weights = self.model.attention.get_attention_weights()
            if attention_weights is not None:
                metrics['attention_weights'] = attention_weights.detach()
        
        self.validation_step_outputs.append(metrics)
        
        return metrics
    
    def on_validation_epoch_end(self) -> None:
        """
        Called at the end of the validation epoch.
        """
        # Calculate and log epoch-level metrics
        if not self.validation_step_outputs:
            return
        
        # Calculate average metrics
        avg_val_loss = torch.stack([x['val_loss'] for x in self.validation_step_outputs]).mean()
        avg_val_mse = torch.stack([x['val_mse'] for x in self.validation_step_outputs]).mean()
        avg_val_mae = torch.stack([x['val_mae'] for x in self.validation_step_outputs]).mean()
        
        # Log to progress bar
        self.log('val_loss', avg_val_loss)
        self.log('val_mse', avg_val_mse)
        self.log('val_mae', avg_val_mae)
        
        # Save attention patterns visualization if enabled
        if self.save_attention_patterns:
            attention_patterns = [
                x['attention_weights'] for x in self.validation_step_outputs
                if 'attention_weights' in x
            ]
            
            if attention_patterns:
                self._save_attention_visualization(attention_patterns[0])
        
        # Clear outputs
        self.validation_step_outputs.clear()
    
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Test step.
        
        Args:
            batch: Tuple of (inputs, targets)
            batch_idx: Index of the batch
            
        Returns:
            Dictionary of test metrics
        """
        inputs, targets = batch
        outputs = self(inputs)
        
        loss = F.mse_loss(outputs, targets)
        
        # Calculate additional metrics
        mse = F.mse_loss(outputs, targets, reduction='none').mean(dim=1)
        mae = F.l1_loss(outputs, targets, reduction='none').mean(dim=1)
        
        # Log metrics
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        
        # Save outputs for epoch end processing
        metrics = {
            'test_loss': loss,
            'test_mse': mse.mean(),
            'test_mae': mae.mean(),
            'outputs': outputs.detach(),
            'targets': targets.detach()
        }
        
        self.test_step_outputs.append(metrics)
        
        return metrics
    
    def on_test_epoch_end(self) -> None:
        """
        Called at the end of the test epoch.
        """
        # Calculate and log epoch-level metrics
        if not self.test_step_outputs:
            return
        
        # Calculate average metrics
        avg_test_loss = torch.stack([x['test_loss'] for x in self.test_step_outputs]).mean()
        avg_test_mse = torch.stack([x['test_mse'] for x in self.test_step_outputs]).mean()
        avg_test_mae = torch.stack([x['test_mae'] for x in self.test_step_outputs]).mean()
        
        # Log to progress bar
        self.log('test_loss', avg_test_loss)
        self.log('test_mse', avg_test_mse)
        self.log('test_mae', avg_test_mae)
        
        # Clear outputs
        self.test_step_outputs.clear()
    
    def configure_optimizers(self) -> Union[Optimizer, Dict[str, Any]]:
        """
        Configure optimizers and schedulers.
        
        Returns:
            Optimizer or dictionary with optimizer and scheduler configuration.
        """
        # Configure optimizer
        match self.optimizer_type.lower():
            case 'adam':
                optimizer = torch.optim.Adam(
                    self.parameters(),
                    lr=self.learning_rate,
                    weight_decay=self.weight_decay
                )
            case 'adamw':
                optimizer = torch.optim.AdamW(
                    self.parameters(),
                    lr=self.learning_rate,
                    weight_decay=self.weight_decay
                )
            case 'sgd':
                optimizer = torch.optim.SGD(
                    self.parameters(),
                    lr=self.learning_rate,
                    momentum=0.9,
                    weight_decay=self.weight_decay
                )
            case _:
                raise ValueError(f"Unknown optimizer type: {self.optimizer_type}")
        
        # Configure scheduler
        if self.scheduler_type is None:
            return optimizer
        
        match self.scheduler_type.lower():
            case 'cosine':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=self.trainer.max_epochs,
                    eta_min=1e-6
                )
            case 'linear':
                scheduler = torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=1.0,
                    end_factor=0.1,
                    total_iters=self.trainer.max_epochs
                )
            case 'step':
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=self.trainer.max_epochs // 3,
                    gamma=0.1
                )
            case _:
                raise ValueError(f"Unknown scheduler type: {self.scheduler_type}")
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }
    
    def _save_attention_visualization(self, attention_weights: torch.Tensor) -> None:
        """
        Save visualization of attention weights.
        
        Args:
            attention_weights: Attention weights tensor to visualize.
        """
        import matplotlib.pyplot as plt
        
        # Create directory for visualizations
        save_dir = Path(self.trainer.log_dir) / 'attention_visualizations'
        os.makedirs(save_dir, exist_ok=True)
        
        # Get current epoch
        current_epoch = self.current_epoch
        
        # Select a single example and attention head
        if attention_weights.dim() > 3:
            # (batch_size, num_heads, seq_len, seq_len) -> (seq_len, seq_len)
            attention_map = attention_weights[0, 0].cpu().numpy()
        else:
            attention_map = attention_weights[0].cpu().numpy()
        
        # Create figure
        plt.figure(figsize=(10, 8))
        plt.imshow(attention_map, cmap='viridis')
        plt.colorbar()
        plt.title(f'{self.attention_type} Attention Pattern - Epoch {current_epoch}')
        plt.xlabel('Token Position')
        plt.ylabel('Token Position')
        
        # Save figure
        save_path = save_dir / f'attention_epoch_{current_epoch}.png'
        plt.savefig(save_path)
        plt.close()


# For backwards compatibility
# This allows existing code that imports TrittentionBaseModel to continue working
TrittentionBaseModel = TrittentionSequenceModel


class TrittentionDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning data module for Trittention experiments.
    
    This module handles data loading, preprocessing, and batch preparation for
    training, validation, and testing.
    """
    
    def __init__(
        self,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        val_split: float = 0.2,
        seed: int = 42
    ):
        """
        Initialize the data module.
        
        Args:
            train_dataset: Training dataset.
            val_dataset: Validation dataset (optional).
            test_dataset: Test dataset (optional).
            batch_size: Batch size for dataloaders.
            num_workers: Number of workers for dataloaders.
            val_split: Fraction of training data to use for validation if val_dataset is None.
            seed: Random seed for reproducibility.
        """
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.seed = seed
    
    def setup(self, stage: Optional[str] = None) -> None:
        """
        Prepare datasets for different stages.
        
        Args:
            stage: Stage ('fit', 'validate', 'test', or None).
        """
        # Set seed for reproducibility
        torch.manual_seed(self.seed)
        
        # Split training dataset into train and validation if validation dataset not provided
        if stage in ('fit', None) and self.val_dataset is None and self.val_split > 0:
            val_size = int(len(self.train_dataset) * self.val_split)
            train_size = len(self.train_dataset) - val_size
            
            self.train_subset, self.val_subset = torch.utils.data.random_split(
                self.train_dataset,
                [train_size, val_size]
            )
        else:
            self.train_subset = self.train_dataset
            self.val_subset = self.val_dataset
    
    def train_dataloader(self) -> DataLoader:
        """
        Create the training dataloader.
        
        Returns:
            Training DataLoader.
        """
        return DataLoader(
            self.train_subset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self) -> Optional[DataLoader]:
        """
        Create the validation dataloader.
        
        Returns:
            Validation DataLoader or None if no validation dataset available.
        """
        if self.val_subset is None:
            return None
        
        return DataLoader(
            self.val_subset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def test_dataloader(self) -> Optional[DataLoader]:
        """
        Create the test dataloader.
        
        Returns:
            Test DataLoader or None if no test dataset available.
        """
        if self.test_dataset is None:
            return None
        
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
