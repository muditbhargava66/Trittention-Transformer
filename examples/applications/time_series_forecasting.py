"""
Time series forecasting example using trittention.

This example demonstrates how to use Trittention-Transformer for time series forecasting
tasks, comparing different attention mechanisms for effectiveness.
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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
from models.lightning_module import TrittentionLightningModule


class TimeSeriesDataset(Dataset):
    """
    Dataset for time series forecasting.
    
    This dataset handles loading and preprocessing time series data.
    It creates sliding windows of sequences and corresponding target values.
    
    Attributes:
        data (numpy.ndarray): Time series data
        sequence_length (int): Length of input sequences
        forecast_horizon (int): Number of steps to forecast
        scaler (object): Scaler for data normalization
    """
    
    def __init__(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        sequence_length: int = 96,  # 24 hours (hourly data)
        forecast_horizon: int = 24,  # Predict next 24 hours
        target_column: Optional[Union[str, int]] = None,
        feature_columns: Optional[List[Union[str, int]]] = None,
        normalize: bool = True,
        stride: int = 1
    ):
        """
        Initialize the dataset.
        
        Args:
            data: Time series data (numpy array or pandas DataFrame)
            sequence_length: Length of input sequences
            forecast_horizon: Number of future steps to predict
            target_column: Column to predict (if DataFrame)
            feature_columns: Columns to use as features (if DataFrame)
            normalize: Whether to normalize the data
            stride: Stride for sliding window
        """
        # Convert DataFrame to numpy array if needed
        if isinstance(data, pd.DataFrame):
            # Handle target and feature columns
            if target_column is None:
                target_column = data.columns[-1]
            
            if feature_columns is None:
                feature_columns = [col for col in data.columns if col != target_column]
            
            # Extract features and target
            features = data[feature_columns].values
            target = data[target_column].values.reshape(-1, 1)
            
            # Concatenate for multi-feature forecasting
            self.data = np.concatenate([features, target], axis=1)
            self.target_idx = -1  # Last column is target
        else:
            self.data = data
            self.target_idx = -1 if target_column is None else target_column
        
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.stride = stride
        
        # Create scaler for normalization
        self.scaler = StandardScaler() if normalize else None
        
        # Normalize data if requested
        if normalize:
            self.data = self.scaler.fit_transform(self.data)
        
        # Calculate number of samples
        self.num_samples = (len(self.data) - sequence_length - forecast_horizon) // stride + 1
    
    def __len__(self) -> int:
        """Return the number of samples."""
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample by index.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (input_sequence, target_sequence)
        """
        # Calculate start index
        start_idx = idx * self.stride
        
        # Extract sequence and target
        sequence = self.data[start_idx:start_idx + self.sequence_length]
        target = self.data[start_idx + self.sequence_length:start_idx + self.sequence_length + self.forecast_horizon]
        
        # Extract only target column for prediction
        if self.target_idx is not None:
            target = target[:, self.target_idx].reshape(-1)
        
        # Convert to tensors
        sequence_tensor = torch.tensor(sequence, dtype=torch.float32)
        target_tensor = torch.tensor(target, dtype=torch.float32)
        
        return sequence_tensor, target_tensor
    
    def get_feature_dim(self) -> int:
        """Get the number of features."""
        return self.data.shape[1]
    
    def get_target_dim(self) -> int:
        """Get the dimension of target."""
        return 1 if self.target_idx is not None else self.data.shape[1]
    
    def inverse_transform(self, normalized_data: np.ndarray, is_target: bool = False) -> np.ndarray:
        """
        Inverse transform normalized data back to original scale.
        
        Args:
            normalized_data: Normalized data
            is_target: Whether the data is target values
            
        Returns:
            Data in original scale
        """
        if self.scaler is None:
            return normalized_data
        
        # For target only, we need to recreate the full matrix
        if is_target and self.target_idx is not None:
            # Create a dummy matrix with zeros
            dummy = np.zeros((normalized_data.shape[0], self.data.shape[1]))
            # Put the target values in the target column
            dummy[:, self.target_idx] = normalized_data
            # Inverse transform
            result = self.scaler.inverse_transform(dummy)
            # Return only the target column
            return result[:, self.target_idx]
        
        # For full sequences
        return self.scaler.inverse_transform(normalized_data)
    
    @classmethod
    def from_csv(
        cls,
        file_path: str,
        sequence_length: int = 96,
        forecast_horizon: int = 24,
        target_column: Optional[str] = None,
        feature_columns: Optional[List[str]] = None,
        date_column: Optional[str] = None,
        normalize: bool = True
    ) -> "TimeSeriesDataset":
        """
        Create dataset from CSV file.
        
        Args:
            file_path: Path to CSV file
            sequence_length: Length of input sequences
            forecast_horizon: Number of future steps to predict
            target_column: Column to predict
            feature_columns: Columns to use as features
            date_column: Column with dates (for indexing)
            normalize: Whether to normalize the data
            
        Returns:
            TimeSeriesDataset
        """
        # Read data
        df = pd.read_csv(file_path)
        
        # Set date as index if specified
        if date_column is not None:
            if date_column in df.columns:
                df[date_column] = pd.to_datetime(df[date_column])
                df.set_index(date_column, inplace=True)
        
        # Select columns
        if feature_columns is not None:
            if target_column is not None and target_column not in feature_columns:
                columns = feature_columns + [target_column]
            else:
                columns = feature_columns
            df = df[columns]
        
        return cls(
            data=df,
            sequence_length=sequence_length,
            forecast_horizon=forecast_horizon,
            target_column=target_column,
            feature_columns=feature_columns,
            normalize=normalize
        )


class TimeSeriesModel(pl.LightningModule):
    """
    Time series forecasting model using attention mechanisms.
    
    This model processes time series data through an attention layer
    and outputs future predictions.
    
    Attributes:
        input_dim (int): Number of input features
        hidden_size (int): Size of hidden layers
        output_dim (int): Number of output features
        forecast_horizon (int): Number of future steps to predict
        attention (nn.Module): Attention mechanism module
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        output_dim: int,
        forecast_horizon: int,
        attention_type: str = "standard",
        num_attention_heads: int = 4,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.01
    ):
        """
        Initialize the model.
        
        Args:
            input_dim: Number of input features
            hidden_size: Size of hidden layers
            output_dim: Number of output features
            forecast_horizon: Number of future steps to predict
            attention_type: Type of attention mechanism to use
            num_attention_heads: Number of attention heads
            dropout: Dropout probability
            learning_rate: Learning rate for optimization
            weight_decay: Weight decay for regularization
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.output_dim = output_dim
        self.forecast_horizon = forecast_horizon
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_size)
        
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
            config.window_size = min(128, forecast_horizon * 2)
        
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
        self.forecast_head = nn.Linear(hidden_size, forecast_horizon * output_dim)
        
        # Save hyperparameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # For tracking metrics
        self.validation_step_outputs = []
        self.test_step_outputs = []
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the model.
        
        Args:
            x: Input tensor of shape [batch_size, seq_length, input_dim]
            
        Returns:
            Forecast of shape [batch_size, forecast_horizon, output_dim]
        """
        # Project input to hidden size
        hidden_states = self.input_projection(x)  # [batch_size, seq_length, hidden_size]
        
        # Apply attention
        attention_output = self.attention(hidden_states)  # [batch_size, seq_length, hidden_size]
        
        # Apply layer normalization and dropout
        normalized = self.layer_norm(attention_output + hidden_states)  # Residual connection
        normalized = self.dropout(normalized)
        
        # Aggregate sequence (use last state or global pooling)
        seq_representation = normalized[:, -1]  # [batch_size, hidden_size]
        
        # Generate forecast
        forecast = self.forecast_head(seq_representation)  # [batch_size, forecast_horizon * output_dim]
        
        # Reshape to [batch_size, forecast_horizon, output_dim]
        forecast = forecast.view(-1, self.forecast_horizon, self.output_dim)
        
        return forecast
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Training step.
        
        Args:
            batch: Tuple of (input_sequence, target_sequence)
            batch_idx: Index of the batch
            
        Returns:
            Loss tensor
        """
        input_sequence, target_sequence = batch
        forecast = self(input_sequence)
        
        # Handle target reshaping if needed
        if target_sequence.dim() == 2 and self.output_dim == 1:
            # Target is [batch_size, forecast_horizon]
            target_sequence = target_sequence.unsqueeze(-1)  # Add output_dim
        
        loss = F.mse_loss(forecast, target_sequence)
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Validation step.
        
        Args:
            batch: Tuple of (input_sequence, target_sequence)
            batch_idx: Index of the batch
            
        Returns:
            Dictionary of validation metrics
        """
        input_sequence, target_sequence = batch
        forecast = self(input_sequence)
        
        # Handle target reshaping if needed
        if target_sequence.dim() == 2 and self.output_dim == 1:
            # Target is [batch_size, forecast_horizon]
            target_sequence = target_sequence.unsqueeze(-1)  # Add output_dim
        
        # Calculate loss
        loss = F.mse_loss(forecast, target_sequence)
        
        # Calculate MAE
        mae = F.l1_loss(forecast, target_sequence)
        
        # Log metrics
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_mae', mae, on_epoch=True, prog_bar=True)
        
        # Store outputs for epoch end processing
        self.validation_step_outputs.append({
            'val_loss': loss,
            'val_mae': mae,
            'forecast': forecast.detach(),
            'target': target_sequence.detach()
        })
        
        return {'val_loss': loss, 'val_mae': mae}
    
    def on_validation_epoch_end(self) -> None:
        """Process validation outputs at the end of the epoch."""
        if not self.validation_step_outputs:
            return
        
        # Calculate average metrics
        avg_val_loss = torch.stack([x['val_loss'] for x in self.validation_step_outputs]).mean()
        avg_val_mae = torch.stack([x['val_mae'] for x in self.validation_step_outputs]).mean()
        
        # Log to progress bar
        self.log('val_loss_epoch', avg_val_loss, prog_bar=True)
        self.log('val_mae_epoch', avg_val_mae, prog_bar=True)
        
        # Clear outputs
        self.validation_step_outputs.clear()
    
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Test step.
        
        Args:
            batch: Tuple of (input_sequence, target_sequence)
            batch_idx: Index of the batch
            
        Returns:
            Dictionary of test metrics
        """
        input_sequence, target_sequence = batch
        forecast = self(input_sequence)
        
        # Handle target reshaping if needed
        if target_sequence.dim() == 2 and self.output_dim == 1:
            # Target is [batch_size, forecast_horizon]
            target_sequence = target_sequence.unsqueeze(-1)  # Add output_dim
        
        # Calculate metrics
        mse = F.mse_loss(forecast, target_sequence)
        mae = F.l1_loss(forecast, target_sequence)
        
        # Store outputs for epoch end processing
        self.test_step_outputs.append({
            'test_mse': mse,
            'test_mae': mae,
            'forecast': forecast.detach(),
            'target': target_sequence.detach(),
            'input': input_sequence.detach()
        })
        
        # Log metrics
        self.log('test_mse', mse, on_epoch=True)
        self.log('test_mae', mae, on_epoch=True)
        
        return {'test_mse': mse, 'test_mae': mae}
    
    def on_test_epoch_end(self) -> None:
        """Process test outputs at the end of the epoch."""
        if not self.test_step_outputs:
            return
        
        # Calculate average metrics
        avg_test_mse = torch.stack([x['test_mse'] for x in self.test_step_outputs]).mean()
        avg_test_mae = torch.stack([x['test_mae'] for x in self.test_step_outputs]).mean()
        
        # Calculate RMSE
        rmse = torch.sqrt(avg_test_mse)
        
        # Log final metrics
        self.log('test_rmse', rmse)
        
        # Print summary
        print(f"\nTest Results:")
        print(f"MSE: {avg_test_mse:.6f}")
        print(f"RMSE: {rmse:.6f}")
        print(f"MAE: {avg_test_mae:.6f}")
        
        # Select a random batch for visualization
        if self.trainer.is_global_zero and hasattr(self, 'plot_forecasts'):
            import random
            sample_idx = random.randint(0, len(self.test_step_outputs) - 1)
            sample = self.test_step_outputs[sample_idx]
            
            # Get input, forecast, and target
            input_seq = sample['input'][0].cpu().numpy()  # First sample in batch
            forecast = sample['forecast'][0].cpu().numpy()
            target = sample['target'][0].cpu().numpy()
            
            # Plot
            self.plot_forecasts(input_seq, forecast, target)
        
        # Clear outputs
        self.test_step_outputs.clear()
    
    def configure_optimizers(self) -> Union[Optimizer, Dict[str, Any]]:
        """
        Configure optimizer and scheduler.
        
        Returns:
            Optimizer or dictionary with optimizer and scheduler
        """
        # Define optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Define scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }
    
    def plot_forecasts(
        self,
        input_sequence: np.ndarray,
        forecast: np.ndarray,
        target: np.ndarray,
        title: str = "Time Series Forecast",
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot input sequence, forecast, and target for visualization.
        
        Args:
            input_sequence: Input sequence
            forecast: Forecast values
            target: Target values
            title: Plot title
            save_path: Path to save the plot
        """
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Define x-axis values
        x_input = np.arange(input_sequence.shape[0])
        x_forecast = np.arange(input_sequence.shape[0], input_sequence.shape[0] + forecast.shape[0])
        
        # Plot input sequence (last dimension if multi-dimensional)
        input_dim = input_sequence.shape[1] if input_sequence.ndim > 1 else 1
        if input_dim > 1:
            # Plot last dimension (assuming it's the target variable)
            plt.plot(x_input, input_sequence[:, -1], 'b-', label='Historical')
        else:
            plt.plot(x_input, input_sequence, 'b-', label='Historical')
        
        # Plot forecast and target
        if forecast.ndim > 1:
            # Multi-dimensional forecast
            plt.plot(x_forecast, forecast[:, 0], 'r-', label='Forecast')
            plt.plot(x_forecast, target[:, 0], 'g-', label='Actual')
        else:
            # One-dimensional forecast
            plt.plot(x_forecast, forecast, 'r-', label='Forecast')
            plt.plot(x_forecast, target, 'g-', label='Actual')
        
        # Add a vertical line to separate input and forecast
        plt.axvline(x=input_sequence.shape[0] - 0.5, color='k', linestyle='--')
        
        # Add labels and legend
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


def load_and_prepare_data(
    data_path: str,
    sequence_length: int = 96,
    forecast_horizon: int = 24,
    target_column: Optional[str] = None,
    feature_columns: Optional[List[str]] = None,
    date_column: Optional[str] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    val_split: float = 0.15,
    test_split: float = 0.15
) -> Tuple[DataLoader, DataLoader, DataLoader, TimeSeriesDataset]:
    """
    Load and prepare data for time series forecasting.
    
    Args:
        data_path: Path to data file (CSV)
        sequence_length: Length of input sequences
        forecast_horizon: Number of future steps to predict
        target_column: Column to predict
        feature_columns: Columns to use as features
        date_column: Column with dates
        batch_size: Batch size for dataloaders
        num_workers: Number of dataloader workers
        val_split: Validation split ratio
        test_split: Test split ratio
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader, dataset)
    """
    # Load dataset
    dataset = TimeSeriesDataset.from_csv(
        file_path=data_path,
        sequence_length=sequence_length,
        forecast_horizon=forecast_horizon,
        target_column=target_column,
        feature_columns=feature_columns,
        date_column=date_column
    )
    
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
    
    return train_loader, val_loader, test_loader, dataset


def train_and_evaluate(
    data_path: str,
    attention_type: str = "standard",
    sequence_length: int = 96,
    forecast_horizon: int = 24,
    target_column: Optional[str] = None,
    feature_columns: Optional[List[str]] = None,
    date_column: Optional[str] = None,
    hidden_size: int = 128,
    num_attention_heads: int = 4,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.01,
    dropout: float = 0.1,
    max_epochs: int = 50,
    patience: int = 10,
    save_dir: str = "./results/time_series",
    use_gpu: bool = True
) -> pl.LightningModule:
    """
    Train and evaluate time series forecasting model.
    
    Args:
        data_path: Path to data file (CSV)
        attention_type: Type of attention mechanism to use
        sequence_length: Length of input sequences
        forecast_horizon: Number of future steps to predict
        target_column: Column to predict
        feature_columns: Columns to use as features
        date_column: Column with dates
        hidden_size: Size of hidden layers
        num_attention_heads: Number of attention heads
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
    train_loader, val_loader, test_loader, dataset = load_and_prepare_data(
        data_path=data_path,
        sequence_length=sequence_length,
        forecast_horizon=forecast_horizon,
        target_column=target_column,
        feature_columns=feature_columns,
        date_column=date_column,
        batch_size=batch_size
    )
    
    # Get input and output dimensions
    input_dim = dataset.get_feature_dim()
    output_dim = dataset.get_target_dim()
    
    # Create model
    model = TimeSeriesModel(
        input_dim=input_dim,
        hidden_size=hidden_size,
        output_dim=output_dim,
        forecast_horizon=forecast_horizon,
        attention_type=attention_type,
        num_attention_heads=num_attention_heads,
        dropout=dropout,
        learning_rate=learning_rate,
        weight_decay=weight_decay
    )
    
    # Create result visualizer
    def plot_sample_forecast(model):
        """Plot a sample forecast from the test set."""
        # Get a sample from the test set
        for batch in test_loader:
            input_seq, target = batch
            # Generate forecast
            with torch.no_grad():
                forecast = model(input_seq)
            
            # Handle reshaping if needed
            if target.dim() == 2 and output_dim == 1:
                target = target.unsqueeze(-1)
            
            # Convert to numpy
            input_np = input_seq[0].cpu().numpy()
            forecast_np = forecast[0].cpu().numpy()
            target_np = target[0].cpu().numpy()
            
            # Inverse transform if dataset has a scaler
            if hasattr(dataset, 'inverse_transform'):
                forecast_np = dataset.inverse_transform(forecast_np, is_target=True)
                target_np = dataset.inverse_transform(target_np, is_target=True)
                last_values = input_np[-forecast_horizon:, -1] if input_np.ndim > 1 else input_np[-forecast_horizon:]
                input_np = dataset.inverse_transform(input_np)
            
            # Plot
            plt.figure(figsize=(12, 6))
            
            # Plot actual vs predicted
            x_input = np.arange(input_np.shape[0])
            x_forecast = np.arange(input_np.shape[0], input_np.shape[0] + forecast_np.shape[0])
            
            # Plot input sequence (last dimension if multi-dimensional)
            if input_np.ndim > 1:
                plt.plot(x_input, input_np[:, -1], 'b-', label='Historical')
            else:
                plt.plot(x_input, input_np, 'b-', label='Historical')
            
            # Plot forecast and target
            if forecast_np.ndim > 1:
                # Multi-dimensional forecast
                plt.plot(x_forecast, forecast_np[:, 0], 'r-', label='Forecast')
                plt.plot(x_forecast, target_np[:, 0], 'g-', label='Actual')
            else:
                # One-dimensional forecast
                plt.plot(x_forecast, forecast_np, 'r-', label='Forecast')
                plt.plot(x_forecast, target_np, 'g-', label='Actual')
            
            # Add vertical line to separate input and forecast
            plt.axvline(x=input_np.shape[0] - 0.5, color='k', linestyle='--')
            
            # Add title and labels
            plt.title(f"Time Series Forecast using {attention_type.capitalize()} Attention")
            plt.xlabel("Time Step")
            plt.ylabel("Value")
            plt.legend()
            plt.grid(True)
            
            # Save the figure
            plt.savefig(save_dir / f"{attention_type}_forecast.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            break
    
    # Attach the plot function to the model for use during testing
    model.plot_forecasts = plot_sample_forecast
    
    # Set up callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=save_dir / "checkpoints",
            filename=f"{attention_type}_model",
            monitor="val_loss",
            mode="min",
            save_top_k=1
        ),
        EarlyStopping(
            monitor="val_loss",
            mode="min",
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
    
    # Plot a sample forecast
    plot_sample_forecast(model)
    
    return model


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Time series forecasting with different attention mechanisms")
    
    # Data parameters
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to data file (CSV)")
    parser.add_argument("--target_column", type=str, default=None,
                        help="Column to predict")
    parser.add_argument("--feature_columns", type=str, nargs="+", default=None,
                        help="Columns to use as features")
    parser.add_argument("--date_column", type=str, default=None,
                        help="Column with dates")
    
    # Model parameters
    parser.add_argument("--attention_type", type=str, default="standard",
                        choices=["standard", "trittention", "sparse", "windowed"],
                        help="Type of attention mechanism to use")
    parser.add_argument("--sequence_length", type=int, default=96,
                        help="Length of input sequences")
    parser.add_argument("--forecast_horizon", type=int, default=24,
                        help="Number of future steps to predict")
    parser.add_argument("--hidden_size", type=int, default=128,
                        help="Size of hidden layers")
    parser.add_argument("--num_attention_heads", type=int, default=4,
                        help="Number of attention heads")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout probability")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="Learning rate for optimization")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay for regularization")
    parser.add_argument("--max_epochs", type=int, default=50,
                        help="Maximum number of training epochs")
    parser.add_argument("--patience", type=int, default=10,
                        help="Patience for early stopping")
    
    # Output parameters
    parser.add_argument("--save_dir", type=str, default="./results/time_series",
                        help="Directory to save results")
    parser.add_argument("--gpu", action="store_true",
                        help="Use GPU if available")
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    print(f"Training time series forecasting model with {args.attention_type} attention")
    print(f"Data path: {args.data_path}")
    print(f"Sequence length: {args.sequence_length}, Forecast horizon: {args.forecast_horizon}")
    print(f"Model parameters: hidden_size={args.hidden_size}, num_attention_heads={args.num_attention_heads}")
    
    # Train and evaluate model
    model = train_and_evaluate(
        data_path=args.data_path,
        attention_type=args.attention_type,
        sequence_length=args.sequence_length,
        forecast_horizon=args.forecast_horizon,
        target_column=args.target_column,
        feature_columns=args.feature_columns,
        date_column=args.date_column,
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_attention_heads,
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
