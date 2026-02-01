"""Model training utilities."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


def train_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    loss_fn: nn.Module,
    num_epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    device: str = 'cpu',
    return_attention: bool = False
) -> Tuple[nn.Module, Dict]:
    """
    Train a model with early stopping.
    
    Args:
        model: PyTorch model
        X_train: Training sequences (samples, seq_len, features)
        y_train: Training targets (samples,)
        X_val: Validation sequences
        y_val: Validation targets
        loss_fn: Loss function
        num_epochs: Maximum number of epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: Device to train on
        return_attention: Whether model returns attention weights
        
    Returns:
        Trained model and training history
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.LongTensor(y_train).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.LongTensor(y_val).to(device)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': []
    }
    
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            
            if return_attention:
                output, _ = model(X_batch, return_attention=True)
            else:
                output = model(X_batch)
            
            loss = loss_fn(output, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        with torch.no_grad():
            if return_attention:
                val_output, _ = model(X_val_t, return_attention=True)
            else:
                val_output = model(X_val_t)
            
            val_loss = loss_fn(val_output, y_val_t).item()
            
            # Calculate accuracy
            predictions = (val_output.squeeze() > 0.5).long()
            val_accuracy = (predictions == y_val_t).float().mean().item()
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        if (epoch + 1) % 10 == 0:
            logger.info(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Val Acc: {val_accuracy:.4f}"
            )
    
    return model, history





