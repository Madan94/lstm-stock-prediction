"""Model training utilities."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


def add_noise_augmentation(X: torch.Tensor, noise_factor: float = 0.01) -> torch.Tensor:
    """Add small Gaussian noise for data augmentation."""
    noise = torch.randn_like(X) * noise_factor
    return X + noise


def train_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    loss_fn: nn.Module,
    num_epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    device: str = 'cpu',
    return_attention: bool = False,
    use_augmentation: bool = True,
    weight_decay: float = 1e-5
) -> Tuple[nn.Module, Dict]:
    """
    Train a model with early stopping, learning rate scheduling, and data augmentation.
    
    Args:
        model: PyTorch model
        X_train: Training sequences (samples, seq_len, features)
        y_train: Training targets (samples,)
        X_val: Validation sequences
        y_val: Validation targets
        loss_fn: Loss function
        num_epochs: Maximum number of epochs
        batch_size: Batch size
        learning_rate: Initial learning rate
        device: Device to train on
        return_attention: Whether model returns attention weights
        use_augmentation: Whether to use data augmentation
        weight_decay: L2 regularization strength
        
    Returns:
        Trained model and training history
    """
    model = model.to(device)
    
    # Use AdamW with weight decay for better regularization
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler with cosine annealing
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=learning_rate * 0.01
    )
    
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.LongTensor(y_train).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.LongTensor(y_val).to(device)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        pin_memory=True if device == 'cuda' else False
    )
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'learning_rate': []
    }
    
    best_val_loss = float('inf')
    best_model_state = None
    patience = 20  # Increased patience
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            
            # Data augmentation
            if use_augmentation and torch.rand(1).item() > 0.5:
                X_batch = add_noise_augmentation(X_batch, noise_factor=0.005)
            
            if return_attention:
                output, _ = model(X_batch, return_attention=True)
            else:
                output = model(X_batch)
            
            loss = loss_fn(output, y_batch)
            
            # Gradient clipping for stability
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
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
        history['learning_rate'].append(current_lr)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
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
                f"Val Acc: {val_accuracy:.4f}, "
                f"LR: {current_lr:.6f}"
            )
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info("Loaded best model based on validation loss")
    
    return model, history







