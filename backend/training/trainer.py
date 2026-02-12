"""Model training utilities."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Tuple, Dict, Optional
import logging
import copy

logger = logging.getLogger(__name__)


def train_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    loss_fn: nn.Module,
    num_epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.0005,
    device: str = 'cpu',
    return_attention: bool = False,
    use_scheduler: bool = True,
    scheduler_patience: int = 5,
    early_stopping_patience: int = 15,
    gradient_clip: float = 1.0
) -> Tuple[nn.Module, Dict]:
    """
    Train a model with early stopping, learning rate scheduling, and gradient clipping.
    
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
        use_scheduler: Whether to use learning rate scheduler
        scheduler_patience: Patience for learning rate scheduler
        early_stopping_patience: Patience for early stopping
        gradient_clip: Gradient clipping threshold
        
    Returns:
        Trained model and training history
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    # Learning rate scheduler
    if use_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=scheduler_patience,
            min_lr=1e-6,
            verbose=True
        )
    
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.LongTensor(y_train).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.LongTensor(y_val).to(device)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'learning_rates': []
    }
    
    best_val_loss = float('inf')
    best_model_state = None
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
                output, _ = model(X_batch, return_attention=False)
            
            loss = loss_fn(output, y_batch)
            loss.backward()
            
            # Gradient clipping
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        with torch.no_grad():
            if return_attention:
                val_output, _ = model(X_val_t, return_attention=True)
            else:
                val_output, _ = model(X_val_t, return_attention=False)
            
            val_loss = loss_fn(val_output, y_val_t).item()
            
            # Calculate accuracy
            predictions = (val_output.squeeze() > 0.5).long()
            val_accuracy = (predictions == y_val_t).float().mean().item()
        
        # Update learning rate scheduler
        current_lr = optimizer.param_groups[0]['lr']
        if use_scheduler:
            scheduler.step(val_loss)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)
        history['learning_rates'].append(current_lr)
        
        # Early stopping with best model saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
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
        logger.info(f"Loaded best model with val_loss: {best_val_loss:.4f}")
    
    return model, history






