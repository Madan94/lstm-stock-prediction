"""Ensemble model combining multiple LSTM architectures."""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional


class WeightedEnsemble:
    """Ensemble multiple models using weighted averaging based on validation performance."""
    
    def __init__(self, models: List[nn.Module], weights: Optional[List[float]] = None):
        """
        Initialize ensemble.
        
        Args:
            models: List of trained PyTorch models
            weights: Optional weights for each model (default: equal weights)
        """
        self.models = models
        
        if weights is None:
            # Equal weights
            self.weights = [1.0 / len(models)] * len(models)
        else:
            # Normalize weights
            total = sum(weights)
            self.weights = [w / total for w in weights]
    
    def predict(
        self,
        X: np.ndarray,
        device: str = 'cpu',
        return_individual: bool = False
    ) -> Tuple[np.ndarray, Optional[Dict[str, np.ndarray]]]:
        """
        Make ensemble predictions.
        
        Args:
            X: Input data (numpy array)
            device: Device to run models on
            return_individual: Whether to return individual model predictions
            
        Returns:
            ensemble_predictions: Weighted average predictions
            individual_predictions: Dict of individual model predictions (if requested)
        """
        all_predictions = []
        individual_preds = {}
        
        # Get predictions from each model
        for i, model in enumerate(self.models):
            model.eval()
            model = model.to(device)
            
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X).to(device)
                pred, _ = model(X_tensor, return_attention=False)
                pred_np = pred.cpu().numpy()
                
                all_predictions.append(pred_np * self.weights[i])
                
                if return_individual:
                    individual_preds[f'model_{i}'] = pred_np
        
        # Weighted average
        ensemble_pred = np.sum(all_predictions, axis=0)
        
        if return_individual:
            return ensemble_pred, individual_preds
        return ensemble_pred, None
    
    def update_weights(self, validation_scores: List[float]):
        """
        Update ensemble weights based on validation performance.
        
        Args:
            validation_scores: Accuracy scores for each model on validation set
        """
        # Softmax weighting (better models get higher weights)
        exp_scores = np.exp(np.array(validation_scores))
        self.weights = (exp_scores / exp_scores.sum()).tolist()


class StackingEnsemble(nn.Module):
    """Stacking ensemble with a meta-learner."""
    
    def __init__(
        self,
        base_models: List[nn.Module],
        meta_hidden_size: int = 64,
        dropout: float = 0.3
    ):
        """
        Initialize stacking ensemble.
        
        Args:
            base_models: List of base models
            meta_hidden_size: Hidden size for meta-learner
            dropout: Dropout rate for meta-learner
        """
        super(StackingEnsemble, self).__init__()
        self.base_models = nn.ModuleList(base_models)
        num_base_models = len(base_models)
        
        # Meta-learner (combines base model predictions)
        self.meta_learner = nn.Sequential(
            nn.Linear(num_base_models, meta_hidden_size),
            nn.LayerNorm(meta_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(meta_hidden_size, meta_hidden_size // 2),
            nn.LayerNorm(meta_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(meta_hidden_size // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        x: torch.Tensor,
        return_base_predictions: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through ensemble.
        
        Args:
            x: Input sequences
            return_base_predictions: Whether to return base model predictions
            
        Returns:
            output: Final ensemble prediction
            base_predictions: Base model predictions (if requested)
        """
        base_predictions = []
        
        # Get predictions from all base models
        for model in self.base_models:
            model.eval()
            with torch.no_grad():
                pred, _ = model(x, return_attention=False)
                base_predictions.append(pred)
        
        # Stack base predictions
        stacked = torch.cat(base_predictions, dim=1)  # (batch, num_models)
        
        # Meta-learner combines base predictions
        output = self.meta_learner(stacked)
        
        if return_base_predictions:
            return output, stacked
        return output, None
    
    def train_meta_learner(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        num_epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        device: str = 'cpu'
    ) -> Dict:
        """
        Train the meta-learner on base model predictions.
        
        Args:
            X_train: Training data
            y_train: Training labels
            X_val: Validation data
            y_val: Validation labels
            num_epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            device: Device to train on
            
        Returns:
            history: Training history
        """
        self.to(device)
        optimizer = torch.optim.Adam(self.meta_learner.parameters(), lr=learning_rate)
        criterion = nn.BCELoss()
        
        history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
        
        for epoch in range(num_epochs):
            # Training
            self.train()
            train_losses = []
            
            for i in range(0, len(X_train), batch_size):
                batch_X = torch.FloatTensor(X_train[i:i+batch_size]).to(device)
                batch_y = torch.FloatTensor(y_train[i:i+batch_size]).unsqueeze(1).to(device)
                
                optimizer.zero_grad()
                output, _ = self(batch_X)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                
                train_losses.append(loss.item())
            
            # Validation
            self.eval()
            with torch.no_grad():
                X_val_tensor = torch.FloatTensor(X_val).to(device)
                y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1).to(device)
                
                val_output, _ = self(X_val_tensor)
                val_loss = criterion(val_output, y_val_tensor)
                
                val_pred = (val_output.cpu().numpy() > 0.5).astype(int)
                val_accuracy = (val_pred == y_val.reshape(-1, 1)).mean()
            
            history['train_loss'].append(np.mean(train_losses))
            history['val_loss'].append(val_loss.item())
            history['val_accuracy'].append(val_accuracy)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs} - "
                      f"Train Loss: {history['train_loss'][-1]:.4f}, "
                      f"Val Loss: {history['val_loss'][-1]:.4f}, "
                      f"Val Acc: {history['val_accuracy'][-1]:.4f}")
        
        return history
