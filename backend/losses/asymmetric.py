"""Asymmetric loss function for directional prediction."""

import torch
import torch.nn as nn
from typing import Optional


class AsymmetricDirectionalLoss(nn.Module):
    """
    Asymmetric loss that penalizes missing upward moves more than false positives.
    
    Loss = α * BCE(pred, target) if target == 1 (upward)
         = β * BCE(pred, target) if target == 0 (downward)
    
    Where typically α > β to emphasize catching upward movements.
    """
    
    def __init__(self, upward_penalty: float = 2.0, downward_penalty: float = 1.0):
        """
        Initialize asymmetric loss.
        
        Args:
            upward_penalty: Penalty multiplier for missing upward moves (α)
            downward_penalty: Penalty multiplier for false positives (β)
        """
        super(AsymmetricDirectionalLoss, self).__init__()
        self.upward_penalty = upward_penalty
        self.downward_penalty = downward_penalty
        self.bce = nn.BCELoss(reduction='none')
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute asymmetric loss.
        
        Args:
            predictions: Model predictions (batch, 1) in [0, 1]
            targets: True labels (batch,) in {0, 1}
            
        Returns:
            Scalar loss value
        """
        # Ensure predictions and targets have compatible shapes
        # Flatten predictions to match targets shape
        if predictions.dim() > 1:
            predictions = predictions.view(-1)  # Flatten to (batch,)
        elif predictions.dim() == 0:
            predictions = predictions.unsqueeze(0)  # Add batch dimension if scalar
        
        # Ensure targets are float and 1D
        targets = targets.float()
        if targets.dim() == 0:
            targets = targets.unsqueeze(0)
        
        # Ensure same shape
        if predictions.shape != targets.shape:
            # Take minimum length to avoid shape mismatch
            min_len = min(len(predictions), len(targets))
            predictions = predictions[:min_len]
            targets = targets[:min_len]
        
        # Compute base BCE loss for each sample
        bce_loss = self.bce(predictions, targets)
        
        # Apply asymmetric weights
        weights = torch.where(
            targets == 1,
            torch.tensor(self.upward_penalty, device=predictions.device),
            torch.tensor(self.downward_penalty, device=predictions.device)
        )
        
        # Weighted loss
        weighted_loss = weights * bce_loss
        
        return weighted_loss.mean()



