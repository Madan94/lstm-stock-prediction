"""Ensemble model for combining multiple predictions."""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class EnsembleModel(nn.Module):
    """
    Ensemble model that combines predictions from multiple models.
    Uses weighted voting or stacking for final prediction.
    """
    
    def __init__(
        self,
        models: List[nn.Module],
        weights: Optional[List[float]] = None,
        use_stacking: bool = True
    ):
        """
        Initialize ensemble model.
        
        Args:
            models: List of trained models
            weights: Optional weights for each model (default: equal weights)
            use_stacking: If True, use learned stacking layer; else use weighted average
        """
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)
        self.use_stacking = use_stacking
        
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        self.weights = weights
        
        # Stacking layer to learn optimal combination
        if use_stacking:
            self.stacking_layer = nn.Sequential(
                nn.Linear(len(models), len(models) * 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(len(models) * 2, 1),
                nn.Sigmoid()
            )
        else:
            self.stacking_layer = None
    
    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through ensemble.
        
        Args:
            x: Input sequences (batch, seq_len, features)
            return_attention: Whether to return attention weights
            
        Returns:
            output: Ensemble prediction probabilities (batch, 1)
            attention_weights: Average attention weights if requested
        """
        predictions = []
        attention_weights_list = []
        
        # Get predictions from all models
        for model in self.models:
            model.eval()
            with torch.no_grad():
                if return_attention:
                    pred, attn = model(x, return_attention=True)
                    attention_weights_list.append(attn)
                else:
                    pred = model(x)
                    if isinstance(pred, tuple):
                        pred = pred[0]
                
                predictions.append(pred)
        
        # Stack predictions
        stacked = torch.cat(predictions, dim=1)  # (batch, num_models)
        
        if self.use_stacking and self.stacking_layer is not None:
            # Learned combination
            output = self.stacking_layer(stacked)
        else:
            # Weighted average
            weights_tensor = torch.tensor(
                self.weights, 
                device=stacked.device, 
                dtype=stacked.dtype
            ).view(1, -1)
            output = (stacked * weights_tensor).sum(dim=1, keepdim=True)
        
        # Average attention weights if requested
        attention_weights = None
        if return_attention and len(attention_weights_list) > 0:
            attention_weights = torch.stack(attention_weights_list).mean(dim=0)
        
        return output, attention_weights


def create_ensemble_from_checkpoints(
    model_class,
    checkpoint_paths: List[str],
    model_configs: List[dict],
    device: str = 'cpu'
) -> EnsembleModel:
    """
    Create ensemble from saved model checkpoints.
    
    Args:
        model_class: Model class to instantiate
        checkpoint_paths: List of paths to saved models
        model_configs: List of config dicts for each model
        device: Device to load models on
        
    Returns:
        EnsembleModel instance
    """
    models = []
    for checkpoint_path, config in zip(checkpoint_paths, model_configs):
        model = model_class(**config)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
        models.append(model)
        logger.info(f"Loaded model from {checkpoint_path}")
    
    return EnsembleModel(models, use_stacking=True)



