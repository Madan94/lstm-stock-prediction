"""Attention-based LSTM model for directional prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class AttentionLayer(nn.Module):
    """Bahdanau-style additive attention mechanism."""
    
    def __init__(self, hidden_size: int):
        super(AttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.attention_weights = nn.Linear(hidden_size, hidden_size)
        self.context_vector = nn.Linear(hidden_size, 1, bias=False)
        
    def forward(self, lstm_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply attention mechanism.
        
        Args:
            lstm_output: LSTM output of shape (batch, seq_len, hidden_size)
            
        Returns:
            context: Weighted context vector (batch, hidden_size)
            attention_weights: Attention weights (batch, seq_len)
        """
        # Compute attention scores
        attention_scores = self.attention_weights(lstm_output)  # (batch, seq_len, hidden_size)
        attention_scores = torch.tanh(attention_scores)
        attention_scores = self.context_vector(attention_scores)  # (batch, seq_len, 1)
        attention_scores = attention_scores.squeeze(-1)  # (batch, seq_len)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch, seq_len)
        
        # Compute weighted context
        context = torch.bmm(
            attention_weights.unsqueeze(1),  # (batch, 1, seq_len)
            lstm_output  # (batch, seq_len, hidden_size)
        ).squeeze(1)  # (batch, hidden_size)
        
        return context, attention_weights


class AttentionLSTM(nn.Module):
    """Attention-based LSTM for financial time series prediction."""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super(AttentionLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Attention mechanism
        self.attention = AttentionLayer(hidden_size)
        
        # Output layer for binary classification
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: Input sequences (batch, seq_len, features)
            return_attention: Whether to return attention weights
            
        Returns:
            output: Prediction probabilities (batch, 1)
            attention_weights: Attention weights if return_attention=True (batch, seq_len)
        """
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size)
        
        # Apply attention
        context, attention_weights = self.attention(lstm_out)
        
        # Final prediction
        output = self.fc(context)  # (batch, 1)
        
        if return_attention:
            return output, attention_weights
        return output, None





