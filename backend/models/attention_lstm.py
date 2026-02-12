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


class ResidualBlock(nn.Module):
    """Residual block for better gradient flow."""
    
    def __init__(self, hidden_size: int, dropout: float = 0.2):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.dropout(out)
        return self.activation(out + residual)


class AttentionLSTM(nn.Module):
    """Enhanced Attention-based LSTM for financial time series prediction."""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_layers: int = 3,
        dropout: float = 0.3,
        use_residual: bool = True
    ):
        super(AttentionLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_residual = use_residual
        
        # Input normalization
        self.input_norm = nn.LayerNorm(input_size)
        
        # Bidirectional LSTM layers for better context
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,  # Will be doubled by bidirectional
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        # Project bidirectional output to desired hidden size
        self.lstm_proj = nn.Linear(hidden_size * 2, hidden_size)
        
        # Attention mechanism
        self.attention = AttentionLayer(hidden_size)
        
        # Residual blocks for deep feature extraction
        if use_residual:
            self.residual_blocks = nn.Sequential(
                ResidualBlock(hidden_size, dropout),
                ResidualBlock(hidden_size, dropout)
            )
        
        # Enhanced output layer with multiple layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.LayerNorm(hidden_size // 4),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_size // 4, 1),
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
        # Normalize input
        x = self.input_norm(x)
        
        # LSTM forward pass (bidirectional)
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size * 2)
        
        # Project to desired hidden size
        lstm_out = self.lstm_proj(lstm_out)  # (batch, seq_len, hidden_size)
        
        # Apply attention
        context, attention_weights = self.attention(lstm_out)
        
        # Apply residual blocks if enabled
        if self.use_residual:
            context = self.residual_blocks(context)
        
        # Final prediction
        output = self.fc(context)  # (batch, 1)
        
        if return_attention:
            return output, attention_weights
        return output, None







