"""Attention-based LSTM model for directional prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism for capturing different aspects of sequences."""
    
    def __init__(self, hidden_size: int, num_heads: int = 4):
        super(MultiHeadAttention, self).__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply multi-head attention.
        
        Args:
            x: Input tensor (batch, seq_len, hidden_size)
            
        Returns:
            output: Attention output (batch, seq_len, hidden_size)
            attention_weights: Average attention weights (batch, seq_len)
        """
        batch_size, seq_len, _ = x.shape
        
        # Linear projections
        Q = self.query(x)  # (batch, seq_len, hidden_size)
        K = self.key(x)
        V = self.value(x)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # Now shape is (batch, num_heads, seq_len, head_dim)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = F.softmax(scores, dim=-1)  # (batch, num_heads, seq_len, seq_len)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, V)  # (batch, num_heads, seq_len, head_dim)
        
        # Concatenate heads
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        
        # Final linear projection
        output = self.fc_out(attended)
        
        # Average attention weights across heads for visualization
        avg_attention = attention_weights.mean(dim=1).mean(dim=1)  # (batch, seq_len)
        
        return output, avg_attention


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
    """Enhanced Attention-based LSTM for financial time series prediction."""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_layers: int = 3,
        dropout: float = 0.3,
        use_multihead: bool = True,
        num_heads: int = 4
    ):
        super(AttentionLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_multihead = use_multihead
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Layer normalization after LSTM
        self.lstm_norm = nn.LayerNorm(hidden_size)
        
        # Multi-head or single-head attention
        if use_multihead:
            self.attention = MultiHeadAttention(hidden_size, num_heads)
            self.attention_norm = nn.LayerNorm(hidden_size)
        else:
            self.attention = AttentionLayer(hidden_size)
        
        # Context attention for final aggregation
        self.context_attention = AttentionLayer(hidden_size)
        
        # Output layer for binary classification
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.LayerNorm(hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
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
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size)
        lstm_out = self.lstm_norm(lstm_out)
        
        # Apply attention mechanism
        if self.use_multihead:
            # Multi-head self-attention
            attended_out, attention_weights = self.attention(lstm_out)
            # Residual connection + layer norm
            lstm_out = self.attention_norm(lstm_out + attended_out)
        
        # Final context aggregation
        context, final_attention_weights = self.context_attention(lstm_out)
        
        # Final prediction
        output = self.fc(context)  # (batch, 1)
        
        if return_attention:
            return output, final_attention_weights
        return output, None






