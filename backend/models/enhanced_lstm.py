"""Enhanced LSTM architectures for improved stock prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class BidirectionalAttentionLSTM(nn.Module):
    """Bidirectional LSTM with advanced attention mechanism for capturing both past and future context."""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 512,  # Increased from 256
        num_layers: int = 4,  # Increased from 3
        dropout: float = 0.4  # Increased dropout
    ):
        super(BidirectionalAttentionLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Input projection for better feature learning
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5)
        )
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True  # Key enhancement
        )
        
        # Layer normalization after LSTM
        self.lstm_norm = nn.LayerNorm(hidden_size * 2)
        
        # Multi-head attention over bidirectional outputs
        self.num_heads = 8
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,
            num_heads=self.num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Context aggregation
        self.context_attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )
        
        # Output layers with deeper architecture
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.LayerNorm(hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass with bidirectional processing."""
        # Project input features
        x = self.input_proj(x)
        
        # Bidirectional LSTM
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size * 2)
        lstm_out = self.lstm_norm(lstm_out)
        
        # Multi-head self-attention
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Residual connection
        lstm_out = lstm_out + attn_out
        
        # Context aggregation attention
        attention_scores = self.context_attention(lstm_out)
        attention_weights = F.softmax(attention_scores.squeeze(-1), dim=1)
        
        # Weighted context
        context = torch.bmm(
            attention_weights.unsqueeze(1),
            lstm_out
        ).squeeze(1)
        
        # Prediction
        output = self.fc(context)
        
        if return_attention:
            return output, attention_weights
        return output, None


class StackedResidualLSTM(nn.Module):
    """Deep LSTM with residual connections to prevent vanishing gradients."""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_layers: int = 4,
        dropout: float = 0.3
    ):
        super(StackedResidualLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Input projection
        self.input_proj = nn.Linear(input_size, hidden_size)
        
        # Stacked LSTM layers with residual connections
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        # Layer normalization for each LSTM layer
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_size) for _ in range(num_layers)
        ])
        
        # Dropout layers
        self.dropouts = nn.ModuleList([
            nn.Dropout(dropout) for _ in range(num_layers)
        ])
        
        # Attention mechanism
        self.attention_weights = nn.Linear(hidden_size, hidden_size)
        self.context_vector = nn.Linear(hidden_size, 1, bias=False)
        
        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
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
        """Forward pass with residual connections."""
        # Project input to hidden size
        x = self.input_proj(x)  # (batch, seq_len, hidden_size)
        
        # Process through stacked LSTM with residuals
        for i, (lstm, norm, dropout) in enumerate(zip(self.lstm_layers, self.layer_norms, self.dropouts)):
            residual = x
            lstm_out, _ = lstm(x)
            
            # Residual connection + layer norm
            if i > 0:  # Skip first layer residual
                x = norm(lstm_out + residual)
            else:
                x = norm(lstm_out)
            
            x = dropout(x)
        
        # Attention mechanism
        attention_scores = self.attention_weights(x)
        attention_scores = torch.tanh(attention_scores)
        attention_scores = self.context_vector(attention_scores).squeeze(-1)
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # Context vector
        context = torch.bmm(
            attention_weights.unsqueeze(1),
            x
        ).squeeze(1)
        
        # Prediction
        output = self.fc(context)
        
        if return_attention:
            return output, attention_weights
        return output, None


class GRUModel(nn.Module):
    """GRU-based model as an alternative to LSTM with fewer parameters."""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_layers: int = 3,
        dropout: float = 0.3
    ):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention_weights = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.context_vector = nn.Linear(hidden_size * 2, 1, bias=False)
        
        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
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
        """Forward pass through GRU."""
        # GRU forward pass
        gru_out, _ = self.gru(x)  # (batch, seq_len, hidden_size * 2)
        
        # Attention mechanism
        attention_scores = self.attention_weights(gru_out)
        attention_scores = torch.tanh(attention_scores)
        attention_scores = self.context_vector(attention_scores).squeeze(-1)
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # Context vector
        context = torch.bmm(
            attention_weights.unsqueeze(1),
            gru_out
        ).squeeze(1)
        
        # Prediction
        output = self.fc(context)
        
        if return_attention:
            return output, attention_weights
        return output, None


class CNNLSTM(nn.Module):
    """Hybrid CNN-LSTM model for pattern extraction and temporal modeling."""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        super(CNNLSTM, self).__init__()
        self.hidden_size = hidden_size
        
        # CNN layers for local pattern extraction
        self.conv1 = nn.Conv1d(
            in_channels=input_size,
            out_channels=64,
            kernel_size=3,
            padding=1
        )
        self.bn1 = nn.BatchNorm1d(64)
        
        self.conv2 = nn.Conv1d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            padding=1
        )
        self.bn2 = nn.BatchNorm1d(128)
        
        self.conv3 = nn.Conv1d(
            in_channels=128,
            out_channels=hidden_size,
            kernel_size=3,
            padding=1
        )
        self.bn3 = nn.BatchNorm1d(hidden_size)
        
        self.conv_dropout = nn.Dropout(dropout)
        
        # LSTM for temporal dependencies
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention_weights = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.context_vector = nn.Linear(hidden_size * 2, 1, bias=False)
        
        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
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
        """Forward pass through CNN then LSTM."""
        # CNN expects (batch, channels, seq_len)
        x = x.transpose(1, 2)  # (batch, features, seq_len)
        
        # Conv layer 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv_dropout(x)
        
        # Conv layer 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv_dropout(x)
        
        # Conv layer 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.conv_dropout(x)
        
        # Back to (batch, seq_len, features) for LSTM
        x = x.transpose(1, 2)
        
        # LSTM processing
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size * 2)
        
        # Attention mechanism
        attention_scores = self.attention_weights(lstm_out)
        attention_scores = torch.tanh(attention_scores)
        attention_scores = self.context_vector(attention_scores).squeeze(-1)
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # Context vector
        context = torch.bmm(
            attention_weights.unsqueeze(1),
            lstm_out
        ).squeeze(1)
        
        # Prediction
        output = self.fc(context)
        
        if return_attention:
            return output, attention_weights
        return output, None
