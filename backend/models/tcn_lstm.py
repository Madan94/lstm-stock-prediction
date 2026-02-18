"""Temporal Convolutional Network (TCN) combined with LSTM for time series prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class TemporalBlock(nn.Module):
    """Temporal Convolutional Block with dilation."""
    
    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float = 0.2
    ):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(
            n_inputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(
            n_outputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )
        
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights."""
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class Chomp1d(nn.Module):
    """Remove padding from the end."""
    
    def __init__(self, chomp_size: int):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :-self.chomp_size].contiguous()


class TCN(nn.Module):
    """Temporal Convolutional Network."""
    
    def __init__(
        self,
        num_inputs: int,
        num_channels: list,
        kernel_size: int = 2,
        dropout: float = 0.2
    ):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [
                TemporalBlock(
                    in_channels, out_channels, kernel_size,
                    stride=1, dilation=dilation_size,
                    padding=(kernel_size-1) * dilation_size,
                    dropout=dropout
                )
            ]
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(x)


class TCNLSTM(nn.Module):
    """Hybrid TCN-LSTM model for time series prediction."""
    
    def __init__(
        self,
        input_size: int,
        tcn_channels: list = [64, 128, 256],
        lstm_hidden_size: int = 256,
        lstm_num_layers: int = 2,
        dropout: float = 0.3,
        kernel_size: int = 3
    ):
        super(TCNLSTM, self).__init__()
        
        # TCN layers for feature extraction
        self.tcn = TCN(
            num_inputs=input_size,
            num_channels=tcn_channels,
            kernel_size=kernel_size,
            dropout=dropout
        )
        
        # LSTM layers for sequence modeling
        tcn_output_size = tcn_channels[-1]
        self.lstm = nn.LSTM(
            input_size=tcn_output_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            dropout=dropout if lstm_num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(lstm_hidden_size * 2, lstm_hidden_size),
            nn.Tanh(),
            nn.Linear(lstm_hidden_size, 1)
        )
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(lstm_hidden_size * 2, lstm_hidden_size),
            nn.LayerNorm(lstm_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden_size, lstm_hidden_size // 2),
            nn.LayerNorm(lstm_hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(lstm_hidden_size // 2, 1),
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
            attention_weights: Attention weights if requested
        """
        batch_size, seq_len, _ = x.size()
        
        # TCN expects (batch, features, seq_len)
        x_tcn = x.transpose(1, 2)  # (batch, features, seq_len)
        tcn_out = self.tcn(x_tcn)  # (batch, tcn_channels[-1], seq_len)
        
        # Transpose back for LSTM: (batch, seq_len, features)
        tcn_out = tcn_out.transpose(1, 2)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(tcn_out)  # (batch, seq_len, hidden_size * 2)
        
        # Attention mechanism
        attention_scores = self.attention(lstm_out)  # (batch, seq_len, 1)
        attention_weights = F.softmax(attention_scores, dim=1)
        context = (attention_weights * lstm_out).sum(dim=1)  # (batch, hidden_size * 2)
        
        # Final prediction
        output = self.output_layers(context)  # (batch, 1)
        
        if return_attention:
            attention_weights = attention_weights.squeeze(-1)  # (batch, seq_len)
            return output, attention_weights
        return output, None




