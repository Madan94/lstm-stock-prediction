"""Advanced Transformer-LSTM hybrid for maximum prediction accuracy."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding."""
        return x + self.pe[:x.size(0)]


class TransformerEncoderBlock(nn.Module):
    """Single transformer encoder block with multi-head attention."""
    
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super(TransformerEncoderBlock, self).__init__()
        
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # Feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through transformer block."""
        # Multi-head self-attention with residual
        attn_output, _ = self.self_attn(x, x, x)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        
        # Feed-forward with residual
        ff_output = self.linear2(self.dropout(F.relu(self.linear1(x))))
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)
        
        return x


class TemporalFusionBlock(nn.Module):
    """Fuses information from multiple temporal scales."""
    
    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.3):
        super(TemporalFusionBlock, self).__init__()
        
        # Multi-scale convolutions
        self.conv_short = nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1)
        self.conv_medium = nn.Conv1d(input_size, hidden_size, kernel_size=5, padding=2)
        self.conv_long = nn.Conv1d(input_size, hidden_size, kernel_size=7, padding=3)
        
        # Fusion layer
        self.fusion = nn.Linear(hidden_size * 3, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, features)
        Returns:
            fused: (batch, seq_len, hidden_size)
        """
        # Transpose for conv1d: (batch, features, seq_len)
        x_t = x.transpose(1, 2)
        
        # Multi-scale feature extraction
        short = F.relu(self.conv_short(x_t))
        medium = F.relu(self.conv_medium(x_t))
        long = F.relu(self.conv_long(x_t))
        
        # Transpose back and concatenate
        short = short.transpose(1, 2)
        medium = medium.transpose(1, 2)
        long = long.transpose(1, 2)
        
        combined = torch.cat([short, medium, long], dim=-1)
        
        # Fuse features
        fused = self.fusion(combined)
        fused = self.norm(fused)
        fused = self.dropout(fused)
        
        return fused


class AdvancedTransformerLSTM(nn.Module):
    """
    State-of-the-art Transformer-LSTM hybrid for stock prediction.
    
    Combines:
    - Temporal fusion for multi-scale patterns
    - Transformer encoders for global dependencies
    - Bidirectional LSTM for sequential modeling
    - Dynamic attention for adaptive feature selection
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 512,
        num_transformer_layers: int = 4,
        num_lstm_layers: int = 3,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.3
    ):
        super(AdvancedTransformerLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        
        # Temporal fusion for multi-scale feature extraction
        self.temporal_fusion = TemporalFusionBlock(input_size, hidden_size, dropout)
        
        # Transformer encoder layers
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderBlock(hidden_size, num_heads, dim_feedforward, dropout)
            for _ in range(num_transformer_layers)
        ])
        
        # Bidirectional LSTM for sequential dependencies
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            dropout=dropout if num_lstm_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        self.lstm_norm = nn.LayerNorm(hidden_size * 2)
        
        # Dynamic attention mechanism
        self.attention_query = nn.Linear(hidden_size * 2, hidden_size)
        self.attention_key = nn.Linear(hidden_size * 2, hidden_size)
        self.attention_value = nn.Linear(hidden_size * 2, hidden_size)
        
        # Volatility regime detector
        self.regime_detector = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 3),  # Low, Medium, High volatility
            nn.Softmax(dim=-1)
        )
        
        # Regime-adaptive prediction heads
        self.pred_heads = nn.ModuleList([
            nn.Sequential(
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
            for _ in range(3)  # One for each volatility regime
        ])
        
        # Confidence estimator
        self.confidence = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
        return_confidence: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: Input sequences (batch, seq_len, features)
            return_attention: Whether to return attention weights
            return_confidence: Whether to return prediction confidence
            
        Returns:
            output: Predictions (batch, 1)
            extras: Dict with attention, confidence, regime if requested
        """
        batch_size, seq_len, _ = x.shape
        
        # Multi-scale temporal fusion
        x = self.temporal_fusion(x)  # (batch, seq_len, hidden_size)
        
        # Transformer encoder layers
        for transformer in self.transformer_layers:
            x = transformer(x)
        
        # Bidirectional LSTM
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size * 2)
        lstm_out = self.lstm_norm(lstm_out)
        
        # Dynamic attention
        Q = self.attention_query(lstm_out)
        K = self.attention_key(lstm_out)
        V = self.attention_value(lstm_out)
        
        # Scaled dot-product attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.hidden_size)
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Apply attention
        context = torch.matmul(attention_weights, V)  # (batch, seq_len, hidden_size)
        
        # Use last timestep for prediction
        final_hidden = context[:, -1, :]  # (batch, hidden_size)
        
        # Detect volatility regime
        regime_probs = self.regime_detector(lstm_out[:, -1, :])  # (batch, 3)
        
        # Regime-adaptive predictions
        regime_predictions = []
        for head in self.pred_heads:
            regime_predictions.append(head(final_hidden))
        
        regime_predictions = torch.stack(regime_predictions, dim=1)  # (batch, 3, 1)
        
        # Weighted combination based on regime
        output = torch.sum(
            regime_predictions.squeeze(-1) * regime_probs,
            dim=1,
            keepdim=True
        )  # (batch, 1)
        
        # Prediction confidence
        pred_confidence = self.confidence(final_hidden)
        
        extras = {}
        if return_attention:
            # Return average attention over sequence
            extras['attention'] = attention_weights.mean(dim=1).mean(dim=1)  # (batch,)
        
        if return_confidence:
            extras['confidence'] = pred_confidence
            extras['regime'] = regime_probs
        
        if extras:
            return output, extras
        return output, None


class EnsembleTransformerLSTM(nn.Module):
    """
    Ensemble of Transformer-LSTM models with different configurations
    for maximum robustness and accuracy.
    """
    
    def __init__(
        self,
        input_size: int,
        num_models: int = 5,
        hidden_size: int = 512,
        dropout: float = 0.3
    ):
        super(EnsembleTransformerLSTM, self).__init__()
        
        # Create diverse models with different architectures
        configs = [
            {'num_transformer_layers': 4, 'num_lstm_layers': 3, 'num_heads': 8},
            {'num_transformer_layers': 3, 'num_lstm_layers': 4, 'num_heads': 4},
            {'num_transformer_layers': 6, 'num_lstm_layers': 2, 'num_heads': 16},
            {'num_transformer_layers': 2, 'num_lstm_layers': 4, 'num_heads': 4},
            {'num_transformer_layers': 4, 'num_lstm_layers': 2, 'num_heads': 8},
        ]
        
        self.models = nn.ModuleList([
            AdvancedTransformerLSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                dropout=dropout,
                **config
            )
            for config in configs[:num_models]
        ])
        
        # Learned ensemble weights
        self.ensemble_weights = nn.Parameter(torch.ones(num_models) / num_models)
    
    def forward(
        self,
        x: torch.Tensor,
        return_confidence: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through ensemble.
        
        Returns weighted average of all models.
        """
        predictions = []
        confidences = []
        
        for model in self.models:
            pred, extras = model(x, return_confidence=return_confidence)
            predictions.append(pred)
            
            if return_confidence and extras:
                confidences.append(extras['confidence'])
        
        # Stack predictions
        predictions = torch.stack(predictions, dim=1)  # (batch, num_models, 1)
        
        # Normalize ensemble weights
        weights = F.softmax(self.ensemble_weights, dim=0)
        
        # Weighted average
        output = torch.sum(predictions.squeeze(-1) * weights, dim=1, keepdim=True)
        
        if return_confidence and confidences:
            avg_confidence = torch.stack(confidences, dim=1).mean(dim=1)
            return output, avg_confidence
        
        return output, None
