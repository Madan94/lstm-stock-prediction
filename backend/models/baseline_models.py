"""Baseline models for comparison."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Optional, Tuple
from pmdarima import auto_arima
import pickle
import os


class VanillaLSTM(nn.Module):
    """Vanilla LSTM without attention mechanism."""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super(VanillaLSTM, self).__init__()
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
        
        # Output layer
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input sequences (batch, seq_len, features)
            
        Returns:
            output: Prediction probabilities (batch, 1)
        """
        # LSTM forward pass
        lstm_out, (h_n, _) = self.lstm(x)
        
        # Use last hidden state
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden_size)
        
        # Final prediction
        output = self.fc(last_hidden)  # (batch, 1)
        
        return output


class ARIMAModel:
    """ARIMA model wrapper for directional prediction."""
    
    def __init__(self):
        self.model = None
        self.is_fitted = False
        
    def fit(self, prices: pd.Series):
        """
        Fit ARIMA model using auto-ARIMA.
        
        Args:
            prices: Time series of prices
        """
        try:
            # Use auto-ARIMA to find best parameters
            self.model = auto_arima(
                prices,
                seasonal=False,
                stepwise=True,
                suppress_warnings=True,
                error_action='ignore',
                max_p=5,
                max_d=2,
                max_q=5
            )
            self.is_fitted = True
        except Exception as e:
            print(f"Error fitting ARIMA: {e}")
            self.is_fitted = False
    
    def predict_direction(self, prices: pd.Series, n_periods: int = 1) -> np.ndarray:
        """
        Predict directional movement.
        
        Args:
            prices: Historical prices
            n_periods: Number of periods to predict ahead
            
        Returns:
            Array of predicted directions (1 = up, 0 = down)
        """
        if not self.is_fitted:
            self.fit(prices)
        
        try:
            # Get forecast
            forecast = self.model.predict(n_periods=n_periods)
            
            # Compare with last known price
            last_price = prices.iloc[-1]
            directions = (forecast > last_price).astype(int)
            
            return directions
        except Exception as e:
            print(f"Error predicting with ARIMA: {e}")
            # Return neutral prediction
            return np.array([0.5] * n_periods)
    
    def predict_proba(self, prices: pd.Series, n_periods: int = 1) -> np.ndarray:
        """
        Predict probability of upward movement.
        
        Args:
            prices: Historical prices
            n_periods: Number of periods to predict ahead
            
        Returns:
            Array of probabilities [0, 1]
        """
        directions = self.predict_direction(prices, n_periods)
        # Convert to probabilities (simple approach)
        return directions.astype(float)
    
    def save(self, filepath: str):
        """Save model to file."""
        if self.is_fitted:
            with open(filepath, 'wb') as f:
                pickle.dump(self.model, f)
    
    def load(self, filepath: str):
        """Load model from file."""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                self.model = pickle.load(f)
            self.is_fitted = True









