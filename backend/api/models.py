"""Pydantic models for API requests and responses."""

from pydantic import BaseModel
from typing import List, Optional, Dict
from datetime import datetime


class IndexInfo(BaseModel):
    """Index information."""
    name: str
    symbol: str
    display_name: str


class MetricsResponse(BaseModel):
    """Model performance metrics."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float


class PredictionItem(BaseModel):
    """Single prediction result."""
    date: str
    actual_direction: int
    predicted_direction: int
    probability: float
    correct: bool


class PredictionsResponse(BaseModel):
    """Predictions response."""
    index: str
    predictions: List[PredictionItem]


class EquityCurvePoint(BaseModel):
    """Equity curve data point."""
    date: str
    value: float


class EquityCurveResponse(BaseModel):
    """Equity curve response."""
    index: str
    equity_curve: List[EquityCurvePoint]


class AttentionWeights(BaseModel):
    """Attention weights for a single prediction."""
    date: str
    weights: List[float]  # Weights for each day in lookback window


class AttentionResponse(BaseModel):
    """Attention weights response."""
    index: str
    attention: List[AttentionWeights]
    lookback_days: int


class BaselineComparison(BaseModel):
    """Baseline model performance."""
    model_name: str
    accuracy: float
    sharpe_ratio: float
    total_return: float
    max_drawdown: float


class BaselineComparisonResponse(BaseModel):
    """Baseline comparison response."""
    index: str
    models: List[BaselineComparison]





