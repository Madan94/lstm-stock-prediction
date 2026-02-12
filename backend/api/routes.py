"""FastAPI route handlers."""

import os
import pickle
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from typing import Dict, List
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from backend.api.models import (
    IndexInfo, MetricsResponse, PredictionsResponse, PredictionItem,
    EquityCurveResponse, EquityCurvePoint,
    AttentionResponse, AttentionWeights,
    BaselineComparisonResponse, BaselineComparison
)
from backend.utils.config import INDICES, MODELS_DIR, RESULTS_DIR, LOOKBACK_WINDOW

router = APIRouter()


def load_results(index: str) -> Dict:
    """Load saved results for an index."""
    # Try advanced results first, then fall back to regular results
    advanced_results_file = os.path.join(RESULTS_DIR, f"{index}_advanced_results.pkl")
    results_file = os.path.join(RESULTS_DIR, f"{index}_results.pkl")
    
    if os.path.exists(advanced_results_file):
        results_file = advanced_results_file
    elif not os.path.exists(results_file):
        raise HTTPException(status_code=404, detail=f"No results found for {index}")
    
    with open(results_file, 'rb') as f:
        results = pickle.load(f)
    
    # If advanced results, extract ensemble data for API compatibility
    if 'ensemble' in results:
        # Use ensemble results as primary metrics
        ensemble_metrics = results.get('ensemble', {}).get('metrics', {})
        if ensemble_metrics:
            results['metrics'] = ensemble_metrics
        # Use ensemble predictions
        ensemble_preds = results.get('ensemble', {})
        if 'predictions' in ensemble_preds:
            results['predictions'] = {
                'dates': ensemble_preds.get('dates', []),
                'actuals': ensemble_preds.get('actuals', []),
                'predictions': ensemble_preds.get('predictions', []),
                'probabilities': ensemble_preds.get('probabilities', [])
            }
        # Use ensemble attention weights if available
        if 'attention_weights' in results.get('transformer', {}):
            results['attention_weights'] = {
                'dates': results.get('transformer', {}).get('dates', []),
                'weights': results.get('transformer', {}).get('attention_weights', [])
            }
    
    return results


@router.get("/indices", response_model=List[IndexInfo])
def get_indices():
    """List all available indices."""
    return [
        IndexInfo(
            name=name,
            symbol=config["symbol"],
            display_name=config["name"]
        )
        for name, config in INDICES.items()
    ]


@router.get("/metrics/{index}", response_model=MetricsResponse)
def get_metrics(index: str):
    """Get model performance metrics for an index."""
    if index not in INDICES:
        raise HTTPException(status_code=404, detail=f"Index {index} not found")
    
    results = load_results(index)
    
    if 'metrics' not in results:
        raise HTTPException(status_code=404, detail=f"No metrics found for {index}")
    
    metrics = results['metrics']
    
    return MetricsResponse(
        accuracy=metrics.get('accuracy', 0.0),
        precision=metrics.get('precision', 0.0),
        recall=metrics.get('recall', 0.0),
        f1_score=metrics.get('f1_score', 0.0)
    )


@router.get("/predictions/{index}", response_model=PredictionsResponse)
def get_predictions(index: str, limit: int = 100):
    """Get recent predictions for an index."""
    if index not in INDICES:
        raise HTTPException(status_code=404, detail=f"Index {index} not found")
    
    results = load_results(index)
    
    if 'predictions' not in results:
        raise HTTPException(status_code=404, detail=f"No predictions found for {index}")
    
    pred_data = results['predictions']
    
    # Convert to response format
    predictions = []
    for i, (date, actual, pred, prob) in enumerate(zip(
        pred_data['dates'],
        pred_data['actuals'],
        pred_data['predictions'],
        pred_data['probabilities']
    )):
        predictions.append(PredictionItem(
            date=date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date),
            actual_direction=int(actual),
            predicted_direction=int(pred),
            probability=float(prob),
            correct=bool(actual == pred)
        ))
    
    # Return most recent predictions
    predictions = predictions[-limit:]
    
    return PredictionsResponse(
        index=index,
        predictions=predictions
    )


@router.get("/equity-curve/{index}", response_model=EquityCurveResponse)
def get_equity_curve(index: str):
    """Get equity curve for backtested strategy."""
    if index not in INDICES:
        raise HTTPException(status_code=404, detail=f"Index {index} not found")
    
    results = load_results(index)
    
    if 'equity_curve' not in results:
        raise HTTPException(status_code=404, detail=f"No equity curve found for {index}")
    
    equity_data = results['equity_curve']
    
    equity_curve = [
        EquityCurvePoint(
            date=date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date),
            value=float(value)
        )
        for date, value in zip(equity_data['dates'], equity_data['values'])
    ]
    
    return EquityCurveResponse(
        index=index,
        equity_curve=equity_curve
    )


@router.get("/attention/{index}", response_model=AttentionResponse)
def get_attention(index: str, limit: int = 10):
    """Get attention weights for recent predictions."""
    if index not in INDICES:
        raise HTTPException(status_code=404, detail=f"Index {index} not found")
    
    results = load_results(index)
    
    if 'attention_weights' not in results:
        raise HTTPException(status_code=404, detail=f"No attention weights found for {index}")
    
    attention_data = results['attention_weights']
    
    # Get most recent attention weights
    dates = attention_data['dates'][-limit:]
    weights_list = attention_data['weights'][-limit:]
    
    attention = [
        AttentionWeights(
            date=date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date),
            weights=[float(w) for w in weights]
        )
        for date, weights in zip(dates, weights_list)
    ]
    
    return AttentionResponse(
        index=index,
        attention=attention,
        lookback_days=LOOKBACK_WINDOW
    )


@router.get("/baseline-comparison/{index}", response_model=BaselineComparisonResponse)
def get_baseline_comparison(index: str):
    """Get performance comparison with baseline models."""
    if index not in INDICES:
        raise HTTPException(status_code=404, detail=f"Index {index} not found")
    
    results = load_results(index)
    
    models = []
    
    # Check for advanced results format (ensemble models)
    if 'ensemble' in results:
        # Add ensemble model (best performing)
        ensemble_metrics = results.get('ensemble', {}).get('metrics', {})
        backtest_metrics = results.get('backtest_metrics', {})
        
        models.append(BaselineComparison(
            model_name='Ensemble (Transformer + TCN-LSTM + Attention-LSTM)',
            accuracy=ensemble_metrics.get('accuracy', 0.0),
            sharpe_ratio=backtest_metrics.get('sharpe_ratio', 0.0),
            total_return=backtest_metrics.get('total_return', 0.0),
            max_drawdown=backtest_metrics.get('max_drawdown', 0.0)
        ))
        
        # Add individual models
        for model_name in ['transformer', 'tcn_lstm', 'attention_lstm']:
            if model_name in results:
                model_results = results[model_name]
                model_metrics = model_results.get('metrics', {})
                
                # Try to get backtest metrics for individual models if available
                # Otherwise use default values
                models.append(BaselineComparison(
                    model_name=model_name.replace('_', ' ').title(),
                    accuracy=model_metrics.get('accuracy', 0.0),
                    sharpe_ratio=0.0,  # Individual models may not have backtest metrics
                    total_return=0.0,
                    max_drawdown=0.0
                ))
    
    # Fall back to baseline_comparison format if available
    elif 'baseline_comparison' in results:
        baseline_data = results['baseline_comparison']
        models = [
            BaselineComparison(
                model_name=model_name,
                accuracy=metrics.get('accuracy', 0.0),
                sharpe_ratio=metrics.get('sharpe_ratio', 0.0),
                total_return=metrics.get('total_return', 0.0),
                max_drawdown=metrics.get('max_drawdown', 0.0)
            )
            for model_name, metrics in baseline_data.items()
        ]
    else:
        raise HTTPException(status_code=404, detail=f"No baseline comparison found for {index}")
    
    return BaselineComparisonResponse(
        index=index,
        models=models
    )







