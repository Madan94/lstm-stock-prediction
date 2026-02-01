#!/usr/bin/env python
"""Script to train models for SP500 only."""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch
from typing import Dict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.data.fetcher import fetch_index_data
from backend.data.preprocessor import clean_data, create_target, prepare_sequences
from backend.features.indicators import engineer_features, get_feature_columns
from backend.models.attention_lstm import AttentionLSTM
from backend.models.baseline_models import VanillaLSTM, ARIMAModel
from backend.losses.asymmetric import AsymmetricDirectionalLoss
from backend.training.trainer import train_model
from backend.training.walk_forward import (
    create_walk_forward_windows,
    get_walk_forward_data,
    walk_forward_predict
)
from backend.backtesting.strategy import long_only_strategy
from backend.backtesting.metrics import calculate_all_metrics
from backend.utils.config import (
    INDICES, YEARS_OF_DATA, LOOKBACK_WINDOW,
    TRAIN_WINDOW_YEARS, TEST_WINDOW_MONTHS,
    LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS, DROPOUT,
    LEARNING_RATE, BATCH_SIZE, NUM_EPOCHS,
    UPWARD_PENALTY, DOWNWARD_PENALTY,
    MODELS_DIR, RESULTS_DIR
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# GPU memory optimization
if torch.cuda.is_available():
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'


def train_attention_lstm(
    index_name: str,
    df: pd.DataFrame,
    device: str = 'cpu'
) -> Dict:
    """Train attention-based LSTM model."""
    logger.info(f"Training Attention LSTM for {index_name}")
    
    # Feature engineering
    df_features = engineer_features(df)
    df_features = clean_data(df_features)
    df_features = create_target(df_features)
    
    # Remove rows with NaN (from indicators)
    df_features = df_features.dropna()
    
    feature_cols = get_feature_columns()
    
    # Walk-forward training
    windows = create_walk_forward_windows(df_features, TRAIN_WINDOW_YEARS, TEST_WINDOW_MONTHS)
    
    all_predictions = []
    all_actuals = []
    all_probabilities = []
    all_dates = []
    all_attention_weights = []
    
    for window_idx, (train_end, test_end) in enumerate(windows):
        logger.info(f"Window {window_idx + 1}/{len(windows)}: {train_end.date()} to {test_end.date()}")
        
        # Get data for this window
        data = get_walk_forward_data(df_features, train_end, test_end, LOOKBACK_WINDOW)
        if data[0] is None:
            continue
        
        X_train, y_train, X_test, y_test, train_dates, test_dates = data
        
        if len(X_train) == 0 or len(X_test) == 0:
            continue
        
        # Create model
        input_size = X_train.shape[2]
        model = AttentionLSTM(
            input_size=input_size,
            hidden_size=LSTM_HIDDEN_SIZE,
            num_layers=LSTM_NUM_LAYERS,
            dropout=DROPOUT
        )
        
        # Split training data for validation
        val_split = int(0.8 * len(X_train))
        X_train_split = X_train[:val_split]
        y_train_split = y_train[:val_split]
        X_val_split = X_train[val_split:]
        y_val_split = y_train[val_split:]
        
        # Loss function
        loss_fn = AsymmetricDirectionalLoss(UPWARD_PENALTY, DOWNWARD_PENALTY)
        
        # Train model
        model, history = train_model(
            model=model,
            X_train=X_train_split,
            y_train=y_train_split,
            X_val=X_val_split,
            y_val=y_val_split,
            loss_fn=loss_fn,
            num_epochs=NUM_EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            device=device,
            return_attention=True
        )
        
        # Make predictions
        predictions, attention_weights = walk_forward_predict(
            model, X_test, device=device, return_attention=True
        )
        
        # Convert to binary predictions
        pred_binary = (predictions.squeeze() > 0.5).astype(int)
        
        # Store results
        all_predictions.extend(pred_binary)
        all_actuals.extend(y_test)
        all_probabilities.extend(predictions.squeeze())
        all_dates.extend(test_dates)
        all_attention_weights.extend(attention_weights)
        
        # Clear memory after each window
        del model, X_train, y_train, X_test, y_test
        del X_train_split, y_train_split, X_val_split, y_val_split
        del predictions, attention_weights, pred_binary
        if device == 'cuda':
            torch.cuda.empty_cache()
    
    # Check if we have any predictions
    if len(all_predictions) == 0 or len(all_actuals) == 0:
        logger.warning(f"No predictions generated for {index_name}. Skipping metrics calculation.")
        return {
            'predictions': [],
            'actuals': [],
            'probabilities': [],
            'dates': [],
            'attention_weights': [],
            'metrics': {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0
            }
        }
    
    # Calculate metrics
    accuracy = accuracy_score(all_actuals, all_predictions)
    precision = precision_score(all_actuals, all_predictions, zero_division=0)
    recall = recall_score(all_actuals, all_predictions, zero_division=0)
    f1 = f1_score(all_actuals, all_predictions, zero_division=0)
    
    logger.info(f"Attention LSTM Metrics - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, "
                f"Recall: {recall:.4f}, F1: {f1:.4f}")
    
    return {
        'predictions': all_predictions,
        'actuals': all_actuals,
        'probabilities': all_probabilities,
        'dates': all_dates,
        'attention_weights': all_attention_weights,
        'metrics': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    }


def train_vanilla_lstm(
    index_name: str,
    df: pd.DataFrame,
    device: str = 'cpu'
) -> Dict:
    """Train vanilla LSTM model."""
    logger.info(f"Training Vanilla LSTM for {index_name}")
    
    df_features = engineer_features(df)
    df_features = clean_data(df_features)
    df_features = create_target(df_features)
    df_features = df_features.dropna()
    
    feature_cols = get_feature_columns()
    windows = create_walk_forward_windows(df_features, TRAIN_WINDOW_YEARS, TEST_WINDOW_MONTHS)
    
    all_predictions = []
    all_actuals = []
    all_probabilities = []
    all_dates = []
    
    for window_idx, (train_end, test_end) in enumerate(windows):
        data = get_walk_forward_data(df_features, train_end, test_end, LOOKBACK_WINDOW)
        if data[0] is None:
            continue
        
        X_train, y_train, X_test, y_test, train_dates, test_dates = data
        
        if len(X_train) == 0 or len(X_test) == 0:
            continue
        
        input_size = X_train.shape[2]
        model = VanillaLSTM(
            input_size=input_size,
            hidden_size=LSTM_HIDDEN_SIZE,
            num_layers=LSTM_NUM_LAYERS,
            dropout=DROPOUT
        )
        
        val_split = int(0.8 * len(X_train))
        loss_fn = AsymmetricDirectionalLoss(UPWARD_PENALTY, DOWNWARD_PENALTY)
        
        model, _ = train_model(
            model=model,
            X_train=X_train[:val_split],
            y_train=y_train[:val_split],
            X_val=X_train[val_split:],
            y_val=y_train[val_split:],
            loss_fn=loss_fn,
            num_epochs=NUM_EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            device=device,
            return_attention=False
        )
        
        predictions, _ = walk_forward_predict(model, X_test, device=device, return_attention=False)
        pred_binary = (predictions.squeeze() > 0.5).astype(int)
        
        all_predictions.extend(pred_binary)
        all_actuals.extend(y_test)
        all_probabilities.extend(predictions.squeeze())
        all_dates.extend(test_dates)
        
        # Clear memory after each window
        del model, X_train, y_train, X_test, y_test, predictions, pred_binary
        if device == 'cuda':
            torch.cuda.empty_cache()
    
    if len(all_predictions) == 0 or len(all_actuals) == 0:
        logger.warning(f"No predictions generated for vanilla LSTM on {index_name}")
        return {
            'predictions': [],
            'actuals': [],
            'probabilities': [],
            'dates': [],
            'metrics': {'accuracy': 0.0}
        }
    
    accuracy = accuracy_score(all_actuals, all_predictions)
    
    return {
        'predictions': all_predictions,
        'actuals': all_actuals,
        'probabilities': all_probabilities,
        'dates': all_dates,
        'metrics': {'accuracy': accuracy}
    }


def train_arima(index_name: str, df: pd.DataFrame) -> Dict:
    """Train ARIMA model."""
    logger.info(f"Training ARIMA for {index_name}")
    
    df_features = engineer_features(df)
    df_features = clean_data(df_features)
    df_features = create_target(df_features)
    df_features = df_features.dropna()
    
    windows = create_walk_forward_windows(df_features, TRAIN_WINDOW_YEARS, TEST_WINDOW_MONTHS)
    
    all_predictions = []
    all_actuals = []
    all_probabilities = []
    all_dates = []
    
    for window_idx, (train_end, test_end) in enumerate(windows):
        test_df = df_features[(df_features.index > train_end) & (df_features.index <= test_end)]
        
        if len(test_df) == 0:
            continue
        
        # Train ARIMA on prices up to train_end
        train_prices = df_features[df_features.index <= train_end]['close']
        
        if len(train_prices) < 50:  # ARIMA needs minimum data
            continue
        
        model = ARIMAModel()
        model.fit(train_prices)
        
        # Predict for test period
        for date in test_df.index:
            try:
                # Use historical prices up to this point
                hist_prices = df_features[df_features.index <= date]['close']
                if len(hist_prices) < 50:
                    continue
                
                prob = model.predict_proba(hist_prices, n_periods=1)[0]
                pred = 1 if prob > 0.5 else 0
                actual = test_df.loc[date, 'target']
                
                all_predictions.append(pred)
                all_actuals.append(actual)
                all_probabilities.append(prob)
                all_dates.append(date)
            except:
                continue
    
    if len(all_actuals) > 0:
        accuracy = accuracy_score(all_actuals, all_predictions)
    else:
        accuracy = 0.0
    
    return {
        'predictions': all_predictions,
        'actuals': all_actuals,
        'metrics': {'accuracy': accuracy}
    }


def backtest_strategy(
    index_name: str,
    df: pd.DataFrame,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    dates: pd.DatetimeIndex
) -> Dict:
    """Backtest strategy and calculate performance metrics."""
    logger.info(f"Backtesting strategy for {index_name}")
    
    # Get prices aligned with predictions
    prices = df.loc[dates, 'close']
    
    # Run strategy
    equity_curve, returns, trades = long_only_strategy(
        predictions, probabilities, prices, dates
    )
    
    # Calculate metrics
    metrics = calculate_all_metrics(equity_curve, returns, trades)
    
    return {
        'equity_curve': {
            'dates': equity_curve.index.tolist(),
            'values': equity_curve.values.tolist()
        },
        'returns': returns.values.tolist(),
        'trades': trades.to_dict('records') if len(trades) > 0 else [],
        'metrics': metrics
    }


def main():
    """Train models for SP500 only."""
    # Create directories
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        logger.info(f"CUDA Version: {torch.version.cuda}")
    else:
        logger.info("Using CPU (GPU not available)")
    
    # Fetch SP500 data
    logger.info("Fetching data for SP500...")
    index_name = "SP500"
    symbol = INDICES[index_name]["symbol"]
    df = fetch_index_data(symbol, years=YEARS_OF_DATA)
    
    if df is None or len(df) == 0:
        logger.error(f"No data available for {index_name}")
        return
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing {index_name}")
    logger.info(f"{'='*60}")
    
    try:
        # Train Attention LSTM
        attention_results = train_attention_lstm(index_name, df, device)
        
        # Check if we have predictions before backtesting
        if len(attention_results['predictions']) == 0:
            logger.warning(f"No predictions for {index_name}, skipping backtest")
            return
        
        # Backtest Attention LSTM
        backtest_results = backtest_strategy(
            index_name,
            df,
            np.array(attention_results['predictions']),
            np.array(attention_results['probabilities']),
            pd.DatetimeIndex(attention_results['dates'])
        )
        
        # Train baseline models
        vanilla_results = train_vanilla_lstm(index_name, df, device)
        arima_results = train_arima(index_name, df)
        
        # Calculate baseline backtests
        vanilla_backtest = {'metrics': {}}
        if len(vanilla_results.get('predictions', [])) > 0:
            try:
                vanilla_backtest = backtest_strategy(
                    index_name, df,
                    np.array(vanilla_results['predictions']),
                    np.array(vanilla_results.get('probabilities', vanilla_results['predictions'])),
                    pd.DatetimeIndex(vanilla_results.get('dates', []))
                )
            except Exception as e:
                logger.warning(f"Error backtesting vanilla LSTM: {e}")
        
        arima_backtest = {'metrics': {}}
        if len(arima_results.get('predictions', [])) > 0:
            try:
                arima_backtest = backtest_strategy(
                    index_name, df,
                    np.array(arima_results['predictions']),
                    np.array(arima_results.get('probabilities', arima_results['predictions'])),
                    pd.DatetimeIndex(arima_results.get('dates', []))
                )
            except Exception as e:
                logger.warning(f"Error backtesting ARIMA: {e}")
        
        # Compile results
        results = {
            'index': index_name,
            'metrics': attention_results['metrics'],
            'predictions': {
                'dates': [d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d) 
                         for d in attention_results['dates']],
                'actuals': attention_results['actuals'],
                'predictions': attention_results['predictions'],
                'probabilities': attention_results['probabilities']
            },
            'equity_curve': backtest_results['equity_curve'],
            'attention_weights': {
                'dates': [d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d)
                         for d in attention_results['dates']],
                'weights': attention_results['attention_weights']
            },
            'baseline_comparison': {
                'attention_lstm': {
                    'accuracy': attention_results['metrics']['accuracy'],
                    'sharpe_ratio': backtest_results['metrics'].get('sharpe_ratio', 0.0),
                    'total_return': backtest_results['metrics'].get('total_return', 0.0),
                    'max_drawdown': backtest_results['metrics'].get('max_drawdown', 0.0)
                },
                'vanilla_lstm': {
                    'accuracy': vanilla_results['metrics']['accuracy'],
                    'sharpe_ratio': vanilla_backtest['metrics'].get('sharpe_ratio', 0.0),
                    'total_return': vanilla_backtest['metrics'].get('total_return', 0.0),
                    'max_drawdown': vanilla_backtest['metrics'].get('max_drawdown', 0.0)
                },
                'arima': {
                    'accuracy': arima_results['metrics']['accuracy'],
                    'sharpe_ratio': arima_backtest['metrics'].get('sharpe_ratio', 0.0),
                    'total_return': arima_backtest['metrics'].get('total_return', 0.0),
                    'max_drawdown': arima_backtest['metrics'].get('max_drawdown', 0.0)
                }
            }
        }
        
        # Save results
        results_file = os.path.join(RESULTS_DIR, f"{index_name}_results.pkl")
        with open(results_file, 'wb') as f:
            pickle.dump(results, f)
        
        logger.info(f"Saved results to {results_file}")
        
    except Exception as e:
        logger.error(f"Error processing {index_name}: {e}", exc_info=True)
    
    logger.info("\nTraining complete!")


if __name__ == "__main__":
    main()
