#!/usr/bin/env python
"""Comprehensive training script for all enhanced LSTM models and ensemble."""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.data.fetcher import fetch_index_data
from backend.data.preprocessor import clean_data, create_target
from backend.features.indicators import engineer_features, get_feature_columns
from backend.models.attention_lstm import AttentionLSTM
from backend.models.enhanced_lstm import (
    BidirectionalAttentionLSTM,
    StackedResidualLSTM,
    GRUModel,
    CNNLSTM
)
from backend.models.transformer_lstm import AdvancedTransformerLSTM
from backend.models.ensemble import WeightedEnsemble, StackingEnsemble
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


def train_single_model(
    model_class,
    model_name: str,
    input_size: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    device: str = 'cpu',
    **model_kwargs
) -> Tuple[torch.nn.Module, Dict]:
    """Train a single model and return it with training history."""
    logger.info(f"Training {model_name}...")
    
    # Create model
    model = model_class(
        input_size=input_size,
        hidden_size=model_kwargs.get('hidden_size', LSTM_HIDDEN_SIZE),
        num_layers=model_kwargs.get('num_layers', LSTM_NUM_LAYERS),
        dropout=model_kwargs.get('dropout', DROPOUT)
    )
    
    # Loss function
    loss_fn = AsymmetricDirectionalLoss(UPWARD_PENALTY, DOWNWARD_PENALTY)
    
    # Train model
    model, history = train_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        loss_fn=loss_fn,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        device=device,
        return_attention=False
    )
    
    return model, history


def train_all_models(
    index_name: str,
    df: pd.DataFrame,
    device: str = 'cpu'
) -> Dict:
    """Train all enhanced LSTM models and create ensemble."""
    logger.info(f"Training all models for {index_name}")
    
    # Feature engineering
    df_features = engineer_features(df)
    df_features = clean_data(df_features)
    df_features = create_target(df_features)
    df_features = df_features.dropna()
    
    feature_cols = get_feature_columns()
    
    # Walk-forward training
    windows = create_walk_forward_windows(df_features, TRAIN_WINDOW_YEARS, TEST_WINDOW_MONTHS)
    
    # Define all models to train
    model_configs = [
        (AttentionLSTM, 'AttentionLSTM', {'use_multihead': True, 'num_heads': 4}),
        (BidirectionalAttentionLSTM, 'BidirectionalLSTM', {}),
        (StackedResidualLSTM, 'StackedResidualLSTM', {'num_layers': 4}),
        (GRUModel, 'GRU', {}),
        (CNNLSTM, 'CNN-LSTM', {}),
        (AdvancedTransformerLSTM, 'Transformer-LSTM', {})
    ]
    
    # Store results for each model
    model_results = {name: {
        'predictions': [],
        'actuals': [],
        'probabilities': [],
        'dates': [],
        'validation_scores': []
    } for _, name, _ in model_configs}
    
    # Train on each window
    for window_idx, (train_end, test_end) in enumerate(windows):
        logger.info(f"\nWindow {window_idx + 1}/{len(windows)}: {train_end.date()} to {test_end.date()}")
        
        # Get data for this window
        data = get_walk_forward_data(df_features, train_end, test_end, LOOKBACK_WINDOW)
        if data[0] is None:
            continue
        
        X_train, y_train, X_test, y_test, train_dates, test_dates = data
        
        if len(X_train) == 0 or len(X_test) == 0:
            continue
        
        # Split training data for validation
        val_split = int(0.8 * len(X_train))
        X_train_split = X_train[:val_split]
        y_train_split = y_train[:val_split]
        X_val_split = X_train[val_split:]
        y_val_split = y_train[val_split:]
        
        input_size = X_train.shape[2]
        trained_models = []
        validation_scores = []
        
        # Train each model
        for model_class, model_name, extra_kwargs in model_configs:
            try:
                # Prepare model kwargs
                model_kwargs = {
                    'hidden_size': LSTM_HIDDEN_SIZE,
                    'num_layers': LSTM_NUM_LAYERS,
                    'dropout': DROPOUT
                }
                model_kwargs.update(extra_kwargs)
                
                # Train model
                model, history = train_single_model(
                    model_class, model_name, input_size,
                    X_train_split, y_train_split,
                    X_val_split, y_val_split,
                    device, **model_kwargs
                )
                
                # Get validation accuracy
                val_accuracy = max(history['val_accuracy'])
                validation_scores.append(val_accuracy)
                model_results[model_name]['validation_scores'].append(val_accuracy)
                
                # Make predictions on test set
                predictions, _ = walk_forward_predict(
                    model, X_test, device=device, return_attention=False
                )
                pred_binary = (predictions.squeeze() > 0.5).astype(int)
                
                # Store results
                model_results[model_name]['predictions'].extend(pred_binary)
                model_results[model_name]['probabilities'].extend(predictions.squeeze())
                
                # Store actuals and dates only once (same for all models)
                if len(model_results[model_name]['actuals']) < len(y_test) * (window_idx + 1):
                    model_results[model_name]['actuals'].extend(y_test)
                    model_results[model_name]['dates'].extend(test_dates)
                
                trained_models.append(model)
                
                logger.info(f"  {model_name}: Val Acc={val_accuracy:.4f}, "
                          f"Test Acc={accuracy_score(y_test, pred_binary):.4f}")
                
                # Clear model from GPU memory
                del model, predictions, pred_binary
                if device == 'cuda':
                    torch.cuda.empty_cache()
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}", exc_info=True)
                continue
        
        # Create ensemble predictions for this window
        if len(trained_models) > 1:
            try:
                # Weighted ensemble
                ensemble = WeightedEnsemble(trained_models)
                ensemble.update_weights(validation_scores)
                
                ensemble_pred, individual_preds = ensemble.predict(
                    X_test, device=device, return_individual=True
                )
                ensemble_binary = (ensemble_pred.squeeze() > 0.5).astype(int)
                
                # Store ensemble results
                if 'Ensemble' not in model_results:
                    model_results['Ensemble'] = {
                        'predictions': [],
                        'actuals': [],
                        'probabilities': [],
                        'dates': [],
                        'validation_scores': []
                    }
                
                model_results['Ensemble']['predictions'].extend(ensemble_binary)
                model_results['Ensemble']['probabilities'].extend(ensemble_pred.squeeze())
                model_results['Ensemble']['actuals'].extend(y_test)
                model_results['Ensemble']['dates'].extend(test_dates)
                
                ensemble_acc = accuracy_score(y_test, ensemble_binary)
                model_results['Ensemble']['validation_scores'].append(ensemble_acc)
                
                logger.info(f"  Ensemble: Test Acc={ensemble_acc:.4f}")
                
                del ensemble, ensemble_pred, individual_preds
                if device == 'cuda':
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                logger.error(f"Error creating ensemble: {e}", exc_info=True)
        
        # Clear memory after each window
        del X_train, y_train, X_test, y_test
        del X_train_split, y_train_split, X_val_split, y_val_split
        if device == 'cuda':
            torch.cuda.empty_cache()
    
    # Calculate final metrics for each model
    final_results = {}
    
    for model_name, results in model_results.items():
        if len(results['predictions']) == 0:
            logger.warning(f"No predictions for {model_name}")
            continue
        
        predictions = np.array(results['predictions'])
        actuals = np.array(results['actuals'])
        probabilities = np.array(results['probabilities'])
        
        # Calculate metrics
        accuracy = accuracy_score(actuals, predictions)
        precision = precision_score(actuals, predictions, zero_division=0)
        recall = recall_score(actuals, predictions, zero_division=0)
        f1 = f1_score(actuals, predictions, zero_division=0)
        
        avg_val_acc = np.mean(results['validation_scores']) if results['validation_scores'] else 0.0
        
        logger.info(f"\n{model_name} Final Metrics:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  F1 Score: {f1:.4f}")
        logger.info(f"  Avg Val Accuracy: {avg_val_acc:.4f}")
        
        # Backtest
        try:
            backtest_results = backtest_strategy(
                index_name, df,
                predictions,
                probabilities,
                pd.DatetimeIndex(results['dates'])
            )
            
            logger.info(f"  Sharpe Ratio: {backtest_results['metrics'].get('sharpe_ratio', 0.0):.4f}")
            logger.info(f"  Total Return: {backtest_results['metrics'].get('total_return', 0.0):.2%}")
            logger.info(f"  Max Drawdown: {backtest_results['metrics'].get('max_drawdown', 0.0):.2%}")
            
        except Exception as e:
            logger.warning(f"Error backtesting {model_name}: {e}")
            backtest_results = {'metrics': {}}
        
        final_results[model_name] = {
            'predictions': predictions.tolist(),
            'actuals': actuals.tolist(),
            'probabilities': probabilities.tolist(),
            'dates': [d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d) 
                     for d in results['dates']],
            'metrics': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'avg_validation_accuracy': avg_val_acc
            },
            'backtest': backtest_results
        }
    
    return final_results


def backtest_strategy(
    index_name: str,
    df: pd.DataFrame,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    dates: pd.DatetimeIndex
) -> Dict:
    """Backtest strategy and calculate performance metrics."""
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
    """Train all enhanced models and create ensemble."""
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
    logger.info(f"Training Enhanced Models for {index_name}")
    logger.info(f"{'='*60}")
    
    try:
        # Train all models
        all_results = train_all_models(index_name, df, device)
        
        # Save results
        results_file = os.path.join(RESULTS_DIR, f"{index_name}_enhanced_results.pkl")
        with open(results_file, 'wb') as f:
            pickle.dump(all_results, f)
        
        logger.info(f"\nSaved results to {results_file}")
        
        # Print summary comparison
        logger.info(f"\n{'='*60}")
        logger.info("FINAL MODEL COMPARISON")
        logger.info(f"{'='*60}")
        logger.info(f"{'Model':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Sharpe':>10}")
        logger.info("-" * 80)
        
        for model_name, results in sorted(all_results.items()):
            metrics = results['metrics']
            backtest = results['backtest']['metrics']
            logger.info(f"{model_name:<20} "
                       f"{metrics['accuracy']:>10.4f} "
                       f"{metrics['precision']:>10.4f} "
                       f"{metrics['recall']:>10.4f} "
                       f"{metrics['f1_score']:>10.4f} "
                       f"{backtest.get('sharpe_ratio', 0.0):>10.4f}")
        
    except Exception as e:
        logger.error(f"Error processing {index_name}: {e}", exc_info=True)
    
    logger.info("\nTraining complete!")


if __name__ == "__main__":
    main()
