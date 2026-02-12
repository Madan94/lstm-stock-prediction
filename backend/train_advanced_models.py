"""Advanced training script with Transformer, TCN-LSTM, and Ensemble models."""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch
from typing import Dict, List
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.data.fetcher import get_all_indices_data
from backend.data.preprocessor import clean_data, create_target, prepare_sequences
from backend.features.indicators import engineer_features, get_feature_columns
from backend.models.attention_lstm import AttentionLSTM
from backend.models.transformer_model import TransformerModel
from backend.models.tcn_lstm import TCNLSTM
from backend.models.ensemble import EnsembleModel
from backend.losses.asymmetric import AsymmetricDirectionalLoss
from backend.training.trainer import train_model
from backend.training.walk_forward import (
    create_walk_forward_windows,
    get_walk_forward_data,
    get_simple_split_data,
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
    MODELS_DIR, RESULTS_DIR, WEIGHT_DECAY,
    USE_SIMPLE_SPLIT, TRAIN_SPLIT_RATIO
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_transformer_model(
    index_name: str,
    df: pd.DataFrame,
    device: str = 'cpu'
) -> Dict:
    """Train Transformer model."""
    logger.info(f"Training Transformer for {index_name}")
    
    # Clean raw data first (before feature engineering)
    df = clean_data(df)
    logger.info(f"After cleaning raw data: {len(df)} rows")
    
    # Feature engineering
    logger.info(f"Starting feature engineering with {len(df)} rows")
    df_features = engineer_features(df)
    logger.info(f"After feature engineering: {len(df_features)} rows")
    
    # Create target
    df_features = create_target(df_features)
    logger.info(f"After creating target: {len(df_features)} rows")
    
    # Only drop rows where target is NaN (not all feature NaNs)
    initial_len = len(df_features)
    df_features = df_features.dropna(subset=['target'])
    logger.info(f"After dropping NaN targets: {len(df_features)} rows (dropped {initial_len - len(df_features)})")
    
    # Check if we have enough data
    if len(df_features) == 0:
        logger.warning(f"No data available for {index_name} after preprocessing")
        return {
            'predictions': [], 'actuals': [], 'probabilities': [],
            'dates': [], 'attention_weights': [],
            'metrics': {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}
        }
    
    # Fill remaining NaN values in features with forward fill, then backward fill
    feature_cols = get_feature_columns()
    available_feature_cols = [col for col in feature_cols if col in df_features.columns]
    
    # Fill NaNs in feature columns only (not target)
    if available_feature_cols:
        df_features[available_feature_cols] = df_features[available_feature_cols].ffill().bfill()
    
    # Only drop rows where target is still NaN (shouldn't happen, but safety check)
    df_features = df_features.dropna(subset=['target'])
    
    # Check if we have enough data after all processing
    if len(df_features) < LOOKBACK_WINDOW + 10:
        logger.warning(f"Not enough data for {index_name}: {len(df_features)} rows, need at least {LOOKBACK_WINDOW + 10}")
        return {
            'predictions': [], 'actuals': [], 'probabilities': [],
            'dates': [], 'attention_weights': [],
            'metrics': {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}
        }
    
    all_predictions = []
    all_actuals = []
    all_probabilities = []
    all_dates = []
    all_attention_weights = []
    
    # Use simple split for limited data, walk-forward for larger datasets
    if USE_SIMPLE_SPLIT:
        logger.info("Using simple train/test split (limited data mode)")
        data = get_simple_split_data(df_features, LOOKBACK_WINDOW, TRAIN_SPLIT_RATIO, normalize=True)
        if data[0] is None:
            logger.warning("Failed to create simple split data")
            return {
                'predictions': [], 'actuals': [], 'probabilities': [],
                'dates': [], 'attention_weights': [],
                'metrics': {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}
            }
        
        X_train, y_train, X_test, y_test, train_dates, test_dates = data
        
        logger.info(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        if len(X_train) == 0 or len(X_test) == 0:
            logger.warning("Insufficient data after split")
            return {
                'predictions': [], 'actuals': [], 'probabilities': [],
                'dates': [], 'attention_weights': [],
                'metrics': {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}
            }
        
        # Check for NaN/Inf in data
        if np.any(np.isnan(X_train)) or np.any(np.isinf(X_train)):
            logger.warning("NaN/Inf in training data, filling with 0")
            X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        if np.any(np.isnan(X_test)) or np.any(np.isinf(X_test)):
            logger.warning("NaN/Inf in test data, filling with 0")
            X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Create Transformer model
        input_size = X_train.shape[2]
        model = TransformerModel(
            input_size=input_size,
            d_model=256,
            num_heads=8,
            num_layers=4,
            dropout=DROPOUT
        )
        
        # Split for validation
        val_split = int(0.8 * len(X_train))
        X_train_split = X_train[:val_split]
        y_train_split = y_train[:val_split]
        X_val_split = X_train[val_split:]
        y_val_split = y_train[val_split:]
        
        # Loss function
        loss_fn = AsymmetricDirectionalLoss(UPWARD_PENALTY, DOWNWARD_PENALTY)
        
        # Train model
        logger.info("Starting Transformer training...")
        model, history = train_model(
            model=model,
            X_train=X_train_split,
            y_train=y_train_split,
            X_val=X_val_split,
            y_val=y_val_split,
            loss_fn=loss_fn,
            num_epochs=NUM_EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE * 0.5,  # Lower LR for Transformer
            device=device,
            return_attention=True,
            use_augmentation=True,
            weight_decay=WEIGHT_DECAY
        )
        
        # Make predictions
        predictions, attention_weights = walk_forward_predict(
            model, X_test, device=device, return_attention=True
        )
        
        if len(predictions) == 0:
            logger.warning("No predictions generated")
            return {
                'predictions': [], 'actuals': [], 'probabilities': [],
                'dates': [], 'attention_weights': [],
                'metrics': {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}
            }
        
        pred_binary = (predictions.squeeze() > 0.5).astype(int)
        
        logger.info(f"Generated {len(pred_binary)} predictions")
        
        all_predictions.extend(pred_binary.tolist() if isinstance(pred_binary, np.ndarray) else pred_binary)
        all_actuals.extend(y_test.tolist() if isinstance(y_test, np.ndarray) else y_test)
        all_probabilities.extend(predictions.squeeze().tolist() if isinstance(predictions, np.ndarray) else predictions.squeeze())
        all_dates.extend(test_dates.tolist() if hasattr(test_dates, 'tolist') else list(test_dates))
        
        if attention_weights is not None:
            if isinstance(attention_weights, np.ndarray):
                all_attention_weights.extend(attention_weights.tolist())
            else:
                all_attention_weights.extend(attention_weights)
    else:
        # Walk-forward approach
        windows = create_walk_forward_windows(df_features, TRAIN_WINDOW_YEARS, TEST_WINDOW_MONTHS)
        
        for window_idx, (train_end, test_end) in enumerate(windows):
            logger.info(f"Window {window_idx + 1}/{len(windows)}: {train_end.date()} to {test_end.date()}")
            
            data = get_walk_forward_data(df_features, train_end, test_end, LOOKBACK_WINDOW, normalize=True)
            if data[0] is None:
                continue
            
            X_train, y_train, X_test, y_test, train_dates, test_dates = data
            
            logger.info(f"  Training samples: {len(X_train)}, Test samples: {len(X_test)}")
            
            if len(X_train) == 0 or len(X_test) == 0:
                logger.warning(f"  Skipping window - insufficient data")
                continue
            
            # Check for NaN/Inf in data
            if np.any(np.isnan(X_train)) or np.any(np.isinf(X_train)):
                logger.warning(f"  Skipping window - NaN/Inf in training data")
                continue
            if np.any(np.isnan(X_test)) or np.any(np.isinf(X_test)):
                logger.warning(f"  Skipping window - NaN/Inf in test data")
                continue
            
            # Create Transformer model
            input_size = X_train.shape[2]
            model = TransformerModel(
                input_size=input_size,
                d_model=256,
                num_heads=8,
                num_layers=4,
                dropout=DROPOUT
            )
            
            # Split for validation
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
                learning_rate=LEARNING_RATE * 0.5,  # Lower LR for Transformer
                device=device,
                return_attention=True,
                use_augmentation=True,
                weight_decay=WEIGHT_DECAY
            )
            
            # Make predictions
            predictions, attention_weights = walk_forward_predict(
                model, X_test, device=device, return_attention=True
            )
            
            if len(predictions) == 0:
                logger.warning(f"  No predictions generated for window {window_idx + 1}")
                continue
            
            pred_binary = (predictions.squeeze() > 0.5).astype(int)
            
            logger.info(f"  Generated {len(pred_binary)} predictions for window {window_idx + 1}")
            
            all_predictions.extend(pred_binary.tolist() if isinstance(pred_binary, np.ndarray) else pred_binary)
            all_actuals.extend(y_test.tolist() if isinstance(y_test, np.ndarray) else y_test)
            all_probabilities.extend(predictions.squeeze().tolist() if isinstance(predictions, np.ndarray) else predictions.squeeze())
            all_dates.extend(test_dates.tolist() if hasattr(test_dates, 'tolist') else list(test_dates))
            
            if attention_weights is not None:
                if isinstance(attention_weights, np.ndarray):
                    all_attention_weights.extend(attention_weights.tolist())
                else:
                    all_attention_weights.extend(attention_weights)
    
    if len(all_predictions) == 0:
        return {
            'predictions': [], 'actuals': [], 'probabilities': [],
            'dates': [], 'attention_weights': [],
            'metrics': {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}
        }
    
    accuracy = accuracy_score(all_actuals, all_predictions)
    precision = precision_score(all_actuals, all_predictions, zero_division=0)
    recall = recall_score(all_actuals, all_predictions, zero_division=0)
    f1 = f1_score(all_actuals, all_predictions, zero_division=0)
    
    logger.info(f"Transformer Metrics - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, "
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


def train_tcn_lstm_model(
    index_name: str,
    df: pd.DataFrame,
    device: str = 'cpu'
) -> Dict:
    """Train TCN-LSTM hybrid model."""
    logger.info(f"Training TCN-LSTM for {index_name}")
    
    # Clean raw data first
    df = clean_data(df)
    df_features = engineer_features(df)
    df_features = create_target(df_features)
    df_features = df_features.dropna(subset=['target'])
    
    if len(df_features) == 0:
        logger.warning(f"No data available for TCN-LSTM on {index_name} after preprocessing")
        return {
            'predictions': [], 'actuals': [], 'probabilities': [],
            'dates': [], 'attention_weights': [],
            'metrics': {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}
        }
    
    feature_cols = get_feature_columns()
    available_feature_cols = [col for col in feature_cols if col in df_features.columns]
    
    if available_feature_cols:
        df_features[available_feature_cols] = df_features[available_feature_cols].ffill().bfill()
    
    df_features = df_features.dropna(subset=['target'])
    
    if len(df_features) < LOOKBACK_WINDOW + 10:
        logger.warning(f"Not enough data for TCN-LSTM on {index_name}: {len(df_features)} rows")
        return {
            'predictions': [], 'actuals': [], 'probabilities': [],
            'dates': [], 'attention_weights': [],
            'metrics': {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}
        }
    
    windows = create_walk_forward_windows(df_features, TRAIN_WINDOW_YEARS, TEST_WINDOW_MONTHS)
    
    all_predictions = []
    all_actuals = []
    all_probabilities = []
    all_dates = []
    all_attention_weights = []
    
    for window_idx, (train_end, test_end) in enumerate(windows):
        logger.info(f"Window {window_idx + 1}/{len(windows)}: {train_end.date()} to {test_end.date()}")
        
        data = get_walk_forward_data(df_features, train_end, test_end, LOOKBACK_WINDOW, normalize=True)
        if data[0] is None:
            continue
        
        X_train, y_train, X_test, y_test, train_dates, test_dates = data
        
        if len(X_train) == 0 or len(X_test) == 0:
            continue
        
        # Create TCN-LSTM model
        input_size = X_train.shape[2]
        model = TCNLSTM(
            input_size=input_size,
            tcn_channels=[64, 128, 256],
            lstm_hidden_size=256,
            lstm_num_layers=2,
            dropout=DROPOUT
        )
        
        val_split = int(0.8 * len(X_train))
        loss_fn = AsymmetricDirectionalLoss(UPWARD_PENALTY, DOWNWARD_PENALTY)
        
        model, history = train_model(
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
            return_attention=True,
            use_augmentation=True,
            weight_decay=WEIGHT_DECAY
        )
        
        predictions, attention_weights = walk_forward_predict(
            model, X_test, device=device, return_attention=True
        )
        
        pred_binary = (predictions.squeeze() > 0.5).astype(int)
        
        all_predictions.extend(pred_binary)
        all_actuals.extend(y_test)
        all_probabilities.extend(predictions.squeeze())
        all_dates.extend(test_dates)
        all_attention_weights.extend(attention_weights)
    
    if len(all_predictions) == 0:
        return {
            'predictions': [], 'actuals': [], 'probabilities': [],
            'dates': [], 'attention_weights': [],
            'metrics': {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}
        }
    
    accuracy = accuracy_score(all_actuals, all_predictions)
    precision = precision_score(all_actuals, all_predictions, zero_division=0)
    recall = recall_score(all_actuals, all_predictions, zero_division=0)
    f1 = f1_score(all_actuals, all_predictions, zero_division=0)
    
    logger.info(f"TCN-LSTM Metrics - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, "
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


def train_attention_lstm_enhanced(
    index_name: str,
    df: pd.DataFrame,
    device: str = 'cpu'
) -> Dict:
    """Train enhanced Attention LSTM model."""
    logger.info(f"Training Enhanced Attention LSTM for {index_name}")
    
    # Clean raw data first
    df = clean_data(df)
    df_features = engineer_features(df)
    df_features = create_target(df_features)
    df_features = df_features.dropna(subset=['target'])
    
    if len(df_features) == 0:
        logger.warning(f"No data available for Attention LSTM on {index_name} after preprocessing")
        return {
            'predictions': [], 'actuals': [], 'probabilities': [],
            'dates': [], 'attention_weights': [],
            'metrics': {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}
        }
    
    feature_cols = get_feature_columns()
    available_feature_cols = [col for col in feature_cols if col in df_features.columns]
    
    if available_feature_cols:
        df_features[available_feature_cols] = df_features[available_feature_cols].ffill().bfill()
    
    df_features = df_features.dropna(subset=['target'])
    
    if len(df_features) < LOOKBACK_WINDOW + 10:
        logger.warning(f"Not enough data for Attention LSTM on {index_name}: {len(df_features)} rows")
        return {
            'predictions': [], 'actuals': [], 'probabilities': [],
            'dates': [], 'attention_weights': [],
            'metrics': {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}
        }
    
    windows = create_walk_forward_windows(df_features, TRAIN_WINDOW_YEARS, TEST_WINDOW_MONTHS)
    
    all_predictions = []
    all_actuals = []
    all_probabilities = []
    all_dates = []
    all_attention_weights = []
    
    for window_idx, (train_end, test_end) in enumerate(windows):
        logger.info(f"Window {window_idx + 1}/{len(windows)}: {train_end.date()} to {test_end.date()}")
        
        data = get_walk_forward_data(df_features, train_end, test_end, LOOKBACK_WINDOW, normalize=True)
        if data[0] is None:
            continue
        
        X_train, y_train, X_test, y_test, train_dates, test_dates = data
        
        if len(X_train) == 0 or len(X_test) == 0:
            continue
        
        input_size = X_train.shape[2]
        model = AttentionLSTM(
            input_size=input_size,
            hidden_size=LSTM_HIDDEN_SIZE,
            num_layers=LSTM_NUM_LAYERS,
            dropout=DROPOUT,
            use_residual=True
        )
        
        val_split = int(0.8 * len(X_train))
        loss_fn = AsymmetricDirectionalLoss(UPWARD_PENALTY, DOWNWARD_PENALTY)
        
        model, history = train_model(
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
            return_attention=True,
            use_augmentation=True,
            weight_decay=WEIGHT_DECAY
        )
        
        predictions, attention_weights = walk_forward_predict(
            model, X_test, device=device, return_attention=True
        )
        
        pred_binary = (predictions.squeeze() > 0.5).astype(int)
        
        all_predictions.extend(pred_binary)
        all_actuals.extend(y_test)
        all_probabilities.extend(predictions.squeeze())
        all_dates.extend(test_dates)
        all_attention_weights.extend(attention_weights)
    
    if len(all_predictions) == 0:
        return {
            'predictions': [], 'actuals': [], 'probabilities': [],
            'dates': [], 'attention_weights': [],
            'metrics': {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}
        }
    
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


def create_ensemble_predictions(
    transformer_results: Dict,
    tcn_lstm_results: Dict,
    attention_lstm_results: Dict
) -> Dict:
    """Create ensemble predictions from multiple models."""
    logger.info("Creating ensemble predictions")
    
    # Get all dates (should be same across models)
    dates = attention_lstm_results.get('dates', [])
    if not dates:
        dates = transformer_results.get('dates', [])
    if not dates:
        dates = tcn_lstm_results.get('dates', [])
    
    if not dates:
        return {
            'predictions': [], 'actuals': [], 'probabilities': [],
            'dates': [], 'metrics': {'accuracy': 0.0}
        }
    
    # Get predictions from each model
    transformer_probs = np.array(transformer_results.get('probabilities', []))
    tcn_lstm_probs = np.array(tcn_lstm_results.get('probabilities', []))
    attention_lstm_probs = np.array(attention_lstm_results.get('probabilities', []))
    
    # Get actuals (use from any model)
    actuals = attention_lstm_results.get('actuals', [])
    if not actuals:
        actuals = transformer_results.get('actuals', [])
    if not actuals:
        actuals = tcn_lstm_results.get('actuals', [])
    
    # Align lengths
    min_len = min(len(transformer_probs), len(tcn_lstm_probs), len(attention_lstm_probs), len(actuals), len(dates))
    transformer_probs = transformer_probs[:min_len]
    tcn_lstm_probs = tcn_lstm_probs[:min_len]
    attention_lstm_probs = attention_lstm_probs[:min_len]
    actuals = actuals[:min_len]
    dates = dates[:min_len]
    
    # Weighted ensemble (can be tuned based on individual model performance)
    transformer_weight = transformer_results['metrics']['accuracy']
    tcn_lstm_weight = tcn_lstm_results['metrics']['accuracy']
    attention_lstm_weight = attention_lstm_results['metrics']['accuracy']
    
    total_weight = transformer_weight + tcn_lstm_weight + attention_lstm_weight
    if total_weight > 0:
        transformer_weight /= total_weight
        tcn_lstm_weight /= total_weight
        attention_lstm_weight /= total_weight
    else:
        transformer_weight = tcn_lstm_weight = attention_lstm_weight = 1.0 / 3.0
    
    # Ensemble probabilities
    ensemble_probs = (
        transformer_probs * transformer_weight +
        tcn_lstm_probs * tcn_lstm_weight +
        attention_lstm_probs * attention_lstm_weight
    )
    
    # Optimize threshold for better balance
    from backend.utils.threshold_optimizer import optimize_ensemble_threshold
    optimal_threshold, optimal_metrics = optimize_ensemble_threshold(
        np.array(actuals), np.array(ensemble_probs)
    )
    
    # Binary predictions with optimized threshold
    ensemble_preds = (ensemble_probs >= optimal_threshold).astype(int)
    
    # Calculate metrics with optimized threshold
    accuracy = accuracy_score(actuals, ensemble_preds)
    precision = precision_score(actuals, ensemble_preds, zero_division=0)
    recall = recall_score(actuals, ensemble_preds, zero_division=0)
    f1 = f1_score(actuals, ensemble_preds, zero_division=0)
    
    logger.info(f"Ensemble Metrics - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, "
                f"Recall: {recall:.4f}, F1: {f1:.4f}")
    logger.info(f"Ensemble Weights - Transformer: {transformer_weight:.3f}, "
                f"TCN-LSTM: {tcn_lstm_weight:.3f}, Attention-LSTM: {attention_lstm_weight:.3f}")
    
    return {
        'predictions': ensemble_preds.tolist(),
        'actuals': actuals,
        'probabilities': ensemble_probs.tolist(),
        'dates': dates,
        'metrics': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        },
        'weights': {
            'transformer': transformer_weight,
            'tcn_lstm': tcn_lstm_weight,
            'attention_lstm': attention_lstm_weight
        },
        'optimal_threshold': optimal_threshold
    }


def main(skip_if_exists: bool = False):
    """Main training function for advanced models."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Fetch data
    logger.info("Fetching data for all indices...")
    all_data = get_all_indices_data(YEARS_OF_DATA)
    
    for index_name, df in all_data.items():
        if df is None or len(df) == 0:
            logger.warning(f"Skipping {index_name} - no data available")
            continue
        
        if skip_if_exists:
            results_file = os.path.join(RESULTS_DIR, f"{index_name}_advanced_results.pkl")
            if os.path.exists(results_file):
                logger.info(f"Skipping {index_name} - results already exist")
                continue
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {index_name} with Advanced Models")
        logger.info(f"{'='*60}")
        
        try:
            # Train all three models
            transformer_results = train_transformer_model(index_name, df, device)
            tcn_lstm_results = train_tcn_lstm_model(index_name, df, device)
            attention_lstm_results = train_attention_lstm_enhanced(index_name, df, device)
            
            # Create ensemble
            ensemble_results = create_ensemble_predictions(
                transformer_results, tcn_lstm_results, attention_lstm_results
            )
            
            # Backtest ensemble
            if len(ensemble_results['predictions']) > 0:
                backtest_results = backtest_strategy(
                    index_name,
                    df,
                    np.array(ensemble_results['predictions']),
                    np.array(ensemble_results['probabilities']),
                    pd.DatetimeIndex(ensemble_results['dates'])
                )
            else:
                backtest_results = {'equity_curve': {'dates': [], 'values': []}, 'metrics': {}}
            
            # Compile results
            results = {
                'index': index_name,
                'transformer': transformer_results,
                'tcn_lstm': tcn_lstm_results,
                'attention_lstm': attention_lstm_results,
                'ensemble': ensemble_results,
                'equity_curve': backtest_results['equity_curve'],
                'backtest_metrics': backtest_results.get('metrics', {})
            }
            
            # Save results
            results_file = os.path.join(RESULTS_DIR, f"{index_name}_advanced_results.pkl")
            with open(results_file, 'wb') as f:
                pickle.dump(results, f)
            
            logger.info(f"Saved advanced results to {results_file}")
            logger.info(f"Best Model - Ensemble Accuracy: {ensemble_results['metrics']['accuracy']:.4f}")
            
        except Exception as e:
            logger.error(f"Error processing {index_name}: {e}", exc_info=True)
            continue
    
    logger.info("\nAdvanced training complete!")


def backtest_strategy(
    index_name: str,
    df: pd.DataFrame,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    dates: pd.DatetimeIndex
) -> Dict:
    """Backtest strategy and calculate performance metrics."""
    logger.info(f"Backtesting strategy for {index_name}")
    
    prices = df.loc[dates, 'close']
    
    equity_curve, returns, trades = long_only_strategy(
        predictions, probabilities, prices, dates
    )
    
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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train advanced financial forecasting models')
    parser.add_argument('--skip-if-exists', action='store_true',
                       help='Skip training for indices that already have results')
    args = parser.parse_args()
    main(skip_if_exists=args.skip_if_exists)

