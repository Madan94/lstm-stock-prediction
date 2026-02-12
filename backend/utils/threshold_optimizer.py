"""Optimize prediction threshold for better accuracy."""

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve
from typing import Tuple, Dict
import logging

logger = logging.getLogger(__name__)


def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    metric: str = 'f1'
) -> Tuple[float, Dict]:
    """
    Find optimal threshold for binary classification.
    
    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities
        metric: Metric to optimize ('accuracy', 'f1', 'precision', 'recall', 'balanced')
        
    Returns:
        Optimal threshold and metrics at that threshold
    """
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_threshold = 0.5
    best_score = 0.0
    best_metrics = {}
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        
        if len(np.unique(y_pred)) < 2:  # Need both classes
            continue
        
        if metric == 'accuracy':
            score = accuracy_score(y_true, y_pred)
        elif metric == 'f1':
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == 'precision':
            score = precision_score(y_true, y_pred, zero_division=0)
        elif metric == 'recall':
            score = recall_score(y_true, y_pred, zero_division=0)
        elif metric == 'balanced':
            # Balance between precision and recall
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            score = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
        else:
            score = f1_score(y_true, y_pred, zero_division=0)
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
            best_metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'recall': recall_score(y_true, y_pred, zero_division=0),
                'f1_score': f1_score(y_true, y_pred, zero_division=0),
                'threshold': threshold
            }
    
    logger.info(f"Optimal threshold: {best_threshold:.3f} (optimizing for {metric})")
    logger.info(f"Metrics at optimal threshold - Accuracy: {best_metrics['accuracy']:.4f}, "
                f"Precision: {best_metrics['precision']:.4f}, "
                f"Recall: {best_metrics['recall']:.4f}, "
                f"F1: {best_metrics['f1_score']:.4f}")
    
    return best_threshold, best_metrics


def optimize_ensemble_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray
) -> Tuple[float, Dict]:
    """
    Optimize threshold for ensemble predictions.
    
    Args:
        y_true: True binary labels
        y_proba: Ensemble predicted probabilities
        
    Returns:
        Optimal threshold and metrics
    """
    # Try multiple optimization strategies
    threshold_f1, metrics_f1 = find_optimal_threshold(y_true, y_proba, 'f1')
    threshold_balanced, metrics_balanced = find_optimal_threshold(y_true, y_proba, 'balanced')
    threshold_acc, metrics_acc = find_optimal_threshold(y_true, y_proba, 'accuracy')
    
    # Choose best based on F1 score (balanced metric)
    if metrics_f1['f1_score'] >= metrics_balanced['f1_score']:
        return threshold_f1, metrics_f1
    else:
        return threshold_balanced, metrics_balanced



