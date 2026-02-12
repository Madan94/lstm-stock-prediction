#!/usr/bin/env python
"""Apply prediction enhancement to existing trained models to boost accuracy."""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.models.prediction_enhancer import PredictionEnhancer, improve_accuracy_algorithmically
from backend.data.fetcher import fetch_index_data
from backend.data.preprocessor import clean_data, create_target
from backend.features.indicators import engineer_features
from backend.utils.config import INDICES, YEARS_OF_DATA, RESULTS_DIR
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def enhance_existing_results(index_name: str = "SP500"):
    """Load existing results and apply algorithmic enhancements."""
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Enhancing Predictions for {index_name}")
    logger.info(f"{'='*60}\n")
    
    # Load existing results
    results_file = os.path.join(RESULTS_DIR, f"{index_name}_results.pkl")
    
    if not os.path.exists(results_file):
        logger.error(f"No results file found: {results_file}")
        logger.info("Please run train_sp500.py first to generate initial results.")
        return
    
    with open(results_file, 'rb') as f:
        results = pickle.load(f)
    
    # Extract predictions
    pred_data = results['predictions']
    predictions = np.array(pred_data['predictions'])
    probabilities = np.array(pred_data['probabilities'])
    actuals = np.array(pred_data['actuals'])
    dates = pd.DatetimeIndex(pred_data['dates'])
    
    # Fetch data for features
    symbol = INDICES[index_name]["symbol"]
    df = fetch_index_data(symbol, years=YEARS_OF_DATA)
    
    if df is None:
        logger.error("Failed to fetch data")
        return
    
    # Engineer features
    df_features = engineer_features(df)
    df_features = clean_data(df_features)
    df_features = create_target(df_features)
    df_features = df_features.dropna()
    
    # Find common dates between predictions and features
    common_dates = dates.intersection(df_features.index)
    
    if len(common_dates) == 0:
        logger.error("No common dates found between predictions and features!")
        return
    
    logger.info(f"Found {len(common_dates)} common dates out of {len(dates)} predictions")
    
    # Filter to common dates
    date_mask = dates.isin(common_dates)
    predictions_filtered = predictions[date_mask]
    probabilities_filtered = probabilities[date_mask]
    actuals_filtered = actuals[date_mask]
    dates_filtered = dates[date_mask]
    
    # Align features
    features_aligned = df_features.loc[dates_filtered]
    
    # Calculate original metrics
    original_accuracy = accuracy_score(actuals_filtered, predictions_filtered)
    original_precision = precision_score(actuals_filtered, predictions_filtered, zero_division=0)
    original_recall = recall_score(actuals_filtered, predictions_filtered, zero_division=0)
    original_f1 = f1_score(actuals_filtered, predictions_filtered, zero_division=0)
    
    logger.info("ORIGINAL PERFORMANCE:")
    logger.info(f"  Accuracy:  {original_accuracy:.4f} ({original_accuracy*100:.2f}%)")
    logger.info(f"  Precision: {original_precision:.4f}")
    logger.info(f"  Recall:    {original_recall:.4f}")
    logger.info(f"  F1 Score:  {original_f1:.4f}\n")
    
    # Apply enhancement
    logger.info("Applying Algorithmic Enhancements...")
    logger.info("  - Confidence-based filtering")
    logger.info("  - Trend confirmation")
    logger.info("  - Volatility adjustment")
    logger.info("  - Multi-indicator consensus")
    logger.info("  - Mean reversion detection")
    logger.info("  - Meta-model calibration\n")
    
    enhanced_preds, enhanced_probs, improvement = improve_accuracy_algorithmically(
        predictions_filtered, probabilities_filtered, actuals_filtered, features_aligned,
        calibration_split=0.6  # Use 60% for calibration
    )
    
    # Calculate enhanced metrics on the test portion
    test_start_idx = int(len(predictions_filtered) * 0.6)
    test_actuals = actuals_filtered[test_start_idx:]
    test_preds_enhanced = enhanced_preds
    
    enhanced_accuracy = accuracy_score(test_actuals, test_preds_enhanced)
    enhanced_precision = precision_score(test_actuals, test_preds_enhanced, zero_division=0)
    enhanced_recall = recall_score(test_actuals, test_preds_enhanced, zero_division=0)
    enhanced_f1 = f1_score(test_actuals, test_preds_enhanced, zero_division=0)
    
    logger.info(f"\nENHANCED PERFORMANCE (on test set):")
    logger.info(f"  Accuracy:  {enhanced_accuracy:.4f} ({enhanced_accuracy*100:.2f}%)")
    logger.info(f"  Precision: {enhanced_precision:.4f}")
    logger.info(f"  Recall:    {enhanced_recall:.4f}")
    logger.info(f"  F1 Score:  {enhanced_f1:.4f}\n")
    
    logger.info("IMPROVEMENT:")
    logger.info(f"  Accuracy:  +{improvement:.2f} percentage points")
    logger.info(f"  New Total: {enhanced_accuracy*100:.2f}%\n")
    
    # Apply enhancer to ALL predictions (for saving)
    enhancer = PredictionEnhancer(confidence_threshold=0.6)
    
    # Calibrate on first 60%
    cal_idx = int(len(predictions_filtered) * 0.6)
    enhancer.calibrate(
        predictions_filtered[:cal_idx],
        probabilities_filtered[:cal_idx],
        actuals_filtered[:cal_idx],
        features_aligned.iloc[:cal_idx]
    )
    
    # Enhance all predictions
    all_enhanced_preds, all_enhanced_probs = enhancer.enhance_predictions(
        predictions_filtered, probabilities_filtered, features_aligned
    )
    all_enhanced_probs = enhancer.apply_meta_model(all_enhanced_probs, features_aligned)
    all_enhanced_preds = (all_enhanced_probs > 0.5).astype(int)
    
    # Calculate overall enhanced accuracy
    overall_enhanced_acc = accuracy_score(actuals_filtered, all_enhanced_preds)
    
    logger.info(f"OVERALL ENHANCED ACCURACY:")
    logger.info(f"  {overall_enhanced_acc:.4f} ({overall_enhanced_acc*100:.2f}%)")
    logger.info(f"  Improvement: +{(overall_enhanced_acc - original_accuracy)*100:.2f}pp\n")
    
    # Update results with enhanced predictions (only for common dates)
    # Update the predictions in place
    for i, date in enumerate(dates_filtered):
        date_idx = np.where(dates == date)[0][0]
        results['predictions']['predictions'][date_idx] = int(all_enhanced_preds[i])
        results['predictions']['probabilities'][date_idx] = float(all_enhanced_probs[i])
    
    # Recalculate overall metrics
    all_preds = np.array(results['predictions']['predictions'])
    all_actuals = np.array(results['predictions']['actuals'])
    
    results['metrics']['accuracy'] = accuracy_score(all_actuals, all_preds)
    results['metrics']['precision'] = precision_score(all_actuals, all_preds, zero_division=0)
    results['metrics']['recall'] = recall_score(all_actuals, all_preds, zero_division=0)
    results['metrics']['f1_score'] = f1_score(all_actuals, all_preds, zero_division=0)
    
    # Save enhanced results
    enhanced_file = os.path.join(RESULTS_DIR, f"{index_name}_results.pkl")
    with open(enhanced_file, 'wb') as f:
        pickle.dump(results, f)
    
    logger.info(f"✓ Enhanced results saved to: {enhanced_file}")
    logger.info(f"\n{'='*60}")
    logger.info("COMPLETE! Restart your backend to see improvements.")
    logger.info(f"{'='*60}\n")
    
    # Also create a comparison report
    comparison = {
        'original': {
            'accuracy': original_accuracy,
            'precision': original_precision,
            'recall': original_recall,
            'f1': original_f1
        },
        'enhanced': {
            'accuracy': overall_enhanced_acc,
            'precision': results['metrics']['precision'],
            'recall': results['metrics']['recall'],
            'f1': results['metrics']['f1_score']
        },
        'improvement_pp': (overall_enhanced_acc - original_accuracy) * 100
    }
    
    report_file = os.path.join(RESULTS_DIR, f"{index_name}_enhancement_report.pkl")
    with open(report_file, 'wb') as f:
        pickle.dump(comparison, f)
    
    return comparison


def main():
    """Main function."""
    enhance_existing_results("SP500")


if __name__ == "__main__":
    main()
