"""Advanced prediction enhancement algorithms for improved accuracy."""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from scipy import stats
from sklearn.linear_model import LogisticRegression


class PredictionEnhancer:
    """
    Advanced post-processing to improve prediction accuracy without retraining.
    
    Uses ensemble of techniques:
    - Confidence-based filtering
    - Trend confirmation
    - Volatility-adjusted predictions
    - Multi-timeframe consensus
    """
    
    def __init__(self, confidence_threshold: float = 0.55):
        self.confidence_threshold = confidence_threshold
        self.meta_model = None
        
    def enhance_predictions(
        self,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        features: pd.DataFrame,
        use_confidence_filter: bool = True,
        use_trend_confirmation: bool = True,
        use_volatility_adjustment: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Enhance predictions using multiple strategies.
        
        Args:
            predictions: Binary predictions (0/1)
            probabilities: Prediction probabilities
            features: DataFrame with technical features
            use_confidence_filter: Apply confidence-based filtering
            use_trend_confirmation: Require trend confirmation
            use_volatility_adjustment: Adjust for volatility
            
        Returns:
            enhanced_predictions: Improved predictions
            enhanced_probabilities: Adjusted probabilities
        """
        enhanced_probs = probabilities.copy()
        enhanced_preds = predictions.copy()
        
        # Strategy 1: Confidence filtering
        if use_confidence_filter:
            enhanced_probs = self._apply_confidence_filter(
                enhanced_probs, features
            )
        
        # Strategy 2: Trend confirmation
        if use_trend_confirmation and 'sma_20' in features.columns:
            enhanced_probs = self._apply_trend_confirmation(
                enhanced_probs, features
            )
        
        # Strategy 3: Volatility adjustment
        if use_volatility_adjustment and 'volatility' in features.columns:
            enhanced_probs = self._apply_volatility_adjustment(
                enhanced_probs, features
            )
        
        # Strategy 4: Multi-indicator consensus
        if self._has_required_features(features):
            enhanced_probs = self._apply_consensus_voting(
                enhanced_probs, features
            )
        
        # Strategy 5: Mean reversion detection
        enhanced_probs = self._apply_mean_reversion(
            enhanced_probs, features
            )
        
        # Convert back to binary predictions
        enhanced_preds = (enhanced_probs > 0.5).astype(int)
        
        return enhanced_preds, enhanced_probs
    
    def _apply_confidence_filter(
        self,
        probabilities: np.ndarray,
        features: pd.DataFrame
    ) -> np.ndarray:
        """Only trade when confidence is high, else predict neutral (0.5)."""
        adjusted_probs = probabilities.copy()
        
        # Calculate prediction confidence (distance from 0.5)
        confidence = np.abs(probabilities - 0.5)
        
        # If confidence is low, pull toward neutral
        low_confidence_mask = confidence < (self.confidence_threshold - 0.5)
        adjusted_probs[low_confidence_mask] = 0.5 + (
            (probabilities[low_confidence_mask] - 0.5) * 0.3
        )
        
        return adjusted_probs
    
    def _apply_trend_confirmation(
        self,
        probabilities: np.ndarray,
        features: pd.DataFrame
    ) -> np.ndarray:
        """Boost predictions that align with prevailing trend."""
        adjusted_probs = probabilities.copy()
        
        # Check if price is above/below MA
        if 'close' not in features.columns or 'sma_50' not in features.columns:
            return adjusted_probs
        
        in_uptrend = (features['close'] > features['sma_50']).values
        
        # Boost bullish predictions in uptrend
        bullish_mask = probabilities > 0.5
        adjusted_probs[bullish_mask & in_uptrend] *= 1.15
        
        # Reduce bullish predictions in downtrend
        adjusted_probs[bullish_mask & ~in_uptrend] *= 0.85
        
        # Clip to [0, 1]
        adjusted_probs = np.clip(adjusted_probs, 0, 1)
        
        return adjusted_probs
    
    def _apply_volatility_adjustment(
        self,
        probabilities: np.ndarray,
        features: pd.DataFrame
    ) -> np.ndarray:
        """Be more conservative in high volatility periods."""
        adjusted_probs = probabilities.copy()
        
        if 'volatility' not in features.columns:
            return adjusted_probs
        
        volatility = features['volatility'].values
        vol_percentile = features.get('volatility_percentile', pd.Series(0.5, index=features.index)).values
        
        # In high volatility, reduce confidence
        high_vol_mask = vol_percentile > 0.8
        adjusted_probs[high_vol_mask] = 0.5 + (
            (probabilities[high_vol_mask] - 0.5) * 0.7
        )
        
        return adjusted_probs
    
    def _apply_consensus_voting(
        self,
        probabilities: np.ndarray,
        features: pd.DataFrame
    ) -> np.ndarray:
        """Combine multiple technical indicators for consensus."""
        adjusted_probs = probabilities.copy()
        
        # Collect indicator signals
        signals = []
        
        # RSI signal
        if 'rsi' in features.columns:
            rsi = features['rsi'].values
            rsi_signal = np.where(rsi > 50, 1, 0)
            signals.append(rsi_signal)
        
        # MACD signal
        if 'macd' in features.columns and 'macd_signal' in features.columns:
            macd_bullish = (features['macd'] > features['macd_signal']).values.astype(int)
            signals.append(macd_bullish)
        
        # Stochastic signal
        if 'stoch_k' in features.columns:
            stoch_bullish = (features['stoch_k'] > 50).values.astype(int)
            signals.append(stoch_bullish)
        
        # Moving average signal
        if 'sma_5_10_cross' in features.columns:
            signals.append(features['sma_5_10_cross'].values)
        
        # Price momentum
        if 'momentum_20' in features.columns:
            mom_bullish = (features['momentum_20'] > 0).values.astype(int)
            signals.append(mom_bullish)
        
        if len(signals) == 0:
            return adjusted_probs
        
        # Calculate consensus (percentage of bullish indicators)
        consensus = np.mean(signals, axis=0)
        
        # Blend model prediction with technical consensus
        # Give 60% weight to model, 40% to technical consensus
        adjusted_probs = 0.6 * probabilities + 0.4 * consensus
        
        return adjusted_probs
    
    def _apply_mean_reversion(
        self,
        probabilities: np.ndarray,
        features: pd.DataFrame
    ) -> np.ndarray:
        """Detect mean reversion setups to refine predictions."""
        adjusted_probs = probabilities.copy()
        
        # Check if price is extended from mean
        if 'bb_position' not in features.columns:
            return adjusted_probs
        
        bb_pos = features['bb_position'].values
        
        # If price is overbought (> 0.9), reduce bullish probability
        overbought = bb_pos > 0.95
        adjusted_probs[overbought] = np.minimum(
            adjusted_probs[overbought],
            0.5 - (bb_pos[overbought] - 0.95) * 5
        )
        
        # If price is oversold (< 0.1), increase bullish probability
        oversold = bb_pos < 0.05
        adjusted_probs[oversold] = np.maximum(
            adjusted_probs[oversold],
            0.5 + (0.05 - bb_pos[oversold]) * 5
        )
        
        return adjusted_probs
    
    def _has_required_features(self, features: pd.DataFrame) -> bool:
        """Check if required features exist."""
        required = ['rsi', 'macd', 'macd_signal']
        return all(feat in features.columns for feat in required)
    
    def calibrate(
        self,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        actuals: np.ndarray,
        features: pd.DataFrame
    ):
        """
        Calibrate the enhancer using historical data.
        Trains a meta-model to learn optimal probability adjustments.
        """
        # Create meta-features
        meta_features = self._create_meta_features(
            predictions, probabilities, features
        )
        
        # Train logistic regression to adjust probabilities
        self.meta_model = LogisticRegression(max_iter=1000)
        self.meta_model.fit(meta_features, actuals)
        
    def _create_meta_features(
        self,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        features: pd.DataFrame
    ) -> np.ndarray:
        """Create features for meta-model."""
        meta_feats = []
        
        # Base probability
        meta_feats.append(probabilities)
        
        # Probability squared (capture non-linearity)
        meta_feats.append(probabilities ** 2)
        
        # Confidence (distance from 0.5)
        meta_feats.append(np.abs(probabilities - 0.5))
        
        # Add technical features if available
        if 'rsi' in features.columns:
            meta_feats.append(features['rsi'].values / 100)
        
        if 'volatility_percentile' in features.columns:
            meta_feats.append(features['volatility_percentile'].values)
        
        if 'momentum_20' in features.columns:
            meta_feats.append(np.clip(features['momentum_20'].values, -0.1, 0.1) * 10)
        
        return np.column_stack(meta_feats)
    
    def apply_meta_model(
        self,
        probabilities: np.ndarray,
        features: pd.DataFrame
    ) -> np.ndarray:
        """Apply calibrated meta-model if available."""
        if self.meta_model is None:
            return probabilities
        
        # Create dummy predictions
        predictions = (probabilities > 0.5).astype(int)
        meta_features = self._create_meta_features(
            predictions, probabilities, features
        )
        
        # Get calibrated probabilities
        calibrated_probs = self.meta_model.predict_proba(meta_features)[:, 1]
        
        return calibrated_probs


def improve_accuracy_algorithmically(
    predictions: np.ndarray,
    probabilities: np.ndarray,
    actuals: np.ndarray,
    features: pd.DataFrame,
    calibration_split: float = 0.7
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Main function to improve accuracy using algorithmic post-processing.
    
    Args:
        predictions: Model predictions
        probabilities: Model probabilities
        actuals: True labels
        features: Technical features DataFrame
        calibration_split: Fraction of data for calibration
        
    Returns:
        enhanced_predictions: Improved predictions
        enhanced_probabilities: Adjusted probabilities
        accuracy_improvement: Percentage point improvement
    """
    from sklearn.metrics import accuracy_score
    
    # Split data for calibration and testing
    split_idx = int(len(predictions) * calibration_split)
    
    # Calibration set
    cal_preds = predictions[:split_idx]
    cal_probs = probabilities[:split_idx]
    cal_actuals = actuals[:split_idx]
    cal_features = features.iloc[:split_idx]
    
    # Test set
    test_preds = predictions[split_idx:]
    test_probs = probabilities[split_idx:]
    test_actuals = actuals[split_idx:]
    test_features = features.iloc[split_idx:]
    
    # Create and calibrate enhancer
    enhancer = PredictionEnhancer(confidence_threshold=0.6)
    enhancer.calibrate(cal_preds, cal_probs, cal_actuals, cal_features)
    
    # Enhance test predictions
    enhanced_preds, enhanced_probs = enhancer.enhance_predictions(
        test_preds, test_probs, test_features
    )
    
    # Apply meta-model
    enhanced_probs = enhancer.apply_meta_model(enhanced_probs, test_features)
    enhanced_preds = (enhanced_probs > 0.5).astype(int)
    
    # Calculate improvement
    original_acc = accuracy_score(test_actuals, test_preds)
    enhanced_acc = accuracy_score(test_actuals, enhanced_preds)
    improvement = (enhanced_acc - original_acc) * 100
    
    print(f"Original Accuracy: {original_acc:.4f}")
    print(f"Enhanced Accuracy: {enhanced_acc:.4f}")
    print(f"Improvement: +{improvement:.2f} percentage points")
    
    return enhanced_preds, enhanced_probs, improvement
