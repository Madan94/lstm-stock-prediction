"""Backtesting performance metrics."""

import numpy as np
import pandas as pd
from typing import Dict
from backend.utils.config import TRADING_DAYS_PER_YEAR


def calculate_sharpe_ratio(returns: pd.Series, annualized: bool = True) -> float:
    """
    Calculate Sharpe ratio.
    
    Args:
        returns: Series of returns
        annualized: Whether to annualize the ratio
        
    Returns:
        Sharpe ratio
    """
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    
    mean_return = returns.mean()
    std_return = returns.std()
    
    if std_return == 0:
        return 0.0
    
    sharpe = mean_return / std_return
    
    if annualized:
        # Annualize assuming daily returns
        sharpe *= np.sqrt(TRADING_DAYS_PER_YEAR)
    
    return sharpe


def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    """
    Calculate maximum drawdown.
    
    Args:
        equity_curve: Series of portfolio values over time
        
    Returns:
        Maximum drawdown as a percentage
    """
    if len(equity_curve) == 0:
        return 0.0
    
    # Calculate running maximum
    running_max = equity_curve.expanding().max()
    
    # Calculate drawdown
    drawdown = (equity_curve - running_max) / running_max
    
    return abs(drawdown.min())


def calculate_total_return(equity_curve: pd.Series) -> float:
    """
    Calculate total return.
    
    Args:
        equity_curve: Series of portfolio values over time
        
    Returns:
        Total return as a percentage
    """
    if len(equity_curve) == 0:
        return 0.0
    
    return (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100


def calculate_win_rate(trades: pd.DataFrame) -> float:
    """
    Calculate win rate of trades.
    
    Args:
        trades: DataFrame with 'return' column
        
    Returns:
        Win rate as a percentage
    """
    if len(trades) == 0:
        return 0.0
    
    winning_trades = (trades['return'] > 0).sum()
    return (winning_trades / len(trades)) * 100


def calculate_all_metrics(
    equity_curve: pd.Series,
    returns: pd.Series,
    trades: pd.DataFrame
) -> Dict[str, float]:
    """
    Calculate all performance metrics.
    
    Args:
        equity_curve: Portfolio values over time
        returns: Daily returns
        trades: DataFrame with trade information
        
    Returns:
        Dictionary of metrics
    """
    return {
        'total_return': calculate_total_return(equity_curve),
        'sharpe_ratio': calculate_sharpe_ratio(returns),
        'max_drawdown': calculate_max_drawdown(equity_curve),
        'win_rate': calculate_win_rate(trades),
        'num_trades': len(trades)
    }









