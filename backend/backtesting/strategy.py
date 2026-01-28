"""Backtesting strategy implementation."""

import pandas as pd
import numpy as np
from typing import Tuple, Dict
from backend.utils.config import TRANSACTION_COST
import logging

logger = logging.getLogger(__name__)


def long_only_strategy(
    predictions: np.ndarray,
    probabilities: np.ndarray,
    prices: pd.Series,
    dates: pd.DatetimeIndex,
    transaction_cost: float = TRANSACTION_COST
) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
    """
    Implement long-only strategy based on predicted direction.
    
    Strategy: Buy when predicted probability > 0.5 (upward prediction)
    
    Args:
        predictions: Binary predictions (1 = up, 0 = down)
        probabilities: Prediction probabilities
        prices: Price series aligned with predictions
        dates: Date index for predictions
        transaction_cost: Transaction cost as fraction (e.g., 0.001 = 0.1%)
        
    Returns:
        equity_curve: Portfolio value over time
        returns: Daily returns
        trades: DataFrame with trade information
    """
    # Initialize portfolio
    initial_capital = 100000  # $100k starting capital
    portfolio_value = initial_capital
    position = 0  # 0 = no position, 1 = long position
    entry_price = None
    
    equity_values = []
    daily_returns = []
    trades = []
    
    for i, (date, pred, prob) in enumerate(zip(dates, predictions, probabilities)):
        current_price = prices.loc[date]
        
        # Trading logic: buy if prediction is up and not already in position
        if pred == 1 and position == 0:
            # Enter long position
            position = 1
            entry_price = current_price
            # Pay transaction cost
            portfolio_value *= (1 - transaction_cost)
            
            trades.append({
                'date': date,
                'action': 'BUY',
                'price': entry_price,
                'probability': prob
            })
        
        # Exit position if prediction is down and in position
        elif pred == 0 and position == 1:
            # Exit long position
            exit_price = current_price
            # Calculate return
            trade_return = (exit_price / entry_price - 1)
            portfolio_value *= (1 + trade_return)
            # Pay transaction cost
            portfolio_value *= (1 - transaction_cost)
            
            trades.append({
                'date': date,
                'action': 'SELL',
                'price': exit_price,
                'entry_price': entry_price,
                'return': trade_return,
                'probability': prob
            })
            
            position = 0
            entry_price = None
        
        # Update portfolio value if in position
        if position == 1:
            # Mark-to-market: update portfolio value based on price movement
            current_return = (current_price / entry_price - 1)
            # Portfolio value tracks the position value
            portfolio_value = initial_capital * (1 + current_return)
        
        # Calculate daily return
        if i == 0:
            daily_return = 0.0
        else:
            prev_price = prices.loc[dates[i-1]]
            if position == 1:
                # If in position, return is based on price change
                daily_return = (current_price / prev_price - 1)
            else:
                # If not in position, no return
                daily_return = 0.0
        
        equity_values.append(portfolio_value)
        daily_returns.append(daily_return)
    
    # Close any open position at the end
    if position == 1 and entry_price is not None:
        final_price = prices.loc[dates[-1]]
        trade_return = (final_price / entry_price - 1)
        portfolio_value *= (1 + trade_return)
        portfolio_value *= (1 - transaction_cost)
        
        trades.append({
            'date': dates[-1],
            'action': 'SELL',
            'price': final_price,
            'entry_price': entry_price,
            'return': trade_return,
            'probability': probabilities[-1]
        })
    
    # Create DataFrames
    equity_curve = pd.Series(equity_values, index=dates)
    returns = pd.Series(daily_returns, index=dates)
    trades_df = pd.DataFrame(trades)
    
    if len(trades_df) > 0:
        # Calculate returns for all trades
        buy_trades = trades_df[trades_df['action'] == 'BUY']
        sell_trades = trades_df[trades_df['action'] == 'SELL']
        
        # Match buy and sell trades
        for idx, sell_trade in sell_trades.iterrows():
            # Find corresponding buy trade
            buy_trade = buy_trades[buy_trades['date'] < sell_trade['date']].iloc[-1]
            if 'return' not in sell_trade or pd.isna(sell_trade['return']):
                sell_trades.loc[idx, 'return'] = (
                    sell_trade['price'] / buy_trade['price'] - 1
                )
        
        trades_df = sell_trades[sell_trades['action'] == 'SELL'].copy()
    
    return equity_curve, returns, trades_df

