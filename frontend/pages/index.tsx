import { useState, useEffect } from 'react';
import { useRouter } from 'next/router';
import { getIndices } from '../services/api';
import type { IndexInfo } from '../types';
import AnimatedStockChart from '../components/AnimatedStockChart';
import FinancialIndicators from '../components/FinancialIndicators';
import FloatingNumbers from '../components/FloatingNumbers';
import CandlestickChart from '../components/CandlestickChart';

export default function Home() {
  const router = useRouter();
  const [indices, setIndices] = useState<IndexInfo[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getIndices()
      .then((data) => {
        setIndices(data);
        setLoading(false);
      })
      .catch((error) => {
        console.error('Error fetching indices:', error);
        setLoading(false);
      });
  }, []);

  const handleIndexSelect = (indexName: string) => {
    router.push({
      pathname: '/overview',
      query: { index: indexName },
    });
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
      {/* Hero Section with Animated Background */}
      <div className="relative mb-16 rounded-lg overflow-hidden border-2 border-black bg-white">
        <div className="absolute inset-0 opacity-20">
          <AnimatedStockChart />
          <FloatingNumbers />
        </div>

        <div className="absolute inset-0 bg-white"></div>

        <div className="relative z-10 px-8 py-16 md:px-12 md:py-20">
          <div className="absolute top-6 right-6 flex gap-2 z-20">
            <span className="px-3 py-1 bg-white text-black border-2 border-black rounded-full text-sm font-semibold">
              LIVE
            </span>
          </div>

          <div className="text-center max-w-4xl mx-auto">
            <h1 className="text-5xl md:text-6xl font-bold text-black mb-4">
              Attention-based LSTM Stock Prediction
            </h1>
            <p className="text-xl md:text-2xl text-black mb-4 font-medium">
              Directional Financial Forecasting
            </p>
            <p className="text-lg text-black max-w-3xl mx-auto leading-relaxed">
              Trained on the last 1 year of market data for S&amp;P 500, Dow Jones, and NASDAQ Composite.
            </p>
          </div>

          <div className="mt-10 flex justify-center">
            <FinancialIndicators />
          </div>
        </div>
      </div>

      {/* Stock Indices Selection */}
      <div className="mb-16">
        <h2 className="text-3xl font-bold text-black mb-8 text-center">Select Stock Index</h2>
        {loading ? (
          <div className="text-center text-black">Loading indices...</div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 max-w-4xl mx-auto">
            {indices.map((index) => (
              <button
                key={index.name}
                onClick={() => handleIndexSelect(index.name)}
                className="trading-card text-left relative overflow-hidden group"
              >
                <div className="absolute inset-0 bg-black opacity-0 group-hover:opacity-5 transition-opacity" />
                <div className="relative z-10">
                  <h2 className="text-2xl font-semibold text-black mb-2">
                    {index.display_name}
                  </h2>
                  <p className="text-black text-sm mb-4 font-mono">{index.symbol}</p>
                  <p className="text-black font-medium">
                    View predictions, attention weights, and strategy performance →
                  </p>
                </div>
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Visual (Static) Candlestick */}
      <div className="mb-16 chart-container">
        <h2 className="text-2xl font-semibold text-black mb-6 text-center">Sample Candlestick Pattern</h2>
        <CandlestickChart />
      </div>

      {/* Features */}
      <div className="mt-16 max-w-6xl mx-auto">
        <h2 className="text-3xl font-bold text-black mb-8 text-center">Platform Features</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="trading-card">
            <h3 className="text-xl font-semibold text-black mb-2">Attention-Based LSTM</h3>
            <p className="text-black">
              Neural network with attention mechanism to highlight important time periods.
            </p>
          </div>
          <div className="trading-card">
            <h3 className="text-xl font-semibold text-black mb-2">Asymmetric Loss</h3>
            <p className="text-black">
              Custom loss function that penalizes missed upward moves more heavily.
            </p>
          </div>
          <div className="trading-card">
            <h3 className="text-xl font-semibold text-black mb-2">Walk-Forward / Simple Split</h3>
            <p className="text-black">
              Robust time-series evaluation designed to avoid look-ahead bias.
            </p>
          </div>
          <div className="trading-card">
            <h3 className="text-xl font-semibold text-black mb-2">Strategy Backtesting</h3>
            <p className="text-black">
              Backtest with equity curve and drawdown visualization.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}



