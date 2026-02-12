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
      {/* Hero Section with Animated Chart Background */}
      <div className="relative mb-16 rounded-lg overflow-hidden shadow-xl">
        {/* Background Chart */}
        <div className="absolute inset-0 opacity-30">
          <AnimatedStockChart />
          <FloatingNumbers />
        </div>
        
        {/* Overlay for better text readability */}
        <div className="absolute inset-0 bg-gradient-to-b from-white/95 via-white/90 to-white/95"></div>
        
        {/* Content Overlay */}
        <div className="relative z-10 px-8 py-16 md:px-12 md:py-20">
          {/* LIVE Badge */}
          <div className="absolute top-6 right-6 flex gap-2 z-20">
            <span className="px-3 py-1 bg-trading-green/20 text-trading-green rounded-full text-sm font-semibold animate-pulse backdrop-blur-sm">
              LIVE
            </span>
          </div>
          
          {/* Hero Text Content */}
          <div className="text-center max-w-4xl mx-auto">
            <h1 className="text-5xl md:text-6xl font-bold text-black mb-4">
              Attention-based LSTM based Stock Prediction
            </h1>
            <p className="text-xl md:text-2xl text-black/80 mb-4 font-medium">
              Directional Financial Forecasting with Asymmetric Loss
            </p>
            <p className="text-lg text-black/70 max-w-3xl mx-auto leading-relaxed">
              To develop an enhanced financial time series forecasting model that integrates asymmetric loss functions within an attention-based LSTM framework to significantly improve directional prediction accuracy for major stock indices (S&P 500, DJI, NASDAQ Composite) over a 33-year period.
            </p>
          </div>
        </div>
      </div>

      {/* Stock Indices Selection */}
      <div className="mb-16">
        <h2 className="text-3xl font-bold text-black mb-8 text-center">Select Stock Index</h2>
        {loading ? (
          <div className="text-center text-black/60">Loading indices...</div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 max-w-4xl mx-auto">
            {indices.map((index) => (
              <button
                key={index.name}
                onClick={() => handleIndexSelect(index.name)}
                className="bg-[#22c55e] border-2 border-gray-200 rounded-lg p-8 hover:border-trading-green hover:shadow-lg hover:shadow-trading-green/20 transition-all text-left relative overflow-hidden group"
              >
                {/* Animated background effect */}
                <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent transform -translate-x-full group-hover:translate-x-full transition-transform duration-1000" />
                <div className="relative z-10">
                  <h2 className="text-2xl font-semibold text-white mb-2">
                    {index.display_name}
                  </h2>
                  <p className="text-white/60 text-sm mb-4">{index.symbol}</p>
                  <p className="text-white/80">
                    View predictions, attention weights, and strategy performance →
                  </p>
                </div>
              </button>
            ))}
          </div>
        )}
      </div>

      <div className="mt-16 max-w-3xl mx-auto">
        <h2 className="text-2xl font-semibold text-black mb-6">Platform Features</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="bg-white border border-gray-200 rounded-lg p-6 shadow-sm">
            <h3 className="text-lg font-semibold text-black mb-2">Attention-Based LSTM</h3>
            <p className="text-black/70">
              Neural network with attention mechanism to identify important time periods
            </p>
          </div>
          <div className="bg-white border border-gray-200 rounded-lg p-6 shadow-sm">
            <h3 className="text-lg font-semibold text-black mb-2">Asymmetric Loss</h3>
            <p className="text-black/70">
              Custom loss function that penalizes missing upward moves more heavily
            </p>
          </div>
          <div className="bg-white border border-gray-200 rounded-lg p-6 shadow-sm">
            <h3 className="text-lg font-semibold text-black mb-2">Walk-Forward Validation</h3>
            <p className="text-black/70">
              Robust training methodology using walk-forward analysis
            </p>
          </div>
          <div className="bg-white border border-gray-200 rounded-lg p-6 shadow-sm">
            <h3 className="text-lg font-semibold text-black mb-2">Strategy Backtesting</h3>
            <p className="text-black/70">
              Long-only strategy with transaction costs and performance metrics
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}



