import { useState, useEffect } from 'react';
import { useRouter } from 'next/router';
import { getIndices } from '../services/api';
import type { IndexInfo } from '../types';

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
      <div className="text-center mb-12">
        <h1 className="text-5xl font-bold text-gray-900 mb-4">
          Financial Forecasting Platform
        </h1>
        <p className="text-xl text-gray-600 mb-2">
          Directional Financial Forecasting with Asymmetric Loss
        </p>
        <p className="text-gray-500">
          Advanced attention-based LSTM models for predicting market direction
        </p>
      </div>

      {loading ? (
        <div className="text-center text-gray-500">Loading indices...</div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 max-w-4xl mx-auto">
          {indices.map((index) => (
            <button
              key={index.name}
              onClick={() => handleIndexSelect(index.name)}
              className="bg-white border-2 border-gray-200 rounded-lg p-8 hover:border-leaf hover:shadow-lg transition-all text-left"
            >
              <h2 className="text-2xl font-semibold text-gray-900 mb-2">
                {index.display_name}
              </h2>
              <p className="text-gray-500 text-sm mb-4">{index.symbol}</p>
              <p className="text-gray-700">
                View predictions, attention weights, and strategy performance →
              </p>
            </button>
          ))}
        </div>
      )}

      <div className="mt-16 max-w-3xl mx-auto">
        <h2 className="text-2xl font-semibold text-gray-900 mb-6">Features</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="bg-white border border-gray-200 rounded-lg p-6 shadow-sm">
            <h3 className="text-lg font-semibold text-gray-900 mb-2">Attention-Based LSTM</h3>
            <p className="text-gray-600">
              Neural network with attention mechanism to identify important time periods
            </p>
          </div>
          <div className="bg-white border border-gray-200 rounded-lg p-6 shadow-sm">
            <h3 className="text-lg font-semibold text-gray-900 mb-2">Asymmetric Loss</h3>
            <p className="text-gray-600">
              Custom loss function that penalizes missing upward moves more heavily
            </p>
          </div>
          <div className="bg-white border border-gray-200 rounded-lg p-6 shadow-sm">
            <h3 className="text-lg font-semibold text-gray-900 mb-2">Walk-Forward Validation</h3>
            <p className="text-gray-600">
              Robust training methodology using walk-forward analysis
            </p>
          </div>
          <div className="bg-white border border-gray-200 rounded-lg p-6 shadow-sm">
            <h3 className="text-lg font-semibold text-gray-900 mb-2">Strategy Backtesting</h3>
            <p className="text-gray-600">
              Long-only strategy with transaction costs and performance metrics
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}



