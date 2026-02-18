import { useState, useEffect } from 'react';
import { useRouter } from 'next/router';
import { getMetrics, getBaselineComparison } from '../services/api';
import type { Metrics, BaselineModel } from '../types';
import MetricCard from '../components/Cards/MetricCard';
import ComparisonChart from '../components/Cards/ComparisonChart';
import Sidebar from '../components/Layout/Sidebar';

export default function Overview() {
  const router = useRouter();
  const { index } = router.query;
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [baseline, setBaseline] = useState<BaselineModel[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!index || typeof index !== 'string') return;

    Promise.all([
      getMetrics(index),
      getBaselineComparison(index),
    ])
      .then(([metricsData, baselineData]) => {
        setMetrics(metricsData);
        // Filter out ARIMA
        const filteredBaseline = baselineData.filter((model) => 
          !model.model_name.toLowerCase().includes('arima')
        );
        setBaseline(filteredBaseline);
        setLoading(false);
      })
      .catch((error) => {
        console.error('Error fetching data:', error);
        setLoading(false);
      });
  }, [index]);

  if (!index) {
    return (
      <div className="flex min-h-screen">
        <Sidebar />
        <div className="flex-1 p-8 flex items-center justify-center">
          <div className="text-center">
            <div className="w-16 h-16 rounded-full border-2 border-black bg-white flex items-center justify-center mx-auto mb-4">
              <svg className="w-8 h-8 text-black" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
            </div>
            <p className="text-black">Please select an index from the sidebar</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex min-h-screen">
      <Sidebar />
      <div className="flex-1 p-8">
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-black mb-2">Performance Dashboard</h1>
          <p className="text-black">
            Ensemble model combining Transformer, TCN-LSTM, and Attention-LSTM architectures
          </p>
        </div>

        {loading ? (
          <div className="flex items-center justify-center py-20">
            <div className="text-center">
              <div className="inline-block animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-black mb-4"></div>
              <p className="text-black">Loading performance metrics...</p>
            </div>
          </div>
        ) : (
          <>
            <div className="mb-8">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-2xl font-semibold text-black">Key Metrics</h2>
                <div className="flex items-center space-x-2 px-3 py-1 rounded-full border-2 border-black bg-white">
                  <div className="w-2 h-2 rounded-full bg-black animate-pulse"></div>
                  <span className="text-black text-xs font-medium">Live Data</span>
                </div>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                <MetricCard
                  title="Accuracy"
                  value={metrics?.accuracy ? (metrics.accuracy * 100).toFixed(2) : '0.00'}
                  subtitle="%"
                  trend={metrics?.accuracy && metrics.accuracy > 0.5 ? 'up' : 'neutral'}
                />
                <MetricCard
                  title="Precision"
                  value={metrics?.precision ? (metrics.precision * 100).toFixed(2) : '0.00'}
                  subtitle="%"
                  trend={metrics?.precision && metrics.precision > 0.5 ? 'up' : 'neutral'}
                />
                <MetricCard
                  title="Recall"
                  value={metrics?.recall ? (metrics.recall * 100).toFixed(2) : '0.00'}
                  subtitle="%"
                  trend={metrics?.recall && metrics.recall > 0.5 ? 'up' : 'neutral'}
                />
                <MetricCard
                  title="F1 Score"
                  value={metrics?.f1_score ? (metrics.f1_score * 100).toFixed(2) : '0.00'}
                  subtitle="%"
                  trend={metrics?.f1_score && metrics.f1_score > 0.5 ? 'up' : 'neutral'}
                />
              </div>
            </div>

            <div className="mb-8">
              <h2 className="text-2xl font-semibold text-black mb-6">Model Comparison</h2>
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div className="chart-container">
                  <ComparisonChart data={baseline} metric="accuracy" title="Accuracy Comparison" />
                </div>
                <div className="chart-container">
                  <ComparisonChart data={baseline} metric="sharpe_ratio" title="Sharpe Ratio Comparison" />
                </div>
                <div className="chart-container">
                  <ComparisonChart data={baseline} metric="total_return" title="Total Return Comparison" />
                </div>
                <div className="chart-container">
                  <ComparisonChart data={baseline} metric="max_drawdown" title="Max Drawdown Comparison" />
                </div>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}





