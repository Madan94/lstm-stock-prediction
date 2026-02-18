import { useState, useEffect } from 'react';
import { useRouter } from 'next/router';
import { getEquityCurve, getBaselineComparison } from '../services/api';
import type { EquityCurvePoint, BaselineModel } from '../types';
import EquityCurve from '../components/Charts/EquityCurve';
import DrawdownChart from '../components/Charts/DrawdownChart';
import MetricCard from '../components/Cards/MetricCard';
import Sidebar from '../components/Layout/Sidebar';

export default function Strategy() {
  const router = useRouter();
  const { index } = router.query;
  const [equityCurve, setEquityCurve] = useState<EquityCurvePoint[]>([]);
  const [baseline, setBaseline] = useState<BaselineModel[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!index || typeof index !== 'string') return;

    setLoading(true);
    Promise.all([getEquityCurve(index), getBaselineComparison(index)])
      .then(([equityData, baselineData]) => {
        setEquityCurve(equityData);
        setBaseline(baselineData.filter((m) => !m.model_name.toLowerCase().includes('arima')));
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
          <p className="text-black">Please select an index from the sidebar</p>
        </div>
      </div>
    );
  }

  const ensembleModel = baseline.find((m) => m.model_name.toLowerCase().includes('ensemble'));
  const attentionModel = baseline.find((m) => m.model_name.toLowerCase().includes('attention'));
  const modelMetrics = ensembleModel || attentionModel || baseline[0];

  return (
    <div className="flex min-h-screen">
      <Sidebar />
      <div className="flex-1 p-8">
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-black mb-2">Strategy Performance</h1>
          <p className="text-black">Backtesting results and trading metrics</p>
        </div>

        {loading ? (
          <div className="flex items-center justify-center py-20">
            <div className="text-center">
              <div className="inline-block animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-black mb-4"></div>
              <p className="text-black">Loading strategy data...</p>
            </div>
          </div>
        ) : (
          <>
            <div className="mb-8">
              <div className="flex items-center justify-between mb-4">
                <div>
                  <h2 className="text-2xl font-semibold text-black mb-2">
                    {ensembleModel ? 'Ensemble Model Performance' : 'Model Performance'}
                  </h2>
                  {ensembleModel && (
                    <p className="text-sm text-black">
                      Combined predictions from Transformer, TCN-LSTM, and Attention-LSTM models
                    </p>
                  )}
                </div>
                <div className="flex items-center space-x-2 px-3 py-1 rounded-full border-2 border-black bg-white">
                  <div className="w-2 h-2 rounded-full bg-black"></div>
                  <span className="text-black text-xs font-medium">Active</span>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                <MetricCard
                  title="Sharpe Ratio"
                  value={modelMetrics?.sharpe_ratio !== undefined ? modelMetrics.sharpe_ratio.toFixed(2) : '0.00'}
                  trend={modelMetrics?.sharpe_ratio !== undefined && modelMetrics.sharpe_ratio > 0 ? 'up' : 'neutral'}
                />
                <MetricCard
                  title="Total Return"
                  value={modelMetrics?.total_return !== undefined ? modelMetrics.total_return.toFixed(2) : '0.00'}
                  subtitle="%"
                  trend={modelMetrics?.total_return !== undefined && modelMetrics.total_return > 0 ? 'up' : 'down'}
                />
                <MetricCard
                  title="Max Drawdown"
                  value={modelMetrics?.max_drawdown !== undefined ? modelMetrics.max_drawdown.toFixed(2) : '0.00'}
                  subtitle="%"
                  trend="down"
                />
                <MetricCard
                  title="Accuracy"
                  value={modelMetrics?.accuracy ? (modelMetrics.accuracy * 100).toFixed(2) : '0.00'}
                  subtitle="%"
                  trend={modelMetrics?.accuracy && modelMetrics.accuracy > 0.5 ? 'up' : 'neutral'}
                />
              </div>
            </div>

            <div className="trading-card mb-8">
              <h3 className="text-lg font-semibold text-black mb-2">Notes</h3>
              <p className="text-black">
                Strategy metrics are produced from a simple backtest on {index}. This is for research and UI visualization only.
              </p>
            </div>

            <div className="mb-8">
              <div className="chart-container">
                <EquityCurve data={equityCurve} />
              </div>
            </div>

            <div className="mb-8">
              <div className="chart-container">
                <DrawdownChart data={equityCurve} />
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}


