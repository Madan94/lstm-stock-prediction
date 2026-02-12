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

    Promise.all([
      getEquityCurve(index),
      getBaselineComparison(index),
    ])
      .then(([equityData, baselineData]) => {
        setEquityCurve(equityData);
        setBaseline(baselineData);
        setLoading(false);
      })
      .catch((error) => {
        console.error('Error fetching data:', error);
        setLoading(false);
      });
  }, [index]);

  if (!index) {
    return (
      <div className="flex">
        <Sidebar />
        <div className="flex-1 p-8">
          <p className="text-gray-600">Please select an index from the sidebar</p>
        </div>
      </div>
    );
  }

  // Get Ensemble model metrics (preferred) or Attention LSTM as fallback
  const ensembleModel = baseline.find((m) => 
    m.model_name.toLowerCase().includes('ensemble')
  );
  const attentionLSTM = baseline.find((m) => 
    m.model_name.toLowerCase().includes('attention') || 
    m.model_name === 'attention_lstm'
  );
  const modelMetrics = ensembleModel || attentionLSTM || baseline[0];

  return (
    <div className="flex">
      <Sidebar />
      <div className="flex-1 p-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-8">Strategy Performance</h1>

        {loading ? (
          <div className="text-gray-600">Loading...</div>
        ) : (
          <>
            <div className="mb-4">
              <h2 className="text-xl font-semibold text-gray-900 mb-2">
                {ensembleModel ? 'Ensemble Model Performance' : 'Model Performance'}
              </h2>
              {ensembleModel && (
                <p className="text-sm text-gray-600 mb-4">
                  Combined predictions from Transformer, TCN-LSTM, and Attention-LSTM models
                </p>
              )}
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
              <MetricCard
                title="Sharpe Ratio"
                value={modelMetrics?.sharpe_ratio.toFixed(2) || '0.00'}
              />
              <MetricCard
                title="Total Return"
                value={modelMetrics?.total_return.toFixed(2) || '0.00'}
                subtitle="%"
              />
              <MetricCard
                title="Max Drawdown"
                value={modelMetrics?.max_drawdown.toFixed(2) || '0.00'}
                subtitle="%"
                trend="down"
              />
              <MetricCard
                title="Accuracy"
                value={modelMetrics?.accuracy ? (modelMetrics.accuracy * 100).toFixed(2) : '0.00'}
                subtitle="%"
              />
            </div>

            <div className="mb-8">
              <EquityCurve data={equityCurve} />
            </div>

            <div className="mb-8">
              <DrawdownChart data={equityCurve} />
            </div>
          </>
        )}
      </div>
    </div>
  );
}







