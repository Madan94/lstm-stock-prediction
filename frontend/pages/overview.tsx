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

  return (
    <div className="flex">
      <Sidebar />
      <div className="flex-1 p-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-8">Model Overview</h1>

        {loading ? (
          <div className="text-gray-600">Loading...</div>
        ) : (
          <>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
              <MetricCard
                title="Directional Accuracy"
                value={metrics?.accuracy ? (metrics.accuracy * 100).toFixed(2) : '0.00'}
                subtitle="%"
              />
              <MetricCard
                title="Precision"
                value={metrics?.precision ? (metrics.precision * 100).toFixed(2) : '0.00'}
                subtitle="%"
              />
              <MetricCard
                title="Recall"
                value={metrics?.recall ? (metrics.recall * 100).toFixed(2) : '0.00'}
                subtitle="%"
              />
              <MetricCard
                title="F1 Score"
                value={metrics?.f1_score ? (metrics.f1_score * 100).toFixed(2) : '0.00'}
                subtitle="%"
              />
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
              <ComparisonChart
                data={baseline}
                metric="accuracy"
                title="Accuracy Comparison"
              />
              <ComparisonChart
                data={baseline}
                metric="sharpe_ratio"
                title="Sharpe Ratio Comparison"
              />
              <ComparisonChart
                data={baseline}
                metric="total_return"
                title="Total Return Comparison"
              />
              <ComparisonChart
                data={baseline}
                metric="max_drawdown"
                title="Max Drawdown Comparison"
              />
            </div>
          </>
        )}
      </div>
    </div>
  );
}





