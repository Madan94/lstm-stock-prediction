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

  // Get Attention LSTM metrics
  const attentionLSTM = baseline.find((m) => m.model_name === 'attention_lstm');

  return (
    <div className="flex">
      <Sidebar />
      <div className="flex-1 p-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-8">Strategy Performance</h1>

        {loading ? (
          <div className="text-gray-600">Loading...</div>
        ) : (
          <>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
              <MetricCard
                title="Sharpe Ratio"
                value={attentionLSTM?.sharpe_ratio.toFixed(2) || '0.00'}
              />
              <MetricCard
                title="Total Return"
                value={attentionLSTM?.total_return.toFixed(2) || '0.00'}
                subtitle="%"
              />
              <MetricCard
                title="Max Drawdown"
                value={attentionLSTM?.max_drawdown.toFixed(2) || '0.00'}
                subtitle="%"
                trend="down"
              />
              <MetricCard
                title="Accuracy"
                value={attentionLSTM?.accuracy ? (attentionLSTM.accuracy * 100).toFixed(2) : '0.00'}
                subtitle="%"
              />
            </div>



            <div className="bg-white border border-gray-200 rounded-lg p-6 mb-8 shadow-sm">
              <h3 className="text-lg font-semibold text-gray-900 mb-2">Market Insights</h3>

              {index === 'SP500' && (
                <p className="text-gray-700">
                  Taking the full history of the S&P 500, the overall average S&P 500 value would be <span className="font-bold">$789.83 billion</span>,
                  with an average annual increase of (approximately) 8.52%.
                  <span className="ml-2" role="img" aria-label="pin">📌</span>
                </p>
              )}

              {index === 'NASDAQ' && (
                <p className="text-gray-700">
                  Nasdaq has a market cap or net worth of <span className="font-bold">$48.25 billion</span>.
                  The enterprise value is $57.00 billion. The last earnings date was Thursday, January 29, 2026, before market open.
                  Nasdaq has 571.00 million shares outstanding. The number of shares has decreased by -0.10% in one year.
                  <span className="ml-2" role="img" aria-label="pin">📌</span>
                </p>
              )}

              {index === 'DJI' && (
                <p className="text-gray-700">
                  The Dow Jones Industrial Average consists of 30 prominent companies listed on stock exchanges in the United States.
                  It has historically returned approximately 5-7% annually when adjusted for inflation.
                  <span className="ml-2" role="img" aria-label="pin">📌</span>
                </p>
              )}
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
    </div >
  );
}





