import { useState, useEffect } from 'react';
import { useRouter } from 'next/router';
import { getAttention } from '../services/api';
import AttentionHeatmap from '../components/Charts/AttentionHeatmap';
import Sidebar from '../components/Layout/Sidebar';

export default function Attention() {
  const router = useRouter();
  const { index } = router.query;
  const [attention, setAttention] = useState<{ attention: any[]; lookback_days: number } | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!index || typeof index !== 'string') return;

    getAttention(index, 10)
      .then((data) => {
        setAttention(data);
        setLoading(false);
      })
      .catch((error) => {
        console.error('Error fetching attention:', error);
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

  return (
    <div className="flex min-h-screen">
      <Sidebar />
      <div className="flex-1 p-8">
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-black mb-2">AI Attention Insights</h1>
          <p className="text-black">
            Visualize which time periods the model focuses on when making predictions
          </p>
        </div>

        {loading ? (
          <div className="flex items-center justify-center py-20">
            <div className="text-center">
              <div className="inline-block animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-black mb-4"></div>
              <p className="text-black">Loading attention data...</p>
            </div>
          </div>
        ) : attention ? (
          <div className="chart-container">
            <div className="mb-4">
              <div className="flex items-center space-x-2 mb-2">
                <div className="w-3 h-3 rounded bg-black"></div>
                <span className="text-sm text-black">Higher attention weight = darker color</span>
              </div>
            </div>
            <AttentionHeatmap
              attention={attention.attention}
              lookbackDays={attention.lookback_days}
            />
          </div>
        ) : (
          <div className="trading-card text-center py-12">
            <div className="w-16 h-16 rounded-full border-2 border-black bg-white flex items-center justify-center mx-auto mb-4">
              <svg className="w-8 h-8 text-black" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
            </div>
            <p className="text-black">No attention data available</p>
          </div>
        )}
      </div>
    </div>
  );
}



