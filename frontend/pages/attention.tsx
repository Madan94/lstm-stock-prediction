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
      <div className="flex">
        <Sidebar />
        <div className="flex-1 p-8">
          <p className="text-black/60">Please select an index from the sidebar</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex">
      <Sidebar />
      <div className="flex-1 p-8">
        <h1 className="text-3xl font-bold text-black mb-4">Attention Visualization</h1>
        <p className="text-black/70 mb-8">
          This heatmap shows which days in the lookback window the model focuses on when making predictions.
          Darker green indicates higher attention weight.
        </p>

        {loading ? (
          <div className="text-black/60">Loading...</div>
        ) : attention ? (
          <AttentionHeatmap
            attention={attention.attention}
            lookbackDays={attention.lookback_days}
          />
        ) : (
          <div className="text-black/60">No attention data available</div>
        )}
      </div>
    </div>
  );
}



