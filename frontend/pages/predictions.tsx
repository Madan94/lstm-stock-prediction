import { useState, useEffect } from 'react';
import { useRouter } from 'next/router';
import { getPredictions } from '../services/api';
import type { Prediction } from '../types';
import PriceChart from '../components/Charts/PriceChart';
import PredictionsTable from '../components/Tables/PredictionsTable';
import Sidebar from '../components/Layout/Sidebar';

export default function Predictions() {
  const router = useRouter();
  const { index } = router.query;
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!index || typeof index !== 'string') return;

    getPredictions(index, 100)
      .then((data) => {
        setPredictions(data);
        setLoading(false);
      })
      .catch((error) => {
        console.error('Error fetching predictions:', error);
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

  const correctCount = predictions.filter((p) => p.correct).length;
  const accuracy = predictions.length > 0 ? (correctCount / predictions.length) * 100 : 0;

  return (
    <div className="flex min-h-screen">
      <Sidebar />
      <div className="flex-1 p-8">
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-black mb-2">Predictions Dashboard</h1>
          <p className="text-black">Real-time AI predictions for market direction</p>
        </div>

        {loading ? (
          <div className="flex items-center justify-center py-20">
            <div className="text-center">
              <div className="inline-block animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-black mb-4"></div>
              <p className="text-black">Loading predictions...</p>
            </div>
          </div>
        ) : (
          <>
            <div className="mb-8">
              <div className="trading-card inline-block">
                <div className="flex items-center space-x-6">
                  <div>
                    <p className="text-sm text-black mb-1 uppercase tracking-wider">Recent Accuracy</p>
                    <div className="flex items-baseline">
                      <span className="text-4xl font-bold text-black">
                        {accuracy.toFixed(2)}%
                      </span>
                      <span className="text-black ml-2 text-sm">
                        ({correctCount} / {predictions.length})
                      </span>
                    </div>
                  </div>
                  <div className="h-12 w-px bg-black"></div>
                  <div>
                    <p className="text-sm text-black mb-1 uppercase tracking-wider">Total Predictions</p>
                    <p className="text-2xl font-bold text-black">{predictions.length}</p>
                  </div>
                </div>
              </div>
            </div>

            <div className="mb-8">
              <div className="chart-container">
                <PriceChart predictions={predictions} />
              </div>
            </div>

            <div className="chart-container">
              <PredictionsTable predictions={predictions} />
            </div>
          </>
        )}
      </div>
    </div>
  );
}





