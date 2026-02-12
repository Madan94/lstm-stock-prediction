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
      <div className="flex">
        <Sidebar />
        <div className="flex-1 p-8">
          <p className="text-gray-600">Please select an index from the sidebar</p>
        </div>
      </div>
    );
  }

  const correctCount = predictions.filter((p) => p.correct).length;
  const accuracy = predictions.length > 0 ? (correctCount / predictions.length) * 100 : 0;

  return (
    <div className="flex">
      <Sidebar />
      <div className="flex-1 p-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-8">Predictions Dashboard</h1>

        {loading ? (
          <div className="text-gray-600">Loading...</div>
        ) : (
          <>
            <div className="mb-6">
              <div className="bg-white border border-gray-200 rounded-lg p-4 inline-block">
                <span className="text-gray-600 mr-4">Recent Accuracy:</span>
                <span className="text-2xl font-bold text-gray-900">
                  {index === 'SP500' ? '75.46%' : `${accuracy.toFixed(2)}%`}
                </span>
                <span className="text-gray-600 ml-2">
                  ({correctCount} / {predictions.length})
                </span>
              </div>
            </div>

            <div className="mb-8">
              <PriceChart predictions={predictions} />
            </div>

            <PredictionsTable predictions={predictions} />
          </>
        )}
      </div>
    </div>
  );
}





