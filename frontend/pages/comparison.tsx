import { useState, useEffect } from 'react';
import { useRouter } from 'next/router';
import { getBaselineComparison } from '../services/api';
import type { BaselineModel } from '../types';
import Sidebar from '../components/Layout/Sidebar';

export default function Comparison() {
  const router = useRouter();
  const { index } = router.query;
  const [baseline, setBaseline] = useState<BaselineModel[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!index || typeof index !== 'string') return;

    getBaselineComparison(index)
      .then((data) => {
        // Filter out ARIMA
        const filteredData = data.filter((model) => 
          !model.model_name.toLowerCase().includes('arima')
        );
        setBaseline(filteredData);
        setLoading(false);
      })
      .catch((error) => {
        console.error('Error fetching comparison data:', error);
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

  // Format model name for display
  const formatModelName = (name: string) => {
    return name
      .split('_')
      .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  };

  // Sort models - put attention_lstm first, then others
  const sortedModels = [...baseline].sort((a, b) => {
    if (a.model_name === 'attention_lstm') return -1;
    if (b.model_name === 'attention_lstm') return 1;
    return b.accuracy - a.accuracy;
  });

  return (
    <div className="flex min-h-screen">
      <Sidebar />
      <div className="flex-1 p-8">
        <h1 className="text-3xl font-bold text-black mb-2">Model Accuracy Comparison</h1>
        <p className="text-black mb-8">Compare accuracy of different models</p>

        {loading ? (
          <div className="text-black">Loading comparison data...</div>
        ) : (
          <div className="max-w-4xl">
            {/* Simple Model Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {sortedModels.slice(0, 5).map((model, idx) => {
                const isOurModel = model.model_name === 'attention_lstm';
                return (
                  <div
                    key={idx}
                    className={`rounded-lg p-6 border-2 border-black ${
                      isOurModel ? 'bg-black text-white' : 'bg-white text-black'
                    }`}
                  >
                    <div className="flex items-center justify-between mb-3">
                      <h3 className={`text-lg font-semibold ${isOurModel ? 'text-white' : 'text-black'}`}>
                        {formatModelName(model.model_name)}
                      </h3>
                      {isOurModel && (
                        <span className="px-2 py-1 bg-white text-black border-2 border-white rounded text-xs font-medium">
                          Our Model
                        </span>
                      )}
                    </div>
                    <div className="mt-4">
                      <p className={`text-4xl font-bold ${isOurModel ? 'text-white' : 'text-black'}`}>
                        {(model.accuracy * 100).toFixed(2)}%
                      </p>
                      <p className={`text-sm mt-2 ${isOurModel ? 'text-white' : 'text-black'}`}>
                        Accuracy
                      </p>
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Simple Bar Visualization */}
            <div className="mt-12 bg-white border-2 border-black rounded-lg p-8">
              <h3 className="text-xl font-semibold text-black mb-6">Accuracy Comparison</h3>
              <div className="space-y-4">
                {sortedModels.slice(0, 5).map((model, idx) => {
                  const isOurModel = model.model_name === 'attention_lstm';
                  const maxAccuracy = Math.max(...sortedModels.slice(0, 5).map((m) => m.accuracy));
                  const percentage = (model.accuracy / maxAccuracy) * 100;

                  return (
                    <div key={idx} className="space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="font-medium text-black">
                          {formatModelName(model.model_name)}
                          {isOurModel && (
                            <span className="ml-2 text-xs text-black font-semibold">(Our Model)</span>
                          )}
                        </span>
                        <span className="font-bold text-black">
                          {(model.accuracy * 100).toFixed(2)}%
                        </span>
                      </div>
                      <div className="w-full bg-white border-2 border-black rounded-full h-4 overflow-hidden">
                        <div
                          className={`h-full rounded-full transition-all duration-500 ${isOurModel ? 'bg-black' : 'bg-white border-r-2 border-black'}`}
                          style={{ width: `${percentage}%` }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

