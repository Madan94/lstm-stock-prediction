import { useState } from 'react';
import type { AttentionWeights } from '../../types';

interface AttentionHeatmapProps {
  attention: AttentionWeights[];
  lookbackDays: number;
}

export default function AttentionHeatmap({ attention, lookbackDays }: AttentionHeatmapProps) {
  const [selectedDate, setSelectedDate] = useState<string | null>(null);

  if (attention.length === 0) {
    return (
      <div>
        <p className="text-black">No attention data available</p>
      </div>
    );
  }

  // Get the most recent attention weights
  const recentAttention = attention.slice(-10); // Show last 10 predictions

  // Create heatmap data
  const maxWeight = Math.max(...recentAttention.flatMap((a) => a.weights));

  return (
    <div>
      <h3 className="text-xl font-semibold mb-4 text-black">Attention Weights Heatmap</h3>
      <p className="text-sm text-black mb-4">
        Showing attention weights for the last {recentAttention.length} predictions
      </p>
      <div className="overflow-x-auto">
        <table className="w-full text-sm trading-table">
          <thead>
            <tr>
              <th className="text-left p-2">Date</th>
              {Array.from({ length: Math.min(lookbackDays, 20) }, (_, i) => (
                <th key={i} className="text-center p-2 text-xs">
                  Day -{lookbackDays - i}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {recentAttention.map((item) => {
              const weights = item.weights.slice(-Math.min(lookbackDays, 20));
              return (
                <tr
                  key={item.date}
                  className={`border-t cursor-pointer ${
                    selectedDate === item.date ? 'bg-black/5' : ''
                  }`}
                  style={{ borderColor: '#000000' }}
                  onClick={() => setSelectedDate(item.date === selectedDate ? null : item.date)}
                >
                  <td className="p-2 text-black text-xs">
                    {new Date(item.date).toLocaleDateString()}
                  </td>
                  {weights.map((weight, idx) => {
                    const intensity = (weight / maxWeight) * 100;
                    const opacity = intensity / 100;
                    return (
                      <td
                        key={idx}
                        className="p-1 text-center"
                        style={{
                          backgroundColor: `rgba(0, 0, 0, ${opacity * 0.3})`,
                        }}
                        title={`Weight: ${weight.toFixed(4)}`}
                      >
                        <span className="text-xs text-black font-semibold">
                          {weight.toFixed(2)}
                        </span>
                      </td>
                    );
                  })}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <div className="mt-4 flex items-center justify-between text-xs text-black">
        <span>Darker = higher attention weight</span>
        <span>Click a row to highlight</span>
      </div>
    </div>
  );
}



