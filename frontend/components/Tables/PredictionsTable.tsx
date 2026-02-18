import type { Prediction } from '../../types';
import { formatDate, formatPercent } from '../../utils/formatters';

interface PredictionsTableProps {
  predictions: Prediction[];
}

export default function PredictionsTable({ predictions }: PredictionsTableProps) {
  return (
    <div>
      <h3 className="text-xl font-semibold mb-4 text-black">Recent Predictions</h3>
      <div className="overflow-x-auto trading-table rounded-lg">
        <table className="w-full text-sm">
          <thead>
            <tr>
              <th className="text-left p-4">Date</th>
              <th className="text-center p-4">Actual</th>
              <th className="text-center p-4">Predicted</th>
              <th className="text-center p-4">Probability</th>
              <th className="text-center p-4">Status</th>
            </tr>
          </thead>
          <tbody>
            {predictions.slice(-20).reverse().map((pred, idx) => (
              <tr
                key={idx}
                className="border-t"
                style={{ borderColor: '#000000' }}
              >
                <td className="p-4 text-black font-mono text-xs">{formatDate(pred.date)}</td>
                <td className="p-4 text-center">
                  <span
                    className={`px-3 py-1.5 rounded-lg font-semibold border-2 ${
                      pred.actual_direction === 1
                        ? 'bg-black text-white border-black'
                        : 'bg-white text-black border-black'
                    }`}
                  >
                    {pred.actual_direction === 1 ? '↑ Up' : '↓ Down'}
                  </span>
                </td>
                <td className="p-4 text-center">
                  <span
                    className={`px-3 py-1.5 rounded-lg font-semibold border-2 ${
                      pred.predicted_direction === 1
                        ? 'bg-black text-white border-black'
                        : 'bg-white text-black border-black'
                    }`}
                  >
                    {pred.predicted_direction === 1 ? '↑ Up' : '↓ Down'}
                  </span>
                </td>
                <td className="p-4 text-center">
                  <div className="flex items-center justify-center">
                    <span className="text-black font-semibold">
                      {formatPercent(pred.probability * 100)}
                    </span>
                    <div className="ml-2 w-16 h-2 rounded-full bg-white border-2 border-black overflow-hidden">
                      <div 
                        className="h-full bg-black"
                        style={{ width: `${pred.probability * 100}%` }}
                      ></div>
                    </div>
                  </div>
                </td>
                <td className="p-4 text-center">
                  <span
                    className={`px-3 py-1.5 rounded-lg font-semibold border-2 ${
                      pred.correct
                        ? 'bg-black text-white border-black'
                        : 'bg-white text-black border-black'
                    }`}
                  >
                    {pred.correct ? '✓ Correct' : '✗ Wrong'}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}



