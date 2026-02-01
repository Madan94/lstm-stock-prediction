import type { Prediction } from '../../types';
import { formatDate, formatPercent } from '../../utils/formatters';

interface PredictionsTableProps {
  predictions: Prediction[];
}

export default function PredictionsTable({ predictions }: PredictionsTableProps) {
  return (
    <div className="bg-white border border-gray-200 rounded-lg p-6 shadow-md">
      <h3 className="text-lg font-semibold mb-4 text-gray-900">Recent Predictions</h3>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-gray-200">
              <th className="text-left p-3 text-gray-600">Date</th>
              <th className="text-center p-3 text-gray-600">Actual</th>
              <th className="text-center p-3 text-gray-600">Predicted</th>
              <th className="text-center p-3 text-gray-600">Probability</th>
              <th className="text-center p-3 text-gray-600">Correct</th>
            </tr>
          </thead>
          <tbody>
            {predictions.slice(-20).reverse().map((pred, idx) => (
              <tr
                key={idx}
                className="border-b border-gray-200 hover:bg-gray-50"
              >
                <td className="p-3 text-gray-900">{formatDate(pred.date)}</td>
                <td className="p-3 text-center">
                  <span
                    className={`px-2 py-1 rounded ${
                      pred.actual_direction === 1
                        ? 'bg-leaf/20 text-leaf'
                        : 'bg-red-500/20 text-red-500'
                    }`}
                  >
                    {pred.actual_direction === 1 ? '↑' : '↓'}
                  </span>
                </td>
                <td className="p-3 text-center">
                  <span
                    className={`px-2 py-1 rounded ${
                      pred.predicted_direction === 1
                        ? 'bg-leaf/20 text-leaf'
                        : 'bg-red-500/20 text-red-500'
                    }`}
                  >
                    {pred.predicted_direction === 1 ? '↑' : '↓'}
                  </span>
                </td>
                <td className="p-3 text-center text-gray-900">
                  {formatPercent(pred.probability * 100)}
                </td>
                <td className="p-3 text-center">
                  <span
                    className={`px-2 py-1 rounded ${
                      pred.correct
                        ? 'bg-leaf/20 text-leaf'
                        : 'bg-red-500/20 text-red-500'
                    }`}
                  >
                    {pred.correct ? '✓' : '✗'}
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



