import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Scatter, ScatterChart } from 'recharts';
import type { Prediction } from '../../types';

interface PriceChartProps {
  predictions: Prediction[];
}

export default function PriceChart({ predictions }: PriceChartProps) {
  // Group predictions by date and calculate average probability
  const chartData = predictions.map((pred) => ({
    date: pred.date,
    probability: pred.probability * 100,
    actual: pred.actual_direction,
    predicted: pred.predicted_direction,
    correct: pred.correct ? 1 : 0,
  }));

  return (
    <div className="bg-white border border-gray-200 rounded-lg p-6 shadow-sm">
      <h3 className="text-lg font-semibold mb-4 text-black">Prediction Probability Over Time</h3>
      <ResponsiveContainer width="100%" height={400}>
        <LineChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
          <XAxis
            dataKey="date"
            stroke="#6b7280"
            tickFormatter={(value) => new Date(value).toLocaleDateString()}
          />
          <YAxis stroke="#6b7280" domain={[0, 100]} />
          <Tooltip
            contentStyle={{
              backgroundColor: '#ffffff',
              border: '1px solid #e5e7eb',
              borderRadius: '4px',
              color: '#000000',
            }}
            labelFormatter={(value) => new Date(value).toLocaleDateString()}
          />
          <Legend />
          <Line
            type="monotone"
            dataKey="probability"
            stroke="#22c55e"
            strokeWidth={2}
            name="Probability (%)"
            dot={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}



