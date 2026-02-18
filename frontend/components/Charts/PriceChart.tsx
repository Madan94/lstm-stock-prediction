import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
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
    <div>
      <h3 className="text-xl font-semibold mb-4 text-black">Prediction Probability Over Time</h3>
      <ResponsiveContainer width="100%" height={400}>
        <LineChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" stroke="#000000" strokeOpacity={0.15} />
          <XAxis
            dataKey="date"
            stroke="#000000"
            tick={{ fill: '#000000', fontSize: 12 }}
            tickFormatter={(value) => new Date(value).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}
          />
          <YAxis 
            stroke="#000000" 
            domain={[0, 100]} 
            tick={{ fill: '#000000', fontSize: 12 }}
            label={{ value: 'Probability (%)', angle: -90, position: 'insideLeft', fill: '#000000' }}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: '#ffffff',
              border: '2px solid #000000',
              borderRadius: '8px',
              color: '#000000',
            }}
            labelStyle={{ color: '#000000' }}
            labelFormatter={(value) => new Date(value).toLocaleDateString()}
          />
          <Legend wrapperStyle={{ color: '#000000' }} />
          <Line
            type="monotone"
            dataKey="probability"
            stroke="#000000"
            strokeWidth={3}
            name="Probability (%)"
            dot={false}
            activeDot={{ r: 6, fill: '#000000' }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}



