import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import type { BaselineModel } from '../../types';

interface ComparisonChartProps {
  data: BaselineModel[];
  metric: 'accuracy' | 'sharpe_ratio' | 'total_return' | 'max_drawdown';
  title: string;
}

export default function ComparisonChart({ data, metric, title }: ComparisonChartProps) {
  const chartData = data.map((model) => ({
    name: model.model_name.replace('_', ' ').toUpperCase(),
    value: model[metric],
  }));

  return (
    <div className="bg-white border border-gray-200 rounded-lg p-6 shadow-md">
      <h3 className="text-lg font-semibold mb-4 text-gray-900">{title}</h3>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
          <XAxis dataKey="name" stroke="#6b7280" />
          <YAxis stroke="#6b7280" />
          <Tooltip
            contentStyle={{
              backgroundColor: '#ffffff',
              border: '1px solid #e5e7eb',
              borderRadius: '4px',
            }}
          />
          <Legend />
          <Bar dataKey="value" fill="#22c55e" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}



