import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell } from 'recharts';
import type { BaselineModel } from '../../types';

interface ComparisonChartProps {
  data: BaselineModel[];
  metric: 'accuracy' | 'sharpe_ratio' | 'total_return' | 'max_drawdown';
  title: string;
}

export default function ComparisonChart({ data, metric, title }: ComparisonChartProps) {
  // Filter out ARIMA
  const filteredData = data.filter((model) => 
    !model.model_name.toLowerCase().includes('arima')
  );
  
  const chartData = filteredData.map((model) => {
    const isAttentionLSTM = model.model_name === 'attention_lstm';
    return {
      name: model.model_name
        .split('_')
        .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' '),
      value: model[metric],
      fill: isAttentionLSTM ? '#22c55e' : '#6b7280',
    };
  });

  return (
    <div className="bg-white border border-gray-200 rounded-lg p-6 shadow-sm">
      <h3 className="text-lg font-semibold mb-4 text-black">{title}</h3>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
          <XAxis
            dataKey="name"
            stroke="#6b7280"
            angle={-45}
            textAnchor="end"
            height={80}
          />
          <YAxis stroke="#6b7280" />
          <Tooltip
            contentStyle={{
              backgroundColor: '#ffffff',
              border: '1px solid #e5e7eb',
              borderRadius: '4px',
              color: '#000000',
            }}
          />
          <Legend />
          <Bar dataKey="value">
            {chartData.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={entry.fill} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}



