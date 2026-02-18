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
      fill: isAttentionLSTM ? '#000000' : '#ffffff',
    };
  });

  return (
    <div>
      <h3 className="text-lg font-semibold mb-4 text-black">{title}</h3>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" stroke="#000000" strokeOpacity={0.15} />
          <XAxis 
            dataKey="name" 
            stroke="#000000" 
            tick={{ fill: '#000000', fontSize: 12 }}
            angle={-45}
            textAnchor="end"
            height={80}
          />
          <YAxis stroke="#000000" tick={{ fill: '#000000', fontSize: 12 }} />
          <Tooltip
            contentStyle={{
              backgroundColor: '#ffffff',
              border: '2px solid #000000',
              borderRadius: '8px',
              color: '#000000',
            }}
            labelStyle={{ color: '#000000' }}
          />
          <Legend wrapperStyle={{ color: '#000000' }} />
          <Bar dataKey="value" radius={[8, 8, 0, 0]} stroke="#000000" strokeWidth={2}>
            {chartData.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={entry.fill} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}



