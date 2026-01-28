import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import type { EquityCurvePoint } from '../../types';
import { formatPercent } from '../../utils/formatters';

interface DrawdownChartProps {
  data: EquityCurvePoint[];
}

export default function DrawdownChart({ data }: DrawdownChartProps) {
  // Calculate drawdown
  let peak = data[0]?.value || 0;
  const drawdownData = data.map((point) => {
    if (point.value > peak) {
      peak = point.value;
    }
    const drawdown = ((point.value - peak) / peak) * 100;
    return {
      date: point.date,
      drawdown: drawdown,
    };
  });

  return (
    <div className="bg-white border border-gray-200 rounded-lg p-6 shadow-md">
      <h3 className="text-lg font-semibold mb-4 text-gray-900">Drawdown</h3>
      <ResponsiveContainer width="100%" height={300}>
        <AreaChart data={drawdownData}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
          <XAxis
            dataKey="date"
            stroke="#6b7280"
            tickFormatter={(value) => new Date(value).toLocaleDateString()}
          />
          <YAxis
            stroke="#6b7280"
            tickFormatter={(value) => `${value.toFixed(1)}%`}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: '#ffffff',
              border: '1px solid #e5e7eb',
              borderRadius: '4px',
            }}
            labelFormatter={(value) => new Date(value).toLocaleDateString()}
            formatter={(value: number) => formatPercent(value, 2)}
          />
          <Area
            type="monotone"
            dataKey="drawdown"
            stroke="#ef4444"
            fill="#ef4444"
            fillOpacity={0.3}
            name="Drawdown"
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}



