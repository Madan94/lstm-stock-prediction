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
    <div>
      <h3 className="text-xl font-semibold mb-4 text-black">Drawdown</h3>
      <ResponsiveContainer width="100%" height={300}>
        <AreaChart data={drawdownData}>
          <CartesianGrid strokeDasharray="3 3" stroke="#000000" strokeOpacity={0.15} />
          <XAxis
            dataKey="date"
            stroke="#000000"
            tick={{ fill: '#000000', fontSize: 12 }}
            tickFormatter={(value) => new Date(value).toLocaleDateString()}
          />
          <YAxis
            stroke="#000000"
            tick={{ fill: '#000000', fontSize: 12 }}
            tickFormatter={(value) => `${value.toFixed(1)}%`}
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
            formatter={(value: number) => formatPercent(value, 2)}
          />
          <Area
            type="monotone"
            dataKey="drawdown"
            stroke="#000000"
            fill="#000000"
            fillOpacity={0.2}
            name="Drawdown"
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}



