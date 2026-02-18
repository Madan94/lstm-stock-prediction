import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import type { EquityCurvePoint } from '../../types';
import { formatCurrency } from '../../utils/formatters';

interface EquityCurveProps {
  data: EquityCurvePoint[];
}

export default function EquityCurve({ data }: EquityCurveProps) {
  return (
    <div>
      <h3 className="text-xl font-semibold mb-4 text-black">Equity Curve</h3>
      <ResponsiveContainer width="100%" height={400}>
        <LineChart data={data}>
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
            tickFormatter={(value) => formatCurrency(value)}
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
            formatter={(value: number) => formatCurrency(value)}
          />
          <Legend wrapperStyle={{ color: '#000000' }} />
          <Line
            type="monotone"
            dataKey="value"
            stroke="#000000"
            strokeWidth={3}
            name="Portfolio Value"
            dot={false}
            activeDot={{ r: 6, fill: '#000000' }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}



