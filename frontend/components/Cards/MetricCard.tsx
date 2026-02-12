interface MetricCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  trend?: 'up' | 'down' | 'neutral';
}

export default function MetricCard({ title, value, subtitle, trend }: MetricCardProps) {
  const trendColor = {
    up: 'text-trading-green',
    down: 'text-red-500',
    neutral: 'text-black',
  }[trend || 'neutral'];

  return (
    <div className="bg-white border border-gray-200 rounded-lg p-6 shadow-sm">
      <p className="text-sm text-black/60 mb-1">{title}</p>
      <p className={`text-3xl font-bold ${trendColor}`}>
        {typeof value === 'number' ? value.toFixed(2) : value}
        {subtitle && <span className="text-lg text-black/50 ml-2">{subtitle}</span>}
      </p>
    </div>
  );
}



