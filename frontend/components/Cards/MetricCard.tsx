interface MetricCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  trend?: 'up' | 'down' | 'neutral';
}

export default function MetricCard({ title, value, subtitle, trend }: MetricCardProps) {
  const trendColor = {
    up: 'text-leaf',
    down: 'text-red-500',
    neutral: 'text-gray-600',
  }[trend || 'neutral'];

  return (
    <div className="bg-white border border-gray-200 rounded-lg p-6 shadow-md">
      <p className="text-sm text-gray-600 mb-1">{title}</p>
      <p className={`text-3xl font-bold ${trendColor}`}>
        {typeof value === 'number' ? value.toFixed(2) : value}
        {subtitle && <span className="text-lg text-gray-500 ml-2">{subtitle}</span>}
      </p>
    </div>
  );
}



