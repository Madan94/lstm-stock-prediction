interface MetricCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  trend?: 'up' | 'down' | 'neutral';
}

export default function MetricCard({ title, value, subtitle, trend }: MetricCardProps) {
  const trendColor = {
    up: 'text-black',
    down: 'text-black',
    neutral: 'text-black',
  }[trend || 'neutral'];

  const cardClass = trend === 'up' ? 'metric-card positive' : trend === 'down' ? 'metric-card negative' : 'metric-card';

  return (
    <div className={cardClass}>
      <p className="text-xs text-black mb-2 uppercase tracking-wider">{title}</p>
      <div className="flex items-baseline">
        <p className={`text-3xl font-bold ${trendColor}`}>
          {typeof value === 'number' ? value.toFixed(2) : value}
        </p>
        {subtitle && (
          <span className="text-sm text-black ml-2 font-medium">{subtitle}</span>
        )}
      </div>
      {trend && (
        <div className="mt-2 flex items-center">
          {trend === 'up' && (
            <span className="text-black text-xs flex items-center font-semibold">
              <svg className="w-3 h-3 mr-1" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M5.293 9.707a1 1 0 010-1.414l4-4a1 1 0 011.414 0l4 4a1 1 0 01-1.414 1.414L11 7.414V15a1 1 0 11-2 0V7.414L6.707 9.707a1 1 0 01-1.414 0z" clipRule="evenodd" />
              </svg>
              Positive
            </span>
          )}
          {trend === 'down' && (
            <span className="text-black text-xs flex items-center font-semibold">
              <svg className="w-3 h-3 mr-1" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M14.707 10.293a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 111.414-1.414L9 12.586V5a1 1 0 012 0v7.586l2.293-2.293a1 1 0 011.414 0z" clipRule="evenodd" />
              </svg>
              Negative
            </span>
          )}
        </div>
      )}
    </div>
  );
}



