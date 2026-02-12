export default function CandlestickChart() {
  const candles = [
    { open: 100, close: 105, high: 108, low: 98, x: 50 },
    { open: 105, close: 103, high: 107, low: 102, x: 100 },
    { open: 103, close: 108, high: 110, low: 102, x: 150 },
    { open: 108, close: 112, high: 115, low: 107, x: 200 },
    { open: 112, close: 110, high: 114, low: 109, x: 250 },
    { open: 110, close: 115, high: 117, low: 109, x: 300 },
    { open: 115, close: 118, high: 120, low: 114, x: 350 },
  ];

  const minPrice = 95;
  const maxPrice = 125;
  const priceRange = maxPrice - minPrice;
  const chartHeight = 150;
  const chartY = 20;

  const getY = (price: number) => {
    return chartY + chartHeight - ((price - minPrice) / priceRange) * chartHeight;
  };

  return (
    <div className="w-full h-48 relative">
      <svg viewBox="0 0 400 200" className="w-full h-full" preserveAspectRatio="none">
        {/* Grid lines */}
        {[0, 1, 2, 3, 4].map((i) => (
          <line
            key={`grid-${i}`}
            x1="0"
            y1={chartY + (i * chartHeight) / 4}
            x2="400"
            y2={chartY + (i * chartHeight) / 4}
            stroke="#e5e7eb"
            strokeWidth="1"
            strokeDasharray="2,2"
          />
        ))}

        {/* Candlesticks */}
        {candles.map((candle, i) => {
          const isGreen = candle.close > candle.open;
          const bodyTop = getY(Math.max(candle.open, candle.close));
          const bodyBottom = getY(Math.min(candle.open, candle.close));
          const bodyHeight = Math.abs(bodyTop - bodyBottom) || 2;
          const wickTop = getY(candle.high);
          const wickBottom = getY(candle.low);

          return (
            <g key={`candle-${i}`} className="animate-fade-in" style={{ animationDelay: `${i * 0.1}s` }}>
              {/* Wick */}
              <line
                x1={candle.x}
                y1={wickTop}
                x2={candle.x}
                y2={wickBottom}
                stroke={isGreen ? '#22c55e' : '#ef4444'}
                strokeWidth="2"
              />
              {/* Body */}
              <rect
                x={candle.x - 8}
                y={bodyTop}
                width="16"
                height={bodyHeight}
                fill={isGreen ? '#22c55e' : '#ef4444'}
                opacity={0.8}
              />
            </g>
          );
        })}

        {/* Price trend line */}
        <path
          d={`M ${candles.map((c, i) => `${c.x},${getY(c.close)}`).join(' L ')}`}
          fill="none"
          stroke="#22c55e"
          strokeWidth="2"
          strokeDasharray="5,5"
          opacity="0.5"
        />
      </svg>
    </div>
  );
}

