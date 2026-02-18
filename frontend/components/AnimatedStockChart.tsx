export default function AnimatedStockChart() {
  return (
    <div className="w-full h-full absolute inset-0">
      <svg
        viewBox="0 0 400 200"
        className="w-full h-full"
        preserveAspectRatio="none"
      >
        {/* Grid lines */}
        <defs>
          <linearGradient id="lineGradient" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor="#000000" stopOpacity="1" />
            <stop offset="100%" stopColor="#000000" stopOpacity="0.3" />
          </linearGradient>
          <linearGradient id="areaGradient" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor="#000000" stopOpacity="0.25" />
            <stop offset="100%" stopColor="#000000" stopOpacity="0.02" />
          </linearGradient>
        </defs>

        {/* Grid */}
        {[0, 1, 2, 3, 4].map((i) => (
          <line
            key={`grid-h-${i}`}
            x1="0"
            y1={40 + i * 40}
            x2="400"
            y2={40 + i * 40}
            stroke="#000000"
            strokeOpacity="0.12"
            strokeWidth="1"
            strokeDasharray="2,2"
          />
        ))}

        {/* Animated stock line */}
        <path
          d="M 0,150 Q 50,120 100,100 T 200,80 T 300,70 T 400,60"
          fill="none"
          stroke="url(#lineGradient)"
          strokeWidth="3"
          className="animate-pulse"
        >
          <animate
            attributeName="d"
            values="M 0,150 Q 50,120 100,100 T 200,80 T 300,70 T 400,60;M 0,160 Q 50,110 100,90 T 200,75 T 300,65 T 400,55;M 0,150 Q 50,120 100,100 T 200,80 T 300,70 T 400,60"
            dur="4s"
            repeatCount="indefinite"
          />
        </path>

        {/* Area under curve */}
        <path
          d="M 0,150 Q 50,120 100,100 T 200,80 T 300,70 T 400,60 L 400,200 L 0,200 Z"
          fill="url(#areaGradient)"
        >
          <animate
            attributeName="d"
            values="M 0,150 Q 50,120 100,100 T 200,80 T 300,70 T 400,60 L 400,200 L 0,200 Z;M 0,160 Q 50,110 100,90 T 200,75 T 300,65 T 400,55 L 400,200 L 0,200 Z;M 0,150 Q 50,120 100,100 T 200,80 T 300,70 T 400,60 L 400,200 L 0,200 Z"
            dur="4s"
            repeatCount="indefinite"
          />
        </path>

        {/* Data points */}
        {[
          { x: 50, y: 120 },
          { x: 100, y: 100 },
          { x: 150, y: 90 },
          { x: 200, y: 80 },
          { x: 250, y: 75 },
          { x: 300, y: 70 },
          { x: 350, y: 65 },
          { x: 400, y: 60 },
        ].map((point, i) => (
          <circle
            key={`point-${i}`}
            cx={point.x}
            cy={point.y}
            r="4"
            fill="#000000"
            className="animate-pulse"
            style={{ animationDelay: `${i * 0.2}s` }}
          />
        ))}
      </svg>
    </div>
  );
}

