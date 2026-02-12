export default function FinancialIndicators() {
  return (
    <div className="flex items-center justify-center gap-8 flex-wrap">
      {/* Trending Up Icon */}
      <div className="relative">
        <svg
          width="60"
          height="60"
          viewBox="0 0 24 24"
          fill="none"
          className="text-trading-green animate-bounce"
          style={{ animationDelay: '0s', animationDuration: '2s' }}
        >
          <path
            d="M7 17L17 7M17 7H7M17 7V17"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
        <div className="absolute -top-2 -right-2 w-4 h-4 bg-trading-green rounded-full animate-ping" />
      </div>

      {/* Chart Icon */}
      <div className="relative">
        <svg
          width="60"
          height="60"
          viewBox="0 0 24 24"
          fill="none"
          className="text-black animate-pulse"
          style={{ animationDelay: '0.5s' }}
        >
          <path
            d="M3 3V21H21"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
          />
          <path
            d="M7 16L12 11L16 15L21 10"
            stroke="#22c55e"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      </div>

      {/* Dollar Sign */}
      <div className="relative">
        <svg
          width="60"
          height="60"
          viewBox="0 0 24 24"
          fill="none"
          className="text-trading-green"
        >
          <path
            d="M12 2V22M17 5H9.5C8.57174 5 7.6815 5.36875 7.02513 6.02513C6.36875 6.6815 6 7.57174 6 8.5C6 9.42826 6.36875 10.3185 7.02513 10.9749C7.6815 11.6313 8.57174 12 9.5 12H14.5C15.4283 12 16.3185 12.3687 16.9749 13.0251C17.6313 13.6815 18 14.5717 18 15.5C18 16.4283 17.6313 17.3185 16.9749 17.9749C16.3185 18.6313 15.4283 19 14.5 19H6"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            className="animate-pulse"
          />
        </svg>
      </div>

      {/* Target/Bullseye */}
      <div className="relative">
        <svg
          width="60"
          height="60"
          viewBox="0 0 24 24"
          fill="none"
          className="text-black"
        >
          <circle
            cx="12"
            cy="12"
            r="10"
            stroke="currentColor"
            strokeWidth="2"
            className="animate-spin"
            style={{ animationDuration: '3s' }}
          />
          <circle
            cx="12"
            cy="12"
            r="6"
            stroke="#22c55e"
            strokeWidth="2"
            className="animate-spin"
            style={{ animationDuration: '2s', animationDirection: 'reverse' }}
          />
          <circle
            cx="12"
            cy="12"
            r="2"
            fill="#22c55e"
            className="animate-pulse"
          />
        </svg>
      </div>
    </div>
  );
}


