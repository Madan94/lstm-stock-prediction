export default function FloatingNumbers() {
  const numbers = [
    { value: '+12.5%' },
    { value: '↑ 1,234' },
    { value: '98.7%' },
    { value: '+5.2%' },
  ];

  return (
    <div className="absolute inset-0 pointer-events-none overflow-hidden">
      {numbers.map((num, i) => (
        <div
          key={i}
          className="absolute text-black font-bold text-lg opacity-70 animate-float"
          style={{
            left: `${15 + i * 25}%`,
            top: `${20 + (i % 2) * 30}%`,
            animationDelay: `${i * 0.5}s`,
            animationDuration: '3s',
          }}
        >
          {num.value}
        </div>
      ))}
    </div>
  );
}


