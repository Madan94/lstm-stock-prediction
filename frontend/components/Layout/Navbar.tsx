import Link from 'next/link';
import { useRouter } from 'next/router';

export default function Navbar() {
  const router = useRouter();

  const navItems: Array<{ href: string; label: string; icon: string }> = [
    { href: '/', label: 'Dashboard', icon: '📊' },
    { href: '/overview', label: 'Performance', icon: '📈' },
    { href: '/predictions', label: 'Predictions', icon: '🔮' },
    { href: '/attention', label: 'AI Insights', icon: '🧠' },
    { href: '/strategy', label: 'Strategy', icon: '💼' },
    { href: '/comparison', label: 'Comparison', icon: '🆚' },
  ];

  return (
    <nav className="border-b-2 border-black bg-white">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16">
          <div className="flex items-center">
            <div className="flex-shrink-0 flex items-center">
              <div className="flex items-center space-x-3">
                <div className="w-10 h-10 rounded-lg border-2 border-black flex items-center justify-center bg-white">
                  <span className="text-black font-bold text-lg">AI</span>
                </div>
                <div>
                  <h1 className="text-xl font-bold text-black">TradingAI</h1>
                  <p className="text-xs text-black">Black &amp; White</p>
                </div>
              </div>
            </div>
            <div className="hidden sm:ml-10 sm:flex sm:space-x-1">
              {navItems.map((item) => (
                <Link
                  key={item.href}
                  href={item.href}
                  className={`inline-flex items-center px-4 py-2 rounded-lg text-sm font-medium transition-all border-2 ${
                    router.pathname === item.href
                      ? 'bg-black text-white border-black'
                      : 'bg-white text-black border-transparent hover:border-black hover:bg-black hover:text-white'
                  }`}
                >
                  <span className="mr-2">{item.icon}</span>
                  {item.label}
                </Link>
              ))}
            </div>
          </div>
          <div className="flex items-center space-x-4">
            <div className="hidden md:flex items-center space-x-2 px-4 py-2 rounded-lg border-2 border-black bg-white">
              <div className="w-2 h-2 rounded-full bg-black animate-pulse"></div>
              <span className="text-black text-sm font-medium">Live</span>
            </div>
          </div>
        </div>
      </div>
    </nav>
  );
}



