import Link from 'next/link';
import { useRouter } from 'next/router';

export default function Navbar() {
  const router = useRouter();

  const navItems = [
    { href: '/', label: 'Home' },
    { href: '/overview', label: 'Overview' },
    { href: '/predictions', label: 'Predictions' },
    { href: '/attention', label: 'Attention' },
    { href: '/strategy', label: 'Strategy' },
    { href: '/comparison', label: 'Comparison' },
  ];

  return (
    <nav className="bg-white border-b border-gray-200">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16">
          <div className="flex space-x-32">
            <div className="flex-shrink-0 flex items-center">
              <a href=""><h1 className="text-xl font-bold text-black">LSTM Stock Prediction</h1></a>
            </div>
            <div className="hidden sm:ml-6 sm:flex sm:space-x-16">
              {navItems.map((item) => (
                <Link
                  key={item.href}
                  href={item.href}
                  className={`inline-flex items-center px-1 pt-1 border-b-2 text-md font-medium transition-colors ${
                    router.pathname === item.href
                      ? 'border-trading-green text-trading-green text-bold'
                      : 'border-transparent text-black/70 hover:text-trading-green text-bold hover:border-trading-green/50'
                  }`}
                >
                  {item.label}
                </Link>
              ))}
            </div>
          </div>
        </div>
      </div>
    </nav>
  );
}



