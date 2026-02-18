import { useState, useEffect } from 'react';
import { getIndices } from '../../services/api';
import type { IndexInfo } from '../../types';
import { useRouter } from 'next/router';

export default function Sidebar() {
  const [indices, setIndices] = useState<IndexInfo[]>([]);
  const [selectedIndex, setSelectedIndex] = useState<string>('');
  const router = useRouter();

  useEffect(() => {
    getIndices()
      .then(setIndices)
      .catch(console.error);
  }, []);

  useEffect(() => {
    const index = router.query.index as string;
    if (index) {
      setSelectedIndex(index);
    }
  }, [router.query]);

  const handleIndexChange = (index: string) => {
    setSelectedIndex(index);
    router.push({
      pathname: router.pathname,
      query: { ...router.query, index },
    });
  };

  return (
    <div className="w-64 border-r-2 border-black p-4 bg-white">
      <h2 className="text-sm font-semibold mb-4 uppercase tracking-wider text-black">Markets</h2>
      <div className="space-y-2">
        {indices.map((index) => (
          <button
            key={index.name}
            onClick={() => handleIndexChange(index.name)}
            className={`w-full text-left px-4 py-3 rounded-lg transition-all border-2 ${
              selectedIndex === index.name
                ? 'bg-black text-white border-black'
                : 'bg-white text-black border-transparent hover:border-black hover:bg-black hover:text-white'
            }`}
          >
            <div className="flex items-center justify-between">
              <div>
                <div className="font-semibold">{index.display_name}</div>
                <div className="text-xs opacity-70">{index.symbol}</div>
              </div>
              {selectedIndex === index.name && (
                <div className="w-2 h-2 rounded-full bg-white"></div>
              )}
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}



