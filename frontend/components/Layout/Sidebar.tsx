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
    <div className="w-64 bg-white border-r border-gray-200 p-4 shadow-sm">
      <h2 className="text-lg font-semibold mb-4 text-gray-900">Select Index</h2>
      <div className="space-y-2">
        {indices.map((index) => (
          <button
            key={index.name}
            onClick={() => handleIndexChange(index.name)}
            className={`w-full text-left px-4 py-2 rounded-lg transition-colors ${
              selectedIndex === index.name
                ? 'bg-leaf text-gray-900'
                : 'bg-gray-50 text-gray-700 hover:bg-leaf-light hover:text-gray-900'
            }`}
          >
            {index.display_name}
          </button>
        ))}
      </div>
    </div>
  );
}



