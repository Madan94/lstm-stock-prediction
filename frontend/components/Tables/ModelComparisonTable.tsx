import type { BaselineModel } from '../../types';

interface ModelComparisonTableProps {
  models: BaselineModel[];
}

// Type for numeric metric keys only (excludes 'model_name')
type NumericMetric = 'accuracy' | 'sharpe_ratio' | 'total_return' | 'max_drawdown';

export default function ModelComparisonTable({ models }: ModelComparisonTableProps) {
  // Filter out ARIMA
  const filteredModels = models.filter((model) => 
    !model.model_name.toLowerCase().includes('arima')
  );

  // Format model name for display
  const formatModelName = (name: string) => {
    return name
      .split('_')
      .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  };

  // Find the best value for each metric
  const getBestValue = (metric: NumericMetric) => {
    if (metric === 'max_drawdown') {
      // For drawdown, lower is better
      return Math.min(...filteredModels.map((m) => m[metric]));
    }
    // For others, higher is better
    return Math.max(...filteredModels.map((m) => m[metric]));
  };

  const bestAccuracy = getBestValue('accuracy');
  const bestSharpe = getBestValue('sharpe_ratio');
  const bestReturn = getBestValue('total_return');
  const bestDrawdown = getBestValue('max_drawdown');

  const isBest = (model: BaselineModel, metric: keyof BaselineModel) => {
    if (metric === 'max_drawdown') {
      return model[metric] === bestDrawdown;
    }
    if (metric === 'accuracy') {
      return model[metric] === bestAccuracy;
    }
    if (metric === 'sharpe_ratio') {
      return model[metric] === bestSharpe;
    }
    if (metric === 'total_return') {
      return model[metric] === bestReturn;
    }
    return false;
  };

  return (
    <div className="bg-white border border-gray-200 rounded-lg p-6 shadow-sm">
      <h3 className="text-lg font-semibold text-black mb-4">Detailed Model Comparison</h3>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-gray-200">
              <th className="text-left p-3 text-black/60 font-semibold">Model</th>
              <th className="text-center p-3 text-black/60 font-semibold">Accuracy (%)</th>
              <th className="text-center p-3 text-black/60 font-semibold">Sharpe Ratio</th>
              <th className="text-center p-3 text-black/60 font-semibold">Total Return (%)</th>
              <th className="text-center p-3 text-black/60 font-semibold">Max Drawdown (%)</th>
            </tr>
          </thead>
          <tbody>
            {filteredModels.map((model, idx) => (
              <tr
                key={idx}
                className={`border-b border-gray-200 hover:bg-gray-50 ${
                  model.model_name === 'attention_lstm' ? 'bg-trading-green/5' : ''
                }`}
              >
                <td className="p-3">
                  <span className="font-semibold text-black">
                    {formatModelName(model.model_name)}
                  </span>
                  {model.model_name === 'attention_lstm' && (
                    <span className="ml-2 px-2 py-0.5 bg-trading-green/20 text-trading-green rounded text-xs font-medium">
                      Our Model
                    </span>
                  )}
                </td>
                <td className="p-3 text-center">
                  <span
                    className={`font-semibold ${
                      isBest(model, 'accuracy')
                        ? 'text-trading-green'
                        : 'text-black'
                    }`}
                  >
                    {(model.accuracy * 100).toFixed(2)}%
                  </span>
                  {isBest(model, 'accuracy') && (
                    <span className="ml-1 text-trading-green">✓</span>
                  )}
                </td>
                <td className="p-3 text-center">
                  <span
                    className={`font-semibold ${
                      isBest(model, 'sharpe_ratio')
                        ? 'text-trading-green'
                        : 'text-black'
                    }`}
                  >
                    {model.sharpe_ratio.toFixed(2)}
                  </span>
                  {isBest(model, 'sharpe_ratio') && (
                    <span className="ml-1 text-trading-green">✓</span>
                  )}
                </td>
                <td className="p-3 text-center">
                  <span
                    className={`font-semibold ${
                      isBest(model, 'total_return')
                        ? 'text-trading-green'
                        : 'text-black'
                    }`}
                  >
                    {model.total_return.toFixed(2)}%
                  </span>
                  {isBest(model, 'total_return') && (
                    <span className="ml-1 text-trading-green">✓</span>
                  )}
                </td>
                <td className="p-3 text-center">
                  <span
                    className={`font-semibold ${
                      isBest(model, 'max_drawdown')
                        ? 'text-trading-green'
                        : 'text-black'
                    }`}
                  >
                    {model.max_drawdown.toFixed(2)}%
                  </span>
                  {isBest(model, 'max_drawdown') && (
                    <span className="ml-1 text-trading-green">✓</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="mt-4 text-xs text-black/60">
        <p>✓ indicates the best performance for each metric</p>
      </div>
    </div>
  );
}

