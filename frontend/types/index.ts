export interface IndexInfo {
  name: string;
  symbol: string;
  display_name: string;
}

export interface Metrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
}

export interface Prediction {
  date: string;
  actual_direction: number;
  predicted_direction: number;
  probability: number;
  correct: boolean;
}

export interface EquityCurvePoint {
  date: string;
  value: number;
}

export interface AttentionWeights {
  date: string;
  weights: number[];
}

export interface BaselineModel {
  model_name: string;
  accuracy: number;
  sharpe_ratio: number;
  total_return: number;
  max_drawdown: number;
}









