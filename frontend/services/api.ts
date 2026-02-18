import axios from 'axios';
import type {
  IndexInfo,
  Metrics,
  Prediction,
  EquityCurvePoint,
  AttentionWeights,
  BaselineModel,
} from '../types';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const getIndices = async (): Promise<IndexInfo[]> => {
  const response = await api.get<IndexInfo[]>('/indices');
  return response.data;
};

export const getMetrics = async (index: string): Promise<Metrics> => {
  const response = await api.get<Metrics>(`/metrics/${index}`);
  return response.data;
};

export const getPredictions = async (index: string, limit?: number): Promise<Prediction[]> => {
  const response = await api.get<{ index: string; predictions: Prediction[] }>(
    `/predictions/${index}`,
    { params: { limit } }
  );
  return response.data.predictions;
};

export const getEquityCurve = async (index: string): Promise<EquityCurvePoint[]> => {
  const response = await api.get<{ index: string; equity_curve: EquityCurvePoint[] }>(
    `/equity-curve/${index}`
  );
  return response.data.equity_curve;
};

export const getAttention = async (index: string, limit?: number): Promise<{
  index: string;
  attention: AttentionWeights[];
  lookback_days: number;
}> => {
  const response = await api.get<{
    index: string;
    attention: AttentionWeights[];
    lookback_days: number;
  }>(`/attention/${index}`, { params: { limit } });
  return response.data;
};

export const getBaselineComparison = async (index: string): Promise<BaselineModel[]> => {
  const response = await api.get<{
    index: string;
    models: BaselineModel[];
  }>(`/baseline-comparison/${index}`);
  return response.data.models;
};
