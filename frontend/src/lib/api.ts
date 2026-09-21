/**
 * KAIRO FastAPI Backend Client
 * Connects to http://127.0.0.1:8000
 */

export const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000';

export interface ExecutionSimulateRequest {
  symbol: string;
  side: string;
  quantity: number;
  horizon_steps: number;
  policy: string;
  scenario: string;
  seed: number;
}

export interface BacktestRequest {
  symbol: string;
  side: string;
  quantity: number;
  horizon_steps: number;
  policies: string[];
  scenario: string;
  seed: number;
}

export interface ExecutionMetrics {
  implementation_shortfall: number;
  implementation_shortfall_bps: number;
  execution_cost: number;
  market_impact_cost: number;
  total_transaction_fees: number;
  terminal_penalty: number;
  completion_rate: number;
  average_execution_price: number;
  arrival_price: number;
  market_vwap_price: number;
  vwap_slippage_bps: number;
  action_counts: Record<string, number>;
}

export interface ExecutionTrajectory {
  execution_id: string;
  inventory_trajectory: number[];
  action_trajectory: number[];
  price_trajectory: number[];
  regime_trajectory?: number[];
}

export interface ExecutionRecord {
  execution_id: string;
  timestamp: string;
  symbol: string;
  side: string;
  target_inventory: number;
  executed_inventory: number;
  remaining_inventory: number;
  policy: string;
  scenario: string;
  status: string;
  metrics: ExecutionMetrics;
}

export interface BacktestResultItem {
  policy: string;
  implementation_shortfall_bps: number;
  execution_cost: number;
  completion_rate: number;
  vwap_slippage_bps: number;
}

export interface BacktestResponse {
  backtest_id: string;
  timestamp: string;
  symbol: string;
  side: string;
  quantity: number;
  scenario: string;
  results: BacktestResultItem[];
}

export interface CurrentRegimeResponse {
  symbol: string;
  timestamp: string;
  current_price: number;
  spread: number;
  volatility: number;
  regime_id: number;
  regime_label: string;
  regime_probabilities: Record<string, number>;
}

export interface BaselineStrategyResponse {
  strategy_id: string;
  name: string;
  description: string;
  parameters: Record<string, any>;
}

export interface ModelMetadataResponse {
  model_id: string;
  name: string;
  algorithm_class: string;
  regime_aware: boolean;
  status: string;
  state_dim: number;
  net_arch: number[];
}

export interface ExperimentSummaryResponse {
  experiment_id: string;
  name: string;
  scenarios: string[];
  strategies: string[];
  status: string;
}

export async function checkBackendHealth(): Promise<{ status: string; service: string }> {
  try {
    const res = await fetch(`${API_BASE_URL}/health`, { cache: 'no-store' });
    if (!res.ok) throw new Error('Healthcheck failed');
    return await res.json();
  } catch (err) {
    throw new Error('Backend offline');
  }
}

export async function simulateExecution(payload: ExecutionSimulateRequest): Promise<ExecutionRecord> {
  const res = await fetch(`${API_BASE_URL}/api/execution/simulate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({ detail: 'Simulation request failed' }));
    throw new Error(errorData.detail || 'Simulation request failed');
  }
  return await res.json();
}

export async function runBacktest(payload: BacktestRequest): Promise<BacktestResponse> {
  const res = await fetch(`${API_BASE_URL}/api/execution/backtest`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({ detail: 'Backtest request failed' }));
    throw new Error(errorData.detail || 'Backtest request failed');
  }
  return await res.json();
}

export async function getExecutionRecord(id: string): Promise<ExecutionRecord> {
  const res = await fetch(`${API_BASE_URL}/api/execution/${id}`);
  if (!res.ok) throw new Error(`Execution ${id} not found`);
  return await res.json();
}

export async function getExecutionTrajectory(id: string): Promise<ExecutionTrajectory> {
  const res = await fetch(`${API_BASE_URL}/api/execution/${id}/trajectory`);
  if (!res.ok) throw new Error(`Trajectory for ${id} not found`);
  return await res.json();
}

export async function getCurrentRegime(symbol = 'AAPL', scenario = 'normal'): Promise<CurrentRegimeResponse> {
  const res = await fetch(`${API_BASE_URL}/api/regime/current?symbol=${symbol}&scenario=${scenario}`);
  if (!res.ok) throw new Error('Regime detection request failed');
  return await res.json();
}

export async function getBaselines(): Promise<BaselineStrategyResponse[]> {
  const res = await fetch(`${API_BASE_URL}/api/baselines`);
  if (!res.ok) throw new Error('Failed to fetch baselines');
  return await res.json();
}

export async function getModels(): Promise<ModelMetadataResponse[]> {
  const res = await fetch(`${API_BASE_URL}/api/models`);
  if (!res.ok) throw new Error('Failed to fetch models');
  return await res.json();
}

export async function getExperiments(): Promise<ExperimentSummaryResponse[]> {
  const res = await fetch(`${API_BASE_URL}/api/experiments`);
  if (!res.ok) throw new Error('Failed to fetch experiments');
  return await res.json();
}
