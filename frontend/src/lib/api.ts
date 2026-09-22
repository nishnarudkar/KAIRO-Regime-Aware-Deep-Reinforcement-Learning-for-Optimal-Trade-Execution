/**
 * KAIRO FastAPI Backend Client
 * The base URL is inlined at build time from NEXT_PUBLIC_API_URL and is used by the browser.
 */

export const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'https://kairo-execution-production.up.railway.app';

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
  n_windows: number;
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
  regime_trajectory?: number[] | null;
  window_start?: number | null;
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
  implementation_shortfall_bps_std: number;
  ci_low: number;
  ci_high: number;
  vs_twap_bps: number | null;
  vs_twap_ci_low: number | null;
  vs_twap_ci_high: number | null;
  execution_cost: number;
  completion_rate: number;
  vwap_slippage_bps: number;
  n_windows: number;
}

export interface BacktestError {
  policy: string;
  detail: string;
}

export interface BacktestResponse {
  backtest_id: string;
  timestamp: string;
  symbol: string;
  side: string;
  quantity: number;
  scenario: string;
  n_windows: number;
  results: BacktestResultItem[];
  errors: BacktestError[];
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
  true_regime?: string | null;
}

export interface BaselineStrategyResponse {
  strategy_id: string;
  name: string;
  description: string;
  parameters: Record<string, unknown>;
}

export interface PairedStat {
  mean: number;
  ci_low: number | null;
  ci_high: number | null;
  p_value: number | null;
  win_rate: number | null;
}

export interface ModelMetadataResponse {
  model_id: string;
  name: string;
  algorithm_class: string;
  regime_aware: boolean;
  status: string;
  trained: boolean;
  state_dim: number;
  net_arch: number[];
  timesteps: number | null;
  trained_at: string | null;
  evaluation: {
    is_bps_mean: number;
    fill_rate_mean: number;
    n_windows: number;
    vs_twap_bps?: PairedStat;
  } | null;
}

export interface ExperimentSummaryResponse {
  experiment_id: string;
  name: string;
  scenarios: string[];
  strategies: string[];
  status: string;
  results_available: boolean;
  seeds: number[] | null;
  train_timesteps: number | null;
}

export interface ComparisonRow {
  rq: string;
  treatment: string;
  control: string;
  scope: string;
  level: 'window' | 'seed';
  n: number;
  mean: number;
  ci_low: number | null;
  ci_high: number | null;
  p_value: number | null;
  win_rate: number | null;
  n_seeds: number | null;
  seed_std: number | null;
  seeds_favouring: number | null;
}

export interface SummaryRow {
  strategy: string;
  mean: number;
  std: number | null;
  min: number;
  max: number;
  n_runs: number;
}

export interface ExperimentResults {
  config: {
    seeds?: number[];
    scenarios?: string[];
    train_timesteps?: number | null;
    n_bars?: number;
    horizon_steps?: number;
    order_participation?: number | null;
    note?: string;
    source?: string;
  };
  summary: SummaryRow[];
  comparisons: ComparisonRow[];
  hmm_validation: { mean_ari?: number | null; mean_accuracy?: number | null; n_runs?: number };
}

/** Extract a readable message from a FastAPI error body. */
async function errorMessage(res: Response, fallback: string): Promise<string> {
  try {
    const body = await res.json();
    const detail = body?.detail;
    if (typeof detail === 'string') return detail;
    if (Array.isArray(detail) && detail.length) {
      return detail.map((d: { loc?: unknown[]; msg?: string }) => `${(d.loc ?? []).slice(1).join('.')}: ${d.msg}`).join('; ');
    }
  } catch {
    /* fall through */
  }
  return fallback;
}

async function postJson<T>(path: string, payload: unknown, fallback: string): Promise<T> {
  const res = await fetch(`${API_BASE_URL}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(await errorMessage(res, fallback));
  return (await res.json()) as T;
}

async function getJson<T>(path: string, fallback: string): Promise<T> {
  const res = await fetch(`${API_BASE_URL}${path}`, { cache: 'no-store' });
  if (!res.ok) throw new Error(await errorMessage(res, fallback));
  return (await res.json()) as T;
}

export async function checkBackendHealth(): Promise<{ status: string; service: string }> {
  try {
    return await getJson('/health', 'Healthcheck failed');
  } catch {
    throw new Error('Backend offline');
  }
}

export const simulateExecution = (payload: ExecutionSimulateRequest) =>
  postJson<ExecutionRecord>('/api/execution/simulate', payload, 'Simulation request failed');

export const runBacktest = (payload: BacktestRequest) =>
  postJson<BacktestResponse>('/api/execution/backtest', payload, 'Backtest request failed');

export const getExecutionRecord = (id: string) =>
  getJson<ExecutionRecord>(`/api/execution/${id}`, `Execution ${id} not found`);

export const getExecutionTrajectory = (id: string) =>
  getJson<ExecutionTrajectory>(`/api/execution/${id}/trajectory`, `Trajectory for ${id} not found`);

export const getCurrentRegime = (symbol = 'AAPL', scenario = 'normal', seed = 42) =>
  getJson<CurrentRegimeResponse>(
    `/api/regime/current?symbol=${encodeURIComponent(symbol)}&scenario=${encodeURIComponent(scenario)}&seed=${seed}`,
    'Regime detection request failed',
  );

export const getBaselines = () => getJson<BaselineStrategyResponse[]>('/api/baselines', 'Failed to fetch baselines');

export const getModels = () => getJson<ModelMetadataResponse[]>('/api/models', 'Failed to fetch models');

export const getExperiments = () => getJson<ExperimentSummaryResponse[]>('/api/experiments', 'Failed to fetch experiments');

export type ResultSuite = 'default' | 'long_horizon' | 'real_data';

export const getExperimentResults = (suite: ResultSuite = 'default') =>
  getJson<ExperimentResults>(`/api/experiments/results?suite=${suite}`, 'No experiment results');
