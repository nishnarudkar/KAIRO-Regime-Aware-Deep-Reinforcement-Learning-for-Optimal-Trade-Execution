'use client';

import React, { useState } from 'react';
import { Sliders, Play, Cpu, ShieldAlert, Sparkles, TrendingUp } from 'lucide-react';
import { simulateExecution, ExecutionRecord } from '../lib/api';

interface NewExecutionTabProps {
  onExecutionCreated: (record: ExecutionRecord) => void;
}

export function NewExecutionTab({ onExecutionCreated }: NewExecutionTabProps) {
  const [symbol, setSymbol] = useState('AAPL');
  const [side, setSide] = useState('BUY');
  const [quantity, setQuantity] = useState(100000);
  const [horizonSteps, setHorizonSteps] = useState(30);
  const [policy, setPolicy] = useState('Regime-Aware DQN');
  const [scenario, setScenario] = useState('normal');
  const [seed, setSeed] = useState(42);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const record = await simulateExecution({
        symbol,
        side,
        quantity: Number(quantity),
        horizon_steps: Number(horizonSteps),
        policy,
        scenario,
        seed: Number(seed),
      });
      onExecutionCreated(record);
    } catch (err: any) {
      setError(err.message || 'Simulation execution failed');
    } finally {
      setLoading(false);
    }
  };

  const policies = [
    { id: 'TWAP', name: 'TWAP Baseline', desc: 'Uniform time-decay baseline', tag: 'Baseline' },
    { id: 'VWAP', name: 'VWAP Baseline', desc: 'Volume-weighted profile fill', tag: 'Baseline' },
    { id: 'POV', name: 'POV Baseline (10%)', desc: 'Fixed 10% volume participation rate', tag: 'Baseline' },
    { id: 'DQN', name: 'Standard DQN', desc: 'Off-policy value iteration (7-dim state)', tag: 'RL Agent' },
    { id: 'Regime-Aware DQN', name: 'Regime-Aware DQN', desc: 'Value iteration + HMM causal regime state (12-dim)', tag: 'Recommended' },
    { id: 'PPO', name: 'Standard PPO', desc: 'On-policy actor-critic gradient (7-dim state)', tag: 'RL Agent' },
    { id: 'Regime-Aware PPO', name: 'Regime-Aware PPO', desc: 'Policy gradient + HMM causal regime state (12-dim)', tag: 'RL Agent' },
  ];

  const scenarios = [
    { id: 'normal', name: 'Normal Market', desc: 'Standard liquid trading conditions' },
    { id: 'high_volatility', name: 'High Volatility', desc: 'Elevated price volatility & wide spreads' },
    { id: 'low_liquidity', name: 'Low Liquidity', desc: 'Thin order book depth & high impact' },
    { id: 'stress', name: 'Market Stress', desc: 'Extreme volatility + illiquidity compound' },
    { id: 'regime_transition', name: 'Regime Transition', desc: 'Mid-horizon volatility regime shift' },
    { id: 'liquidity_shock', name: 'Liquidity Shock', desc: 'Sudden order book liquidity collapse' },
  ];

  return (
    <div className="max-w-5xl mx-auto space-y-6">
      <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6 shadow-xl backdrop-blur-sm">
        <div className="flex items-center gap-3 mb-6">
          <div className="p-2.5 rounded-xl bg-indigo-500/10 border border-indigo-500/20 text-indigo-400">
            <Sliders className="h-5 w-5" />
          </div>
          <div>
            <h2 className="text-lg font-bold text-white">Configure New Execution Order</h2>
            <p className="text-xs text-slate-400">Select parent order parameters, execution policy, and market scenario</p>
          </div>
        </div>

        {error && (
          <div className="mb-6 p-4 rounded-xl bg-rose-500/10 border border-rose-500/20 text-rose-400 text-sm flex items-center gap-2">
            <ShieldAlert className="h-4 w-4 shrink-0" />
            <span>{error}</span>
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-6">
          {/* Order Parameters */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div>
              <label className="block text-xs font-medium text-slate-300 mb-1.5">Symbol</label>
              <select
                value={symbol}
                onChange={(e) => setSymbol(e.target.value)}
                className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-sm text-white focus:border-indigo-500 outline-none"
              >
                <option value="AAPL">AAPL (Apple Inc.)</option>
                <option value="MSFT">MSFT (Microsoft)</option>
                <option value="NVDA">NVDA (NVIDIA)</option>
                <option value="TSLA">TSLA (Tesla)</option>
                <option value="GOOGL">GOOGL (Alphabet)</option>
                <option value="AMZN">AMZN (Amazon)</option>
              </select>
            </div>

            <div>
              <label className="block text-xs font-medium text-slate-300 mb-1.5">Order Side</label>
              <div className="grid grid-cols-2 gap-1.5 p-1 bg-slate-950 border border-slate-800 rounded-xl">
                <button
                  type="button"
                  onClick={() => setSide('BUY')}
                  className={`py-1 text-xs font-medium rounded-lg transition-all ${
                    side === 'BUY' ? 'bg-emerald-600 text-white shadow' : 'text-slate-400 hover:text-white'
                  }`}
                >
                  BUY
                </button>
                <button
                  type="button"
                  onClick={() => setSide('SELL')}
                  className={`py-1 text-xs font-medium rounded-lg transition-all ${
                    side === 'SELL' ? 'bg-rose-600 text-white shadow' : 'text-slate-400 hover:text-white'
                  }`}
                >
                  SELL
                </button>
              </div>
            </div>

            <div>
              <label className="block text-xs font-medium text-slate-300 mb-1.5">Target Shares Quantity</label>
              <input
                type="number"
                value={quantity}
                onChange={(e) => setQuantity(Number(e.target.value))}
                className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-sm text-white focus:border-indigo-500 outline-none"
                placeholder="100000"
                min="1000"
                step="1000"
              />
            </div>

            <div>
              <label className="block text-xs font-medium text-slate-300 mb-1.5">Horizon Steps (Minutes)</label>
              <input
                type="number"
                value={horizonSteps}
                onChange={(e) => setHorizonSteps(Number(e.target.value))}
                className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-sm text-white focus:border-indigo-500 outline-none"
                placeholder="30"
                min="5"
                max="120"
              />
            </div>
          </div>

          {/* Policy Selection */}
          <div>
            <label className="block text-xs font-medium text-slate-300 mb-2">Execution Policy</label>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
              {policies.map((p) => (
                <div
                  key={p.id}
                  onClick={() => setPolicy(p.id)}
                  className={`p-3.5 rounded-xl border cursor-pointer transition-all ${
                    policy === p.id
                      ? 'bg-indigo-600/15 border-indigo-500 shadow-md shadow-indigo-500/10'
                      : 'bg-slate-950/60 border-slate-800/80 hover:border-slate-700'
                  }`}
                >
                  <div className="flex items-center justify-between mb-1">
                    <span className="font-semibold text-sm text-white">{p.name}</span>
                    <span className={`text-[10px] px-2 py-0.5 rounded-full font-medium ${
                      p.tag === 'Recommended'
                        ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                        : p.tag === 'RL Agent'
                        ? 'bg-indigo-500/20 text-indigo-400 border border-indigo-500/30'
                        : 'bg-slate-800 text-slate-400'
                    }`}>
                      {p.tag}
                    </span>
                  </div>
                  <p className="text-xs text-slate-400">{p.desc}</p>
                </div>
              ))}
            </div>
          </div>

          {/* Scenario & Seed Grid */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="md:col-span-2">
              <label className="block text-xs font-medium text-slate-300 mb-1.5">Market Condition Scenario</label>
              <select
                value={scenario}
                onChange={(e) => setScenario(e.target.value)}
                className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-sm text-white focus:border-indigo-500 outline-none"
              >
                {scenarios.map((s) => (
                  <option key={s.id} value={s.id}>
                    {s.name} — {s.desc}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-xs font-medium text-slate-300 mb-1.5">Random Seed (Repeatability)</label>
              <input
                type="number"
                value={seed}
                onChange={(e) => setSeed(Number(e.target.value))}
                className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-sm text-white focus:border-indigo-500 outline-none"
                placeholder="42"
              />
            </div>
          </div>

          {/* Submit Action */}
          <button
            type="submit"
            disabled={loading}
            className="w-full py-3.5 px-4 rounded-xl bg-gradient-to-r from-indigo-600 via-blue-600 to-emerald-500 hover:opacity-95 text-white font-semibold text-sm shadow-lg shadow-indigo-500/25 flex items-center justify-center gap-2 transition-all disabled:opacity-50"
          >
            {loading ? (
              <>
                <div className="h-4 w-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                <span>Simulating Order Execution...</span>
              </>
            ) : (
              <>
                <Play className="h-4 w-4 fill-white" />
                <span>Launch Execution Simulation</span>
              </>
            )}
          </button>
        </form>
      </div>
    </div>
  );
}
