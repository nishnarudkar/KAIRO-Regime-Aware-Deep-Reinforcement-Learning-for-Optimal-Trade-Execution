'use client';

import React, { useState } from 'react';
import { Layers, Play, Trophy, ArrowRight, ShieldCheck, Sparkles } from 'lucide-react';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid } from 'recharts';
import { runBacktest, BacktestResponse } from '../lib/api';

export function StrategyComparisonTab() {
  const [scenario, setScenario] = useState('normal');
  const [quantity, setQuantity] = useState(100000);
  const [horizonSteps, setHorizonSteps] = useState(30);
  const [seed, setSeed] = useState(42);
  const [loading, setLoading] = useState(false);
  const [backtestData, setBacktestData] = useState<BacktestResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleRunBacktest = async () => {
    setLoading(true);
    setError(null);

    try {
      const data = await runBacktest({
        symbol: 'AAPL',
        side: 'BUY',
        quantity: Number(quantity),
        horizon_steps: Number(horizonSteps),
        policies: ['TWAP', 'VWAP', 'POV', 'DQN', 'Regime-Aware DQN', 'PPO', 'Regime-Aware PPO'],
        scenario,
        seed: Number(seed),
      });
      setBacktestData(data);
    } catch (err: any) {
      setError(err.message || 'Backtest comparison failed');
    } finally {
      setLoading(false);
    }
  };

  const chartData = (backtestData?.results || []).map((r) => ({
    policy: r.policy,
    is_bps: r.implementation_shortfall_bps,
    fill: r.policy.includes('Regime') ? '#10b981' : r.policy.includes('DQN') || r.policy.includes('PPO') ? '#6366f1' : '#64748b',
  }));

  const bestStrategy = backtestData?.results.reduce((prev, current) =>
    prev.implementation_shortfall_bps < current.implementation_shortfall_bps ? prev : current
  , backtestData.results[0]);

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      {/* Control Card */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6 shadow-xl">
        <div className="flex flex-wrap items-center justify-between gap-4 mb-6">
          <div className="flex items-center gap-3">
            <div className="p-2.5 rounded-xl bg-indigo-500/10 border border-indigo-500/20 text-indigo-400">
              <Layers className="h-5 w-5" />
            </div>
            <div>
              <h2 className="text-lg font-bold text-white">Multi-Strategy Benchmark Comparison</h2>
              <p className="text-xs text-slate-400">Run head-to-head backtest across baselines and DRL agents under identical market conditions</p>
            </div>
          </div>

          <button
            onClick={handleRunBacktest}
            disabled={loading}
            className="px-5 py-2.5 rounded-xl bg-gradient-to-r from-indigo-600 to-blue-600 hover:opacity-95 text-white font-semibold text-xs shadow-lg shadow-indigo-500/20 flex items-center gap-2 transition-all disabled:opacity-50"
          >
            {loading ? (
              <>
                <div className="h-3.5 w-3.5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                <span>Running Backtest Suite...</span>
              </>
            ) : (
              <>
                <Play className="h-3.5 w-3.5 fill-white" />
                <span>Run Backtest Benchmark</span>
              </>
            )}
          </button>
        </div>

        {/* Backtest Config Inputs */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 pt-2">
          <div>
            <label className="block text-xs font-medium text-slate-300 mb-1.5">Market Scenario</label>
            <select
              value={scenario}
              onChange={(e) => setScenario(e.target.value)}
              className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-xs text-white focus:border-indigo-500 outline-none"
            >
              <option value="normal">Normal Market</option>
              <option value="high_volatility">High Volatility</option>
              <option value="low_liquidity">Low Liquidity</option>
              <option value="stress">Market Stress</option>
              <option value="regime_transition">Regime Transition</option>
              <option value="liquidity_shock">Liquidity Shock</option>
            </select>
          </div>

          <div>
            <label className="block text-xs font-medium text-slate-300 mb-1.5">Target Inventory</label>
            <input
              type="number"
              value={quantity}
              onChange={(e) => setQuantity(Number(e.target.value))}
              className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-xs text-white focus:border-indigo-500 outline-none"
            />
          </div>

          <div>
            <label className="block text-xs font-medium text-slate-300 mb-1.5">Random Seed</label>
            <input
              type="number"
              value={seed}
              onChange={(e) => setSeed(Number(e.target.value))}
              className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-xs text-white focus:border-indigo-500 outline-none"
            />
          </div>
        </div>
      </div>

      {error && (
        <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/20 text-rose-400 text-sm">
          {error}
        </div>
      )}

      {/* Results View */}
      {backtestData && (
        <div className="space-y-6">
          {/* Winner Highlight Box */}
          {bestStrategy && (
            <div className="bg-emerald-500/10 border border-emerald-500/20 rounded-2xl p-5 flex items-center justify-between gap-4">
              <div className="flex items-center gap-3">
                <div className="p-3 rounded-xl bg-emerald-500/20 text-emerald-400">
                  <Trophy className="h-6 w-6" />
                </div>
                <div>
                  <div className="text-xs font-medium text-emerald-400">Top Performing Strategy</div>
                  <h3 className="text-lg font-bold text-white">{bestStrategy.policy}</h3>
                  <p className="text-xs text-slate-300">Achieved lowest Implementation Shortfall of {bestStrategy.implementation_shortfall_bps.toFixed(2)} bps</p>
                </div>
              </div>
              <div className="text-right">
                <span className="text-xs text-slate-400 block">Execution Cost</span>
                <span className="text-lg font-bold text-emerald-400">${bestStrategy.execution_cost.toLocaleString()}</span>
              </div>
            </div>
          )}

          {/* Comparison Bar Chart */}
          <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6 shadow-xl">
            <h3 className="text-sm font-bold text-white mb-1">Implementation Shortfall Comparison (IS bps)</h3>
            <p className="text-xs text-slate-400 mb-4">Lower implementation shortfall represents superior execution quality</p>

            <div className="h-64 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="policy" stroke="#64748b" tick={{ fontSize: 10 }} />
                  <YAxis stroke="#64748b" tick={{ fontSize: 11 }} />
                  <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155' }} />
                  <Bar dataKey="is_bps" fill="#6366f1" radius={[6, 6, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Benchmark Results Table */}
          <div className="bg-slate-900/60 border border-slate-800 rounded-2xl overflow-hidden shadow-xl">
            <div className="p-4 border-b border-slate-800">
              <h3 className="text-sm font-bold text-white">Backtest Benchmark Results Table</h3>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead className="bg-slate-950 text-slate-400 border-b border-slate-800">
                  <tr>
                    <th className="p-3.5 font-semibold">Policy Name</th>
                    <th className="p-3.5 font-semibold text-right">IS (bps)</th>
                    <th className="p-3.5 font-semibold text-right">Execution Cost ($)</th>
                    <th className="p-3.5 font-semibold text-right">Completion Rate</th>
                    <th className="p-3.5 font-semibold text-right">VWAP Slippage (bps)</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 text-slate-300">
                  {backtestData.results.map((r, i) => (
                    <tr key={i} className="hover:bg-slate-800/30 transition-all">
                      <td className="p-3.5 font-semibold text-white flex items-center gap-2">
                        <span>{r.policy}</span>
                        {r.policy.includes('Regime') && (
                          <span className="text-[10px] px-1.5 py-0.5 rounded bg-emerald-500/20 text-emerald-400 font-normal">Regime-Aware</span>
                        )}
                      </td>
                      <td className="p-3.5 text-right font-mono font-bold text-indigo-400">{r.implementation_shortfall_bps.toFixed(2)}</td>
                      <td className="p-3.5 text-right font-mono">${r.execution_cost.toLocaleString(undefined, { minimumFractionDigits: 2 })}</td>
                      <td className="p-3.5 text-right font-mono text-emerald-400">{(r.completion_rate * 100).toFixed(1)}%</td>
                      <td className="p-3.5 text-right font-mono text-blue-400">{r.vwap_slippage_bps.toFixed(2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
