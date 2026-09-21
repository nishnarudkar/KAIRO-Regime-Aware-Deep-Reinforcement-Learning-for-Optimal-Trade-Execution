'use client';

import React, { useEffect, useState } from 'react';
import { Database, RefreshCw, ShieldCheck, Activity, HelpCircle, Layers } from 'lucide-react';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid } from 'recharts';
import { getCurrentRegime, CurrentRegimeResponse, getExperiments, ExperimentSummaryResponse } from '../lib/api';

export function ResearchTab() {
  const [regimeData, setRegimeData] = useState<CurrentRegimeResponse | null>(null);
  const [experiments, setExperiments] = useState<ExperimentSummaryResponse[]>([]);
  const [scenario, setScenario] = useState('normal');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchRegime = async (selectedScenario = scenario) => {
    setLoading(true);
    setError(null);
    try {
      const data = await getCurrentRegime('AAPL', selectedScenario);
      setRegimeData(data);
    } catch (err: any) {
      setError(err.message || 'Failed to detect market regime');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchRegime();
    getExperiments()
      .then(setExperiments)
      .catch(() => setExperiments([]));
  }, []);

  const probChartData = regimeData?.regime_probabilities
    ? Object.entries(regimeData.regime_probabilities).map(([name, prob]) => ({
        name,
        probability: Number((prob * 100).toFixed(1)),
      }))
    : [];

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      {/* Causal Regime Detection Card */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6 shadow-xl">
        <div className="flex flex-wrap items-center justify-between gap-4 mb-6">
          <div className="flex items-center gap-3">
            <div className="p-2.5 rounded-xl bg-indigo-500/10 border border-indigo-500/20 text-indigo-400">
              <Database className="h-5 w-5" />
            </div>
            <div>
              <h2 className="text-lg font-bold text-white">Causal HMM Market Regime Detection</h2>
              <p className="text-xs text-slate-400">Real-time online forward inference P(S_t = k | X_1:t) without temporal lookahead</p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <select
              value={scenario}
              onChange={(e) => {
                setScenario(e.target.value);
                fetchRegime(e.target.value);
              }}
              className="bg-slate-950 border border-slate-800 rounded-xl px-3 py-1.5 text-xs text-white focus:border-indigo-500 outline-none"
            >
              <option value="normal">Scenario: Normal Market</option>
              <option value="high_volatility">Scenario: High Volatility</option>
              <option value="low_liquidity">Scenario: Low Liquidity</option>
              <option value="stress">Scenario: Market Stress</option>
              <option value="regime_transition">Scenario: Regime Transition</option>
            </select>

            <button
              onClick={() => fetchRegime()}
              disabled={loading}
              className="p-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 transition-all"
            >
              <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
            </button>
          </div>
        </div>

        {error && (
          <div className="mb-4 p-3 rounded-xl bg-rose-500/10 border border-rose-500/20 text-rose-400 text-xs">
            {error}
          </div>
        )}

        {regimeData && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {/* Regime State Box */}
            <div className="bg-slate-950/60 border border-slate-800 rounded-xl p-5 flex flex-col justify-between">
              <div>
                <span className="text-xs text-slate-400 block mb-1">Detected Hidden State</span>
                <h3 className="text-2xl font-extrabold text-emerald-400">{regimeData.regime_label}</h3>
                <span className="text-xs text-slate-500 font-mono">Regime ID: {regimeData.regime_id}</span>
              </div>

              <div className="space-y-2 mt-4 pt-4 border-t border-slate-800 text-xs">
                <div className="flex justify-between">
                  <span className="text-slate-400">Current Price:</span>
                  <span className="text-white font-mono">${regimeData.current_price.toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Bid-Ask Spread:</span>
                  <span className="text-white font-mono">${regimeData.spread.toFixed(4)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Volatility:</span>
                  <span className="text-white font-mono">{(regimeData.volatility * 100).toFixed(3)}%</span>
                </div>
              </div>
            </div>

            {/* Posterior Probabilities Bar Chart */}
            <div className="md:col-span-2 bg-slate-950/60 border border-slate-800 rounded-xl p-5">
              <h4 className="text-xs font-bold text-white mb-1">Posterior Regime Probabilities P(S_t = k | X_1:t)</h4>
              <p className="text-[11px] text-slate-400 mb-4">Probability distribution over the 4 canonical HMM market regimes</p>

              <div className="h-44 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={probChartData} layout="vertical">
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                    <XAxis type="number" domain={[0, 100]} stroke="#64748b" tick={{ fontSize: 10 }} />
                    <YAxis dataKey="name" type="category" stroke="#64748b" tick={{ fontSize: 10 }} width={100} />
                    <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155' }} />
                    <Bar dataKey="probability" fill="#10b981" radius={[0, 4, 4, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Research Questions Summary Box */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6 shadow-xl">
        <div className="flex items-center gap-3 mb-4">
          <HelpCircle className="h-5 w-5 text-indigo-400" />
          <h3 className="text-base font-bold text-white">Research Questions (RQ1 – RQ4)</h3>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
          <div className="p-4 rounded-xl bg-slate-950/60 border border-slate-800">
            <h4 className="font-bold text-indigo-300 mb-1">RQ1: DRL Baseline Outperformance</h4>
            <p className="text-slate-400">Does Deep Reinforcement Learning outperform conventional execution baselines (TWAP, VWAP, POV) under identical market conditions?</p>
          </div>

          <div className="p-4 rounded-xl bg-slate-950/60 border border-slate-800">
            <h4 className="font-bold text-indigo-300 mb-1">RQ2: Regime Feature Value</h4>
            <p className="text-slate-400">Does causally-inferred market regime information improve RL trade execution quality and lower implementation shortfall?</p>
          </div>

          <div className="p-4 rounded-xl bg-slate-950/60 border border-slate-800">
            <h4 className="font-bold text-indigo-300 mb-1">RQ3: Transition Robustness & Ablation</h4>
            <p className="text-slate-400">Does regime-awareness improve execution robustness during regime transitions, and does it outperform shuffled-regime controls?</p>
          </div>

          <div className="p-4 rounded-xl bg-slate-950/60 border border-slate-800">
            <h4 className="font-bold text-indigo-300 mb-1">RQ4: Algorithmic Generalisation</h4>
            <p className="text-slate-400">Is the regime-aware performance improvement algorithm-class-agnostic across both value-based (DQN) and policy-gradient (PPO) models?</p>
          </div>
        </div>
      </div>
    </div>
  );
}
