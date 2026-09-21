'use client';

import React, { useState } from 'react';
import { simulateExecution, ExecutionRecord } from '../lib/api';
import { Panel, PanelHeader, PageTitle, KeyValueTable, fmtInt } from './ui';

interface NewExecutionTabProps {
  onExecutionCreated: (record: ExecutionRecord) => void;
}

const POLICIES = [
  { id: 'TWAP', name: 'TWAP', group: 'Baseline', state: '—', desc: 'Uniform slices across the horizon' },
  { id: 'VWAP', name: 'VWAP', group: 'Baseline', state: '—', desc: 'Follows the volume profile' },
  { id: 'POV', name: 'POV (10%)', group: 'Baseline', state: '—', desc: 'Fixed 10% participation of market volume' },
  { id: 'DQN', name: 'DQN', group: 'RL agent', state: '7', desc: 'Value-based agent, market state only' },
  { id: 'Regime-Aware DQN', name: 'Regime-Aware DQN', group: 'RL agent', state: '12', desc: 'DQN plus causal HMM regime features' },
  { id: 'PPO', name: 'PPO', group: 'RL agent', state: '7', desc: 'Policy-gradient agent, market state only' },
  { id: 'Regime-Aware PPO', name: 'Regime-Aware PPO', group: 'RL agent', state: '12', desc: 'PPO plus causal HMM regime features' },
];

const SCENARIOS = [
  { id: 'normal', name: 'Normal market', desc: 'Standard liquid trading conditions' },
  { id: 'high_volatility', name: 'High volatility', desc: 'Elevated price volatility and wide spreads' },
  { id: 'low_liquidity', name: 'Low liquidity', desc: 'Thin order book depth and high impact' },
  { id: 'stress', name: 'Market stress', desc: 'Extreme volatility compounded with illiquidity' },
  { id: 'regime_transition', name: 'Regime transition', desc: 'Volatility regime shifts mid-horizon' },
  { id: 'liquidity_shock', name: 'Liquidity shock', desc: 'Order book liquidity collapses suddenly' },
];

const SYMBOLS = [
  ['AAPL', 'Apple'],
  ['MSFT', 'Microsoft'],
  ['NVDA', 'NVIDIA'],
  ['TSLA', 'Tesla'],
  ['GOOGL', 'Alphabet'],
  ['AMZN', 'Amazon'],
];

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
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Simulation execution failed');
    } finally {
      setLoading(false);
    }
  };

  const selectedPolicy = POLICIES.find((p) => p.id === policy);
  const selectedScenario = SCENARIOS.find((s) => s.id === scenario);
  const perStep = horizonSteps > 0 ? Math.round(quantity / horizonSteps) : 0;

  return (
    <div>
      <PageTitle eyebrow="New execution" title="Order ticket" />

      <form onSubmit={handleSubmit} className="grid grid-cols-1 lg:grid-cols-[minmax(0,1fr)_320px] gap-6 items-start">
        <div className="space-y-6">
          <Panel>
            <PanelHeader title="Parent order" />
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-5 gap-y-4 p-5">
              <div>
                <label htmlFor="symbol" className="label block mb-1.5">Symbol</label>
                <select id="symbol" value={symbol} onChange={(e) => setSymbol(e.target.value)} className="field">
                  {SYMBOLS.map(([s, n]) => (
                    <option key={s} value={s}>
                      {s} · {n}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <span className="label block mb-1.5" id="side-label">Side</span>
                <div role="radiogroup" aria-labelledby="side-label" className="grid grid-cols-2 border border-line-strong rounded-[3px] overflow-hidden">
                  {['BUY', 'SELL'].map((s) => {
                    const on = side === s;
                    return (
                      <button
                        key={s}
                        type="button"
                        role="radio"
                        aria-checked={on}
                        onClick={() => setSide(s)}
                        className={`py-[7px] text-[13px] font-medium transition-colors ${
                          on
                            ? s === 'BUY'
                              ? 'bg-pos text-[#08140e]'
                              : 'bg-neg text-[#170907]'
                            : 'text-ink-3 hover:text-ink hover:bg-raised'
                        }`}
                      >
                        {s === 'BUY' ? 'Buy' : 'Sell'}
                      </button>
                    );
                  })}
                </div>
              </div>

              <div>
                <label htmlFor="qty" className="label block mb-1.5">Quantity (shares)</label>
                <input
                  id="qty"
                  type="number"
                  value={quantity}
                  onChange={(e) => setQuantity(Number(e.target.value))}
                  className="field num"
                  min="1000"
                  step="1000"
                />
              </div>

              <div>
                <label htmlFor="horizon" className="label block mb-1.5">Horizon (minutes)</label>
                <input
                  id="horizon"
                  type="number"
                  value={horizonSteps}
                  onChange={(e) => setHorizonSteps(Number(e.target.value))}
                  className="field num"
                  min="5"
                  max="120"
                />
              </div>
            </div>
          </Panel>

          <Panel>
            <PanelHeader title="Execution policy" note="Baselines are rule-based. RL agents are trained policies; regime-aware variants observe 5 extra HMM features." />
            <div role="radiogroup" aria-label="Execution policy">
              <div className="hidden sm:grid grid-cols-[1fr_88px_72px] px-5 py-2 border-b border-line label">
                <span>Policy</span>
                <span>Type</span>
                <span className="text-right">State dim</span>
              </div>
              {POLICIES.map((p) => {
                const on = policy === p.id;
                return (
                  <button
                    key={p.id}
                    type="button"
                    role="radio"
                    aria-checked={on}
                    onClick={() => setPolicy(p.id)}
                    className={`w-full text-left grid grid-cols-1 sm:grid-cols-[1fr_88px_72px] items-baseline gap-x-4 px-5 py-3 border-b border-line last:border-b-0 relative transition-colors ${
                      on ? 'bg-raised' : 'hover:bg-raised/50'
                    }`}
                  >
                    {on && <span className="absolute left-0 top-0 bottom-0 w-[2px] bg-accent" />}
                    <span>
                      <span className={`block text-[13.5px] ${on ? 'text-ink font-medium' : 'text-ink'}`}>{p.name}</span>
                      <span className="block text-[12.5px] text-ink-3">{p.desc}</span>
                    </span>
                    <span className="text-[12.5px] text-ink-2 hidden sm:block">{p.group}</span>
                    <span className="num text-[12.5px] text-ink-2 text-right hidden sm:block">{p.state}</span>
                  </button>
                );
              })}
            </div>
          </Panel>

          <Panel>
            <PanelHeader title="Market conditions" />
            <div className="grid grid-cols-1 sm:grid-cols-[1fr_140px] gap-x-5 gap-y-4 p-5">
              <div>
                <label htmlFor="scenario" className="label block mb-1.5">Scenario</label>
                <select id="scenario" value={scenario} onChange={(e) => setScenario(e.target.value)} className="field">
                  {SCENARIOS.map((s) => (
                    <option key={s.id} value={s.id}>
                      {s.name}
                    </option>
                  ))}
                </select>
                <p className="text-[12.5px] text-ink-3 mt-1.5">{selectedScenario?.desc}</p>
              </div>
              <div>
                <label htmlFor="seed" className="label block mb-1.5">Random seed</label>
                <input id="seed" type="number" value={seed} onChange={(e) => setSeed(Number(e.target.value))} className="field num" />
              </div>
            </div>
          </Panel>
        </div>

        {/* Summary rail */}
        <aside className="lg:sticky lg:top-[88px]">
          <Panel>
            <div className="px-5 py-4 border-b border-line">
              <p className="label mb-1.5">Order summary</p>
              <p className="text-[18px] font-semibold text-ink leading-snug">
                <span className={side === 'BUY' ? 'text-pos' : 'text-neg'}>{side === 'BUY' ? 'Buy' : 'Sell'}</span>{' '}
                <span className="tabular-nums">{fmtInt(Number(quantity))}</span> {symbol}
              </p>
            </div>
            <div className="px-5 py-1">
              <KeyValueTable
                rows={[
                  { k: 'Policy', v: <span className="font-sans">{selectedPolicy?.name}</span> },
                  { k: 'Scenario', v: <span className="font-sans">{selectedScenario?.name}</span> },
                  { k: 'Horizon', v: `${horizonSteps} min` },
                  { k: 'Avg. per minute', v: `${fmtInt(perStep)} sh` },
                  { k: 'Seed', v: seed },
                ]}
              />
            </div>
            <div className="p-5 pt-4 border-t border-line">
              {error && (
                <p role="alert" className="mb-3 text-[12.5px] text-neg border-l-2 border-neg pl-3">
                  {error}
                </p>
              )}
              <button type="submit" disabled={loading} className="btn btn-primary w-full">
                {loading ? (
                  <>
                    <span className="spinner" />
                    Simulating…
                  </>
                ) : (
                  'Run simulation'
                )}
              </button>
              <p className="text-[12px] text-ink-3 mt-3">Simulated fills only. No orders are sent to a broker.</p>
            </div>
          </Panel>
        </aside>
      </form>
    </div>
  );
}
