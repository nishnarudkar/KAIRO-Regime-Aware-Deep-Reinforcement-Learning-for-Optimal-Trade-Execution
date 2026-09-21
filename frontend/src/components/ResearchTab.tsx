'use client';

import React, { useEffect, useState } from 'react';
import { getCurrentRegime, CurrentRegimeResponse, getExperiments, ExperimentSummaryResponse } from '../lib/api';
import { Panel, PanelHeader, PageTitle, KeyValueTable, regimeColor, fmtUsd } from './ui';

const REGIME_ORDER = ['Low Volatility', 'Normal', 'High Volatility', 'Stress'];

const SCENARIOS = [
  ['normal', 'Normal market'],
  ['high_volatility', 'High volatility'],
  ['low_liquidity', 'Low liquidity'],
  ['stress', 'Market stress'],
  ['regime_transition', 'Regime transition'],
  ['liquidity_shock', 'Liquidity shock'],
];

const QUESTIONS = [
  {
    id: 'RQ1',
    title: 'Deep RL versus execution baselines',
    body: 'Does deep reinforcement learning outperform TWAP, VWAP and POV under identical market conditions?',
  },
  {
    id: 'RQ2',
    title: 'Value of regime information',
    body: 'Does causally inferred market regime information lower implementation shortfall for an RL agent?',
  },
  {
    id: 'RQ3',
    title: 'Robustness through transitions',
    body: 'Does regime-awareness help during regime transitions, and does it beat a shuffled-regime control?',
  },
  {
    id: 'RQ4',
    title: 'Algorithm generality',
    body: 'Does the regime-aware improvement hold for both value-based (DQN) and policy-gradient (PPO) agents?',
  },
];

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
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to detect market regime');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchRegime();
    getExperiments()
      .then(setExperiments)
      .catch(() => setExperiments([]));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const probs = regimeData
    ? REGIME_ORDER.filter((n) => n in regimeData.regime_probabilities).map((n) => ({
        name: n,
        p: regimeData.regime_probabilities[n],
      }))
    : [];

  return (
    <div className="space-y-6">
      <PageTitle eyebrow="Regime detection" title="Research">
        <div className="flex items-center gap-2">
          <label htmlFor="rs-scenario" className="sr-only">Scenario</label>
          <select
            id="rs-scenario"
            value={scenario}
            onChange={(e) => {
              setScenario(e.target.value);
              fetchRegime(e.target.value);
            }}
            className="field !w-auto"
          >
            {SCENARIOS.map(([id, name]) => (
              <option key={id} value={id}>
                {name}
              </option>
            ))}
          </select>
          <button onClick={() => fetchRegime()} disabled={loading} className="btn btn-ghost">
            {loading ? <span className="spinner" /> : null}
            Refresh
          </button>
        </div>
      </PageTitle>

      {error && (
        <p role="alert" className="text-[13px] text-neg border-l-2 border-neg pl-3">
          {error}
        </p>
      )}

      {regimeData && (
        <Panel>
          <PanelHeader
            title="Causal HMM regime"
            note="Forward-filtered posterior P(Sₜ = k | X₁:ₜ), computed online with no lookahead."
          />
          <div className="grid grid-cols-1 md:grid-cols-[280px_1fr] md:divide-x divide-line">
            <div className="p-5">
              <p className="label">Detected state</p>
              <p className="text-[26px] font-semibold leading-tight mt-1.5 flex items-center gap-2.5" style={{ color: regimeColor(regimeData.regime_label) }}>
                {regimeData.regime_label}
              </p>
              <p className="num text-[12px] text-ink-3 mt-0.5">State {regimeData.regime_id}</p>
              <div className="mt-4">
                <KeyValueTable
                  rows={[
                    { k: 'Price', v: fmtUsd(regimeData.current_price) },
                    { k: 'Bid-ask spread', v: fmtUsd(regimeData.spread, 4) },
                    { k: 'Volatility', v: `${(regimeData.volatility * 100).toFixed(3)}%` },
                  ]}
                />
              </div>
            </div>

            <div className="p-5">
              <p className="label mb-4">Posterior probability</p>
              <ul className="space-y-3.5">
                {probs.map(({ name, p }) => {
                  const on = name === regimeData.regime_label;
                  return (
                    <li key={name} className="grid grid-cols-[110px_1fr_56px] items-center gap-3">
                      <span className={`text-[13px] ${on ? 'text-ink font-medium' : 'text-ink-2'}`}>{name}</span>
                      <span className="h-2.5 bg-raised block" role="img" aria-label={`${name}: ${(p * 100).toFixed(1)}%`}>
                        <span
                          className="block h-full transition-[width] duration-500"
                          style={{ width: `${p * 100}%`, background: regimeColor(name), opacity: on ? 1 : 0.55 }}
                        />
                      </span>
                      <span className={`num text-[13px] text-right ${on ? 'text-ink' : 'text-ink-2'}`}>{(p * 100).toFixed(1)}%</span>
                    </li>
                  );
                })}
              </ul>
            </div>
          </div>
        </Panel>
      )}

      {experiments.length > 0 && (
        <Panel>
          <PanelHeader title="Experiments" note="Registered experiment suites from the backend" />
          <div className="overflow-x-auto">
            <table className="w-full text-[13px] min-w-[640px]">
              <thead>
                <tr className="label text-left border-b border-line">
                  <th className="px-5 py-2.5 font-medium">Name</th>
                  <th className="py-2.5 font-medium">Scenarios</th>
                  <th className="py-2.5 font-medium">Strategies</th>
                  <th className="px-5 py-2.5 font-medium text-right">Status</th>
                </tr>
              </thead>
              <tbody>
                {experiments.map((e) => (
                  <tr key={e.experiment_id} className="border-b border-line last:border-b-0">
                    <td className="px-5 py-3 text-ink">{e.name}</td>
                    <td className="py-3 num text-ink-2">{e.scenarios.length}</td>
                    <td className="py-3 num text-ink-2">{e.strategies.length}</td>
                    <td className="px-5 py-3 text-right text-ink-2">{e.status}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Panel>
      )}

      <Panel>
        <PanelHeader title="Research questions" />
        <ol>
          {QUESTIONS.map((q) => (
            <li key={q.id} className="grid grid-cols-[48px_1fr] gap-x-2 px-5 py-4 border-b border-line last:border-b-0">
              <span className="num text-[12.5px] text-accent pt-0.5">{q.id}</span>
              <div>
                <h3 className="text-[14px] font-medium text-ink">{q.title}</h3>
                <p className="text-[13px] text-ink-2 mt-0.5 max-w-2xl">{q.body}</p>
              </div>
            </li>
          ))}
        </ol>
      </Panel>
    </div>
  );
}
