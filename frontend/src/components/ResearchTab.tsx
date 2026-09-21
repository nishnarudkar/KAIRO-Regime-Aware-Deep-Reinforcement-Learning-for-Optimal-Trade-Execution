'use client';

import React, { useEffect, useState } from 'react';
import {
  getCurrentRegime,
  getExperimentResults,
  CurrentRegimeResponse,
  ExperimentResults,
  ComparisonRow,
} from '../lib/api';
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
  { id: 'RQ1', title: 'Deep RL versus execution baselines', body: 'Does deep reinforcement learning outperform TWAP, VWAP and POV under identical market conditions?' },
  { id: 'RQ2', title: 'Value of regime information', body: 'Does causally inferred market regime information lower implementation shortfall for an RL agent?' },
  { id: 'RQ3', title: 'Robustness through transitions', body: 'Does regime-awareness help in turbulent markets, and does it beat a shuffled-regime control?' },
  { id: 'RQ4', title: 'Algorithm generality', body: 'Does the regime-aware effect appear for both value-based (DQN) and policy-gradient (PPO) agents?' },
];

type Verdict = { label: string; tone: string };

/** Plain-language reading of a paired difference (negative = treatment cheaper). */
function verdict(row: ComparisonRow): Verdict {
  if (row.ci_low === null || row.ci_high === null) return { label: '—', tone: 'text-ink-3' };
  if (row.ci_high < 0) return { label: 'cheaper', tone: 'text-pos' };
  if (row.ci_low > 0) return { label: 'costlier', tone: 'text-neg' };
  return { label: 'no clear difference', tone: 'text-ink-2' };
}

const fmtP = (p: number | null) => (p === null ? '—' : p < 0.001 ? '<0.001' : p.toFixed(3));

function ExperimentPanel({ results }: { results: ExperimentResults }) {
  const [scope, setScope] = useState('ALL');
  const scopes = ['ALL', 'TURBULENT', ...(results.config.scenarios ?? [])];

  const windowRows = results.comparisons.filter((c) => c.level === 'window' && c.scope === scope);
  const seedRow = (r: ComparisonRow) =>
    results.comparisons.find(
      (c) => c.level === 'seed' && c.scope === scope && c.treatment === r.treatment && c.control === r.control,
    );
  const groups: [string, string][] = [
    ['RQ1', 'Learned policy vs baseline'],
    ['RQ2', 'Regime-aware vs same agent without regime'],
    ['RQ3-control', 'Regime-aware vs shuffled-regime control'],
  ];

  const cfg = results.config;
  return (
    <Panel>
      <PanelHeader
        title="Recorded experiment results"
        note={`${cfg.scenarios?.length ?? '?'} scenarios × ${cfg.seeds?.length ?? '?'} seeds, ${(cfg.train_timesteps ?? 0).toLocaleString('en-US')} training steps per model, evaluated on non-overlapping out-of-sample windows of ${cfg.horizon_steps ?? 30} bars. Differences are paired: treatment minus control in implementation-shortfall bps, so negative means cheaper.`}
        right={
          <div>
            <label htmlFor="rs-scope" className="sr-only">Scope</label>
            <select id="rs-scope" value={scope} onChange={(e) => setScope(e.target.value)} className="field !w-auto">
              {scopes.map((s) => (
                <option key={s} value={s}>
                  {s === 'ALL' ? 'All scenarios' : s === 'TURBULENT' ? 'Turbulent scenarios' : s.replace(/_/g, ' ')}
                </option>
              ))}
            </select>
          </div>
        }
      />

      <div className="overflow-x-auto">
        <table className="w-full text-[13px] min-w-[760px]">
          <thead>
            <tr className="label text-left border-b border-line">
              <th className="px-5 py-2.5 font-medium">Comparison</th>
              <th className="py-2.5 font-medium text-right">Δ shortfall (bps)</th>
              <th className="py-2.5 font-medium text-right">95% CI</th>
              <th className="py-2.5 font-medium text-right">p (Wilcoxon)</th>
              <th className="py-2.5 font-medium text-right">Seeds favouring</th>
              <th className="px-5 py-2.5 font-medium text-right">Reading</th>
            </tr>
          </thead>
          <tbody>
            {groups.map(([rq, title]) => {
              const rows = windowRows.filter((r) => r.rq === rq);
              if (rows.length === 0) return null;
              return (
                <React.Fragment key={rq}>
                  <tr className="border-b border-line bg-raised/40">
                    <td colSpan={6} className="px-5 py-2 text-[12px] text-ink-3">
                      <span className="text-accent num mr-2">{rq.replace('-control', '')}</span>
                      {title}
                    </td>
                  </tr>
                  {rows.map((r) => {
                    const v = verdict(r);
                    const sr = seedRow(r);
                    return (
                      <tr key={`${r.treatment}-${r.control}`} className="border-b border-line last:border-b-0">
                        <td className="px-5 py-2.5 text-ink">
                          {r.treatment} <span className="text-ink-3">vs</span> {r.control}
                        </td>
                        <td className={`py-2.5 num text-right ${v.tone}`}>
                          {r.mean > 0 ? '+' : ''}
                          {r.mean.toFixed(2)}
                        </td>
                        <td className="py-2.5 num text-right text-ink-2">
                          [{r.ci_low?.toFixed(1)}, {r.ci_high?.toFixed(1)}]
                        </td>
                        <td className="py-2.5 num text-right text-ink-2">{fmtP(r.p_value)}</td>
                        <td className="py-2.5 num text-right text-ink-2">
                          {sr && sr.seeds_favouring !== null ? `${sr.seeds_favouring}/${sr.n_seeds}` : '—'}
                        </td>
                        <td className={`px-5 py-2.5 text-right ${v.tone}`}>{v.label}</td>
                      </tr>
                    );
                  })}
                </React.Fragment>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className="px-5 py-4 border-t border-line">
        <p className="label mb-2">Mean shortfall by strategy (all scenarios, all seeds)</p>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-x-8">
          {[...results.summary]
            .sort((a, b) => a.mean - b.mean)
            .map((s) => (
              <div key={s.strategy} className="flex items-baseline justify-between py-1.5 border-b border-line text-[13px]">
                <span className="text-ink-2">{s.strategy}</span>
                <span className="num text-ink">
                  {s.mean.toFixed(2)}
                  <span className="text-ink-3"> ± {(s.std ?? 0).toFixed(1)}</span>
                </span>
              </div>
            ))}
        </div>
        <p className="text-[12px] text-ink-3 mt-3">
          The ± is the spread across scenario/seed runs and mostly reflects market-path luck, not strategy quality; use the
          paired differences above to compare strategies.
          {results.hmm_validation.mean_ari != null &&
            ` The causal HMM recovers the latent regime with mean adjusted Rand index ${results.hmm_validation.mean_ari.toFixed(2)} and accuracy ${((results.hmm_validation.mean_accuracy ?? 0) * 100).toFixed(0)}% on the test region.`}
        </p>
      </div>
    </Panel>
  );
}

export function ResearchTab() {
  const [scenario, setScenario] = useState('normal');
  const [refreshKey, setRefreshKey] = useState(0);
  const [regime, setRegime] = useState<{ key: string; data?: CurrentRegimeResponse; error?: string } | null>(null);
  const [results, setResults] = useState<ExperimentResults | null | undefined>(undefined);

  const requestKey = `${scenario}:${refreshKey}`;
  const loading = regime?.key !== requestKey;

  useEffect(() => {
    let cancelled = false;
    getCurrentRegime('AAPL', scenario, 42)
      .then((data) => !cancelled && setRegime({ key: requestKey, data }))
      .catch((err: unknown) =>
        !cancelled && setRegime({ key: requestKey, error: err instanceof Error ? err.message : 'Failed to detect market regime' }),
      );
    return () => {
      cancelled = true;
    };
  }, [scenario, requestKey]);

  useEffect(() => {
    getExperimentResults()
      .then(setResults)
      .catch(() => setResults(null));
  }, []);

  const regimeData = regime?.data;
  const probs = regimeData
    ? REGIME_ORDER.filter((n) => n in regimeData.regime_probabilities).map((n) => ({
        name: n,
        p: regimeData.regime_probabilities[n],
      }))
    : [];

  return (
    <div className="space-y-6">
      <PageTitle eyebrow="Regime detection and results" title="Research">
        <div className="flex items-center gap-2">
          <label htmlFor="rs-scenario" className="sr-only">Scenario</label>
          <select id="rs-scenario" value={scenario} onChange={(e) => setScenario(e.target.value)} className="field !w-auto">
            {SCENARIOS.map(([id, name]) => (
              <option key={id} value={id}>
                {name}
              </option>
            ))}
          </select>
          <button onClick={() => setRefreshKey((k) => k + 1)} disabled={loading} className="btn btn-ghost">
            {loading ? <span className="spinner" /> : null}
            Refresh
          </button>
        </div>
      </PageTitle>

      {regime?.error && (
        <p role="alert" className="text-[13px] text-neg border-l-2 border-neg pl-3">
          {regime.error}
        </p>
      )}

      {regimeData && (
        <Panel>
          <PanelHeader
            title="Causal HMM regime"
            note="Forward-filtered posterior P(Sₜ = k | X₁:ₜ) at the latest bar of a synthetic market, computed online with no lookahead."
          />
          <div className="grid grid-cols-1 md:grid-cols-[280px_1fr] md:divide-x divide-line">
            <div className="p-5">
              <p className="label">Detected state</p>
              <p className="text-[26px] font-semibold leading-tight mt-1.5" style={{ color: regimeColor(regimeData.regime_label) }}>
                {regimeData.regime_label}
              </p>
              <p className="num text-[12px] text-ink-3 mt-0.5">
                State {regimeData.regime_id}
                {regimeData.true_regime && ` · true regime: ${regimeData.true_regime}`}
              </p>
              <div className="mt-4">
                <KeyValueTable
                  rows={[
                    { k: 'Price', v: fmtUsd(regimeData.current_price) },
                    { k: 'Bid-ask spread', v: fmtUsd(regimeData.spread, 4) },
                    { k: 'Volatility (20-bar)', v: `${(regimeData.volatility * 100).toFixed(3)}%` },
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

      {results === undefined && <p className="text-[13px] text-ink-3">Loading experiment results…</p>}
      {results === null && (
        <p className="text-[13px] text-ink-3 border-l-2 border-line-strong pl-3">
          No experiment results found on the backend. Run <span className="num">python scripts/run_experiments.py</span> to
          produce them.
        </p>
      )}
      {results && <ExperimentPanel results={results} />}

      <Panel>
        <PanelHeader title="Research questions" note="Answers are the paired statistics above, not a fixed claim." />
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
