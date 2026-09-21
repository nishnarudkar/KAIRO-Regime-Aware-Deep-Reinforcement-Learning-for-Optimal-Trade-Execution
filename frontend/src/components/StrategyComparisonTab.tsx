'use client';

import React, { useState } from 'react';
import { runBacktest, BacktestResponse, BacktestResultItem } from '../lib/api';
import { Panel, PanelHeader, PageTitle, fmtUsd, fmtInt } from './ui';

const SCENARIOS = [
  ['normal', 'Normal market'],
  ['high_volatility', 'High volatility'],
  ['low_liquidity', 'Low liquidity'],
  ['stress', 'Market stress'],
  ['regime_transition', 'Regime transition'],
  ['liquidity_shock', 'Liquidity shock'],
];

const BASELINES = ['TWAP', 'VWAP', 'POV'];
const isBaseline = (p: string) => BASELINES.includes(p);
const isRegimeAware = (p: string) => p.includes('Regime');

/** Paired difference vs TWAP, with a plain-language significance label from its CI. */
function VsTwap({ r }: { r: BacktestResultItem }) {
  if (r.vs_twap_bps === null || r.vs_twap_ci_low === null || r.vs_twap_ci_high === null) {
    return <span className="text-ink-3">reference</span>;
  }
  const spansZero = r.vs_twap_ci_low <= 0 && r.vs_twap_ci_high >= 0;
  const tone = spansZero ? 'text-ink-2' : r.vs_twap_bps < 0 ? 'text-pos' : 'text-neg';
  return (
    <span className={tone}>
      {r.vs_twap_bps > 0 ? '+' : ''}
      {r.vs_twap_bps.toFixed(2)}
      <span className="text-ink-3">
        {' '}
        [{r.vs_twap_ci_low.toFixed(1)}, {r.vs_twap_ci_high.toFixed(1)}]
      </span>
      {spansZero && <span className="text-ink-3"> n.s.</span>}
    </span>
  );
}

export function StrategyComparisonTab() {
  const [scenario, setScenario] = useState('normal');
  const [quantity, setQuantity] = useState(100000);
  const [horizonSteps, setHorizonSteps] = useState(30);
  const [nWindows, setNWindows] = useState(10);
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
        n_windows: Number(nWindows),
        policies: ['TWAP', 'VWAP', 'POV', 'DQN', 'Regime-Aware DQN', 'PPO', 'Regime-Aware PPO'],
        scenario,
        seed: Number(seed),
      });
      setBacktestData(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Backtest comparison failed');
    } finally {
      setLoading(false);
    }
  };

  const ranked: BacktestResultItem[] = backtestData
    ? [...backtestData.results].sort((a, b) => a.implementation_shortfall_bps - b.implementation_shortfall_bps)
    : [];
  const best = ranked[0];
  const bestBaseline = ranked.find((r) => isBaseline(r.policy));
  const bestRl = ranked.find((r) => !isBaseline(r.policy));

  // Bars diverge from a zero line so the sign of the shortfall stays visible.
  const lo = Math.min(0, ...ranked.map((r) => r.implementation_shortfall_bps));
  const hi = Math.max(0, ...ranked.map((r) => r.implementation_shortfall_bps));
  const span = Math.max(1e-9, hi - lo);
  const zeroPct = (-lo / span) * 100;

  return (
    <div className="space-y-6">
      <PageTitle eyebrow="Backtest" title="Strategy comparison" />

      <Panel>
        <PanelHeader
          title="Benchmark setup"
          note="Runs every available policy on the same out-of-sample windows of one synthetic market, so results are paired."
        />
        <div className="p-5 grid grid-cols-2 lg:grid-cols-[1.4fr_1fr_1fr_0.8fr_0.8fr_auto] gap-x-5 gap-y-4 items-end">
          <div className="col-span-2 lg:col-span-1">
            <label htmlFor="cmp-scenario" className="label block mb-1.5">Scenario</label>
            <select id="cmp-scenario" value={scenario} onChange={(e) => setScenario(e.target.value)} className="field">
              {SCENARIOS.map(([id, name]) => (
                <option key={id} value={id}>
                  {name}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label htmlFor="cmp-qty" className="label block mb-1.5">Quantity</label>
            <input id="cmp-qty" type="number" value={quantity} onChange={(e) => setQuantity(Number(e.target.value))} className="field num" min="1000" step="1000" />
          </div>
          <div>
            <label htmlFor="cmp-horizon" className="label block mb-1.5">Horizon (min)</label>
            <input id="cmp-horizon" type="number" value={horizonSteps} onChange={(e) => setHorizonSteps(Number(e.target.value))} className="field num" min="5" max="120" />
          </div>
          <div>
            <label htmlFor="cmp-windows" className="label block mb-1.5">Windows</label>
            <input id="cmp-windows" type="number" value={nWindows} onChange={(e) => setNWindows(Number(e.target.value))} className="field num" min="1" max="30" />
          </div>
          <div>
            <label htmlFor="cmp-seed" className="label block mb-1.5">Seed</label>
            <input id="cmp-seed" type="number" value={seed} onChange={(e) => setSeed(Number(e.target.value))} className="field num" />
          </div>
          <button onClick={handleRunBacktest} disabled={loading} className="btn btn-primary col-span-2 lg:col-span-1">
            {loading ? (
              <>
                <span className="spinner" />
                Running…
              </>
            ) : (
              'Run benchmark'
            )}
          </button>
        </div>
      </Panel>

      {error && (
        <p role="alert" className="text-[13px] text-neg border-l-2 border-neg pl-3">
          {error}
        </p>
      )}

      {!backtestData && !loading && !error && (
        <p className="text-[13px] text-ink-3">No results yet. Configure the run above and start the benchmark.</p>
      )}

      {backtestData && backtestData.errors.length > 0 && (
        <div role="alert" className="border border-warn/40 rounded-[4px] px-5 py-4">
          <p className="text-[13px] text-warn font-medium mb-1.5">
            {backtestData.errors.length} {backtestData.errors.length === 1 ? 'policy was' : 'policies were'} not evaluated
          </p>
          <ul className="text-[12.5px] text-ink-2 space-y-1">
            {backtestData.errors.map((e) => (
              <li key={e.policy}>
                <span className="text-ink">{e.policy}</span> — {e.detail}
              </li>
            ))}
          </ul>
        </div>
      )}

      {backtestData && best && (
        <>
          <Panel>
            <div className="px-5 py-5 grid grid-cols-1 sm:grid-cols-3 gap-y-4 sm:divide-x divide-line">
              <div className="sm:pr-5">
                <p className="label">Lowest mean shortfall</p>
                <p className="text-[18px] font-semibold text-ink mt-1.5">{best.policy}</p>
                <p className="num text-[13px] text-ink-2 mt-0.5">
                  {best.implementation_shortfall_bps.toFixed(2)} bps · {fmtUsd(best.execution_cost)}
                </p>
              </div>
              <div className="sm:px-5">
                <p className="label">Best baseline</p>
                <p className="text-[18px] font-semibold text-ink mt-1.5">{bestBaseline?.policy ?? '—'}</p>
                <p className="num text-[13px] text-ink-2 mt-0.5">
                  {bestBaseline ? `${bestBaseline.implementation_shortfall_bps.toFixed(2)} bps` : ''}
                </p>
              </div>
              <div className="sm:pl-5">
                <p className="label">Best RL agent vs TWAP (paired)</p>
                {bestRl ? (
                  <>
                    <p className="text-[18px] font-semibold text-ink mt-1.5">{bestRl.policy}</p>
                    <p className="num text-[13px] mt-0.5">
                      <VsTwap r={bestRl} />
                    </p>
                  </>
                ) : (
                  <p className="text-ink-3 mt-1.5">No trained RL agent available</p>
                )}
              </div>
            </div>
          </Panel>

          <Panel>
            <PanelHeader
              title="Results"
              note={`Mean over ${backtestData.n_windows} out-of-sample window${backtestData.n_windows === 1 ? '' : 's'} · ${backtestData.symbol} ${backtestData.side.toLowerCase()} ${fmtInt(backtestData.quantity)} · ${backtestData.scenario.replace(/_/g, ' ')} · ranked by shortfall, lowest first`}
              right={
                <span className="flex items-center gap-4 text-[12px] text-ink-3">
                  <span className="flex items-center gap-1.5"><span className="h-2 w-2 bg-ink-3 inline-block" />Baseline</span>
                  <span className="flex items-center gap-1.5"><span className="h-2 w-2 bg-info inline-block" />RL</span>
                  <span className="flex items-center gap-1.5"><span className="h-2 w-2 bg-accent inline-block" />Regime-aware</span>
                </span>
              }
            />
            <div className="overflow-x-auto">
              <table className="w-full text-[13px] min-w-[860px]">
                <thead>
                  <tr className="label text-left border-b border-line">
                    <th className="px-5 py-2.5 font-medium w-10">#</th>
                    <th className="py-2.5 font-medium">Policy</th>
                    <th className="py-2.5 font-medium w-[30%]">Shortfall (bps)</th>
                    <th className="py-2.5 font-medium text-right">vs TWAP, 95% CI</th>
                    <th className="py-2.5 font-medium text-right">Cost</th>
                    <th className="px-5 py-2.5 font-medium text-right">Fill</th>
                  </tr>
                </thead>
                <tbody>
                  {ranked.map((r, i) => {
                    const barColor = isRegimeAware(r.policy) ? 'var(--accent)' : isBaseline(r.policy) ? 'var(--ink-3)' : 'var(--info)';
                    const valPct = ((r.implementation_shortfall_bps - lo) / span) * 100;
                    const left = Math.min(zeroPct, valPct);
                    const w = Math.abs(valPct - zeroPct);
                    return (
                      <tr key={r.policy} className="border-b border-line last:border-b-0 hover:bg-raised/50">
                        <td className="px-5 py-3 num text-ink-3">{i + 1}</td>
                        <td className="py-3 text-ink">{r.policy}</td>
                        <td className="py-3 pr-6">
                          <div className="flex items-center gap-3">
                            <span className="num w-14 text-right text-ink">{r.implementation_shortfall_bps.toFixed(2)}</span>
                            <span className="flex-1 h-2 bg-raised relative">
                              <span className="absolute inset-y-0 w-px bg-line-strong" style={{ left: `${zeroPct}%` }} />
                              <span className="absolute inset-y-0" style={{ left: `${left}%`, width: `${w}%`, background: barColor }} />
                            </span>
                          </div>
                          <p className="num text-[11.5px] text-ink-3 mt-0.5 pl-[68px]">
                            95% CI [{r.ci_low.toFixed(1)}, {r.ci_high.toFixed(1)}]
                          </p>
                        </td>
                        <td className="py-3 num text-right">
                          <VsTwap r={r} />
                        </td>
                        <td className="py-3 num text-right text-ink-2">{fmtUsd(r.execution_cost)}</td>
                        <td className="px-5 py-3 num text-right text-ink-2">{(r.completion_rate * 100).toFixed(1)}%</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            <p className="px-5 py-3 border-t border-line text-[12px] text-ink-3">
              Negative shortfall is cheaper. “n.s.” means the paired confidence interval includes zero, so the difference is
              not distinguishable from noise on these windows. The Research tab reports the full multi-seed experiment.
            </p>
          </Panel>
        </>
      )}
    </div>
  );
}
