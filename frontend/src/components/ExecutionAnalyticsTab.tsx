'use client';

import React, { useEffect, useState } from 'react';
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
} from 'recharts';
import { ExecutionRecord, ExecutionTrajectory, getExecutionTrajectory } from '../lib/api';
import {
  Panel,
  PanelHeader,
  PageTitle,
  EmptyState,
  StatStrip,
  ChartTooltip,
  regimeColor,
  fmtUsd,
  fmtInt,
  AXIS_TICK,
  AXIS_LINE,
  GRID_STROKE,
} from './ui';

interface ExecutionAnalyticsTabProps {
  executionRecord: ExecutionRecord | null;
  onNavigateToNew: () => void;
}

const REGIME_NAMES = ['Low Volatility', 'Normal', 'High Volatility', 'Stress'];

export function ExecutionAnalyticsTab({ executionRecord, onNavigateToNew }: ExecutionAnalyticsTabProps) {
  // Keyed by execution id, so "loading" is derived instead of set synchronously in the effect.
  const [loaded, setLoaded] = useState<{ id: string; data: ExecutionTrajectory | null } | null>(null);
  const executionId = executionRecord?.execution_id ?? null;

  useEffect(() => {
    if (!executionId) return;
    let cancelled = false;
    getExecutionTrajectory(executionId)
      .then((data) => !cancelled && setLoaded({ id: executionId, data }))
      .catch(() => !cancelled && setLoaded({ id: executionId, data: null }));
    return () => {
      cancelled = true;
    };
  }, [executionId]);

  const loading = executionId !== null && loaded?.id !== executionId;
  const trajectory = loaded?.id === executionId ? loaded?.data ?? null : null;

  if (!executionRecord) {
    return (
      <div>
        <PageTitle eyebrow="Execution" title="Analytics" />
        <EmptyState
          title="No analytics yet"
          body="Run a simulation to see the inventory, price and action trajectories with a cost breakdown."
          action={
            <button onClick={onNavigateToNew} className="btn btn-ghost">
              Open order ticket
            </button>
          }
        />
      </div>
    );
  }

  const { metrics } = executionRecord;

  const trajectoryData = (trajectory?.inventory_trajectory || []).map((inv, idx) => ({
    step: idx,
    inventory: inv,
    price: trajectory?.price_trajectory[idx] || metrics.arrival_price,
  }));

  const regimes = trajectory?.regime_trajectory ?? [];

  const hasActions = Object.values(metrics.action_counts).some((c) => c > 0);
  const actionDistributionData = [
    { name: '0%', count: metrics.action_counts['0'] || 0 },
    { name: '10%', count: metrics.action_counts['1'] || 0 },
    { name: '25%', count: metrics.action_counts['2'] || 0 },
    { name: '50%', count: metrics.action_counts['3'] || 0 },
  ];

  const chartMargin = { top: 8, right: 8, bottom: 0, left: 0 };

  return (
    <div className="space-y-6">
      <PageTitle eyebrow={`${executionRecord.symbol} · ${executionRecord.policy}`} title="Analytics" />

      <StatStrip
        items={[
          {
            label: 'Implementation shortfall',
            value: `${metrics.implementation_shortfall_bps.toFixed(2)} bps`,
            sub: fmtUsd(metrics.implementation_shortfall),
          },
          {
            label: 'Fill rate',
            value: `${(metrics.completion_rate * 100).toFixed(1)}%`,
            sub: `${fmtInt(executionRecord.executed_inventory)} shares`,
            tone: metrics.completion_rate >= 0.9999 ? 'pos' : 'warn',
          },
          {
            label: 'Market impact',
            value: fmtUsd(metrics.market_impact_cost),
            sub: `Fees ${fmtUsd(metrics.total_transaction_fees)}`,
          },
          {
            label: 'Slippage vs VWAP',
            value: `${metrics.vwap_slippage_bps.toFixed(2)} bps`,
            sub: `Avg. fill ${fmtUsd(metrics.average_execution_price)}`,
          },
        ]}
      />

      {loading && <p className="text-[13px] text-ink-3">Loading trajectory…</p>}
      {!loading && !trajectory && (
        <p className="text-[13px] text-ink-3 border-l-2 border-line-strong pl-3">
          Trajectory data is unavailable for this execution, so the time-series charts are empty.
        </p>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Panel>
          <PanelHeader title="Remaining inventory" note="Shares still to execute at each step" />
          <div className="h-64 p-4 pl-1">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={trajectoryData} margin={chartMargin}>
                <CartesianGrid stroke={GRID_STROKE} vertical={false} />
                <XAxis dataKey="step" minTickGap={24} tick={AXIS_TICK} axisLine={AXIS_LINE} tickLine={false} />
                <YAxis tick={AXIS_TICK} axisLine={false} tickLine={false} width={56} tickFormatter={(v) => fmtInt(v)} />
                <Tooltip
                  cursor={{ stroke: '#414850' }}
                  content={<ChartTooltip labelPrefix="Step " format={(v) => `${fmtInt(v)} shares`} />}
                />
                <Area type="stepAfter" dataKey="inventory" stroke="#d8b46a" strokeWidth={1.5} fill="#d8b46a" fillOpacity={0.08} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </Panel>

        <Panel>
          <PanelHeader title="Market price" note="Asset price across the execution horizon" />
          <div className="h-64 p-4 pl-1">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={trajectoryData} margin={chartMargin}>
                <CartesianGrid stroke={GRID_STROKE} vertical={false} />
                <XAxis dataKey="step" minTickGap={24} tick={AXIS_TICK} axisLine={AXIS_LINE} tickLine={false} />
                <YAxis domain={['auto', 'auto']} tick={AXIS_TICK} axisLine={false} tickLine={false} width={56} tickFormatter={(v) => v.toFixed(2)} />
                <Tooltip
                  cursor={{ stroke: '#414850' }}
                  content={<ChartTooltip labelPrefix="Step " format={(v) => fmtUsd(v)} />}
                />
                <Line type="monotone" dataKey="price" stroke="#e4e6e8" strokeWidth={1.5} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
          {regimes.length > 0 && (
            <div className="px-5 pb-4 -mt-1">
              <p className="label mb-1.5">Regime by step</p>
              <div className="flex h-2.5 gap-px" aria-hidden>
                {regimes.map((r, i) => (
                  <span
                    key={i}
                    className="flex-1"
                    title={`Step ${i}: ${REGIME_NAMES[r] ?? `Regime ${r}`}`}
                    style={{ background: regimeColor(REGIME_NAMES[r] ?? '') }}
                  />
                ))}
              </div>
              <div className="flex flex-wrap gap-x-4 gap-y-1 mt-2 text-[12px] text-ink-3">
                {REGIME_NAMES.map((n) => (
                  <span key={n} className="flex items-center gap-1.5">
                    <span className="h-2 w-2 inline-block" style={{ background: regimeColor(n) }} />
                    {n}
                  </span>
                ))}
              </div>
            </div>
          )}
        </Panel>
      </div>

      {hasActions && (
        <Panel>
          <PanelHeader title="Action distribution" note="How often the policy chose each fraction of remaining inventory" />
          <div className="h-56 p-4 pl-1">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={actionDistributionData} margin={chartMargin}>
                <CartesianGrid stroke={GRID_STROKE} vertical={false} />
                <XAxis dataKey="name" tick={AXIS_TICK} axisLine={AXIS_LINE} tickLine={false} />
                <YAxis tick={AXIS_TICK} axisLine={false} tickLine={false} width={40} allowDecimals={false} />
                <Tooltip cursor={{ fill: '#16181a' }} content={<ChartTooltip labelPrefix="Fill " format={(v) => `${v} steps`} />} />
                <Bar dataKey="count" fill="#7d9bb8" radius={[1, 1, 0, 0]} maxBarSize={56} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Panel>
      )}
    </div>
  );
}
