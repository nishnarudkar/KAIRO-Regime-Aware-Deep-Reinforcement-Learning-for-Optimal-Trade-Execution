'use client';

import React from 'react';
import { ExecutionRecord } from '../lib/api';
import { Panel, PanelHeader, PageTitle, EmptyState, StatStrip, KeyValueTable, fmtUsd, fmtInt } from './ui';

interface ExecutionMonitorTabProps {
  executionRecord: ExecutionRecord | null;
  onNavigateToAnalytics: () => void;
  onNavigateToNew: () => void;
}

const titleCase = (s: string) =>
  s
    .replace(/_/g, ' ')
    .replace(/^\w/, (c) => c.toUpperCase());

export function ExecutionMonitorTab({
  executionRecord,
  onNavigateToAnalytics,
  onNavigateToNew,
}: ExecutionMonitorTabProps) {
  if (!executionRecord) {
    return (
      <div>
        <PageTitle eyebrow="Execution" title="Monitor" />
        <EmptyState
          title="No execution yet"
          body="Run a simulation from the order ticket and its fill progress and cost figures will appear here."
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
  const fillPct = Math.min(100, metrics.completion_rate * 100);
  const isComplete = metrics.completion_rate >= 0.9999;

  return (
    <div className="space-y-6">
      <PageTitle eyebrow={`Execution ${executionRecord.execution_id.slice(0, 8)}`} title="Monitor">
        <button onClick={onNavigateToAnalytics} className="btn btn-ghost">
          View analytics →
        </button>
      </PageTitle>

      <Panel>
        <div className="px-5 py-5 flex flex-wrap items-start justify-between gap-4">
          <div>
            <p className="text-[22px] font-semibold text-ink leading-tight">
              <span className={executionRecord.side.toUpperCase() === 'BUY' ? 'text-pos' : 'text-neg'}>
                {titleCase(executionRecord.side.toLowerCase())}
              </span>{' '}
              <span className="tabular-nums">{fmtInt(executionRecord.target_inventory)}</span> {executionRecord.symbol}
            </p>
            <p className="text-[13px] text-ink-2 mt-1">
              {executionRecord.policy} · {titleCase(executionRecord.scenario)}
            </p>
          </div>
          <div className="text-right">
            <p className="label">Status</p>
            <p className="text-[13px] text-ink mt-1">{titleCase(executionRecord.status)}</p>
          </div>
        </div>

        <div className="px-5 pb-5">
          <div className="flex items-baseline justify-between mb-2">
            <span className="label">Filled</span>
            <span className="num text-[13px] text-ink">
              {fmtInt(executionRecord.executed_inventory)} / {fmtInt(executionRecord.target_inventory)}
              <span className="text-ink-3"> · {fillPct.toFixed(1)}%</span>
            </span>
          </div>
          <div
            className="h-2 bg-raised rounded-[1px] overflow-hidden"
            role="progressbar"
            aria-valuenow={Math.round(fillPct)}
            aria-valuemin={0}
            aria-valuemax={100}
            aria-label="Fill progress"
          >
            <div
              className={`h-full transition-[width] duration-500 ${isComplete ? 'bg-pos' : 'bg-warn'}`}
              style={{ width: `${fillPct}%` }}
            />
          </div>
          {!isComplete && (
            <p className="text-[12.5px] text-ink-3 mt-2">
              <span className="num">{fmtInt(executionRecord.remaining_inventory)}</span> shares were left unfilled
              at the end of the horizon.
              {metrics.terminal_penalty > 0 && (
                <> A terminal penalty of <span className="num">{fmtUsd(metrics.terminal_penalty)}</span> applies.</>
              )}
            </p>
          )}
        </div>
      </Panel>

      <StatStrip
        items={[
          {
            label: 'Implementation shortfall',
            value: `${metrics.implementation_shortfall_bps.toFixed(2)} bps`,
            sub: fmtUsd(metrics.implementation_shortfall),
          },
          {
            label: 'Avg. fill price',
            value: fmtUsd(metrics.average_execution_price),
            sub: `Arrival ${fmtUsd(metrics.arrival_price)}`,
          },
          {
            label: 'Slippage vs VWAP',
            value: `${metrics.vwap_slippage_bps.toFixed(2)} bps`,
            sub: `Market VWAP ${fmtUsd(metrics.market_vwap_price)}`,
          },
          {
            label: 'Execution cost',
            value: fmtUsd(metrics.execution_cost),
            sub: `Impact ${fmtUsd(metrics.market_impact_cost)}`,
          },
        ]}
      />

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Panel>
          <PanelHeader title="Order" />
          <div className="px-5 py-1">
            <KeyValueTable
              rows={[
                { k: 'Execution ID', v: <span className="text-[12px]">{executionRecord.execution_id}</span> },
                { k: 'Submitted', v: new Date(executionRecord.timestamp).toLocaleString('en-US') },
                { k: 'Target', v: fmtInt(executionRecord.target_inventory) },
                { k: 'Executed', v: fmtInt(executionRecord.executed_inventory) },
                { k: 'Remaining', v: fmtInt(executionRecord.remaining_inventory) },
              ]}
            />
          </div>
        </Panel>
        <Panel>
          <PanelHeader title="Cost components" />
          <div className="px-5 py-1">
            <KeyValueTable
              rows={[
                { k: 'Market impact', v: fmtUsd(metrics.market_impact_cost) },
                { k: 'Transaction fees', v: fmtUsd(metrics.total_transaction_fees) },
                { k: 'Terminal penalty', v: fmtUsd(metrics.terminal_penalty) },
                { k: 'Total execution cost', v: fmtUsd(metrics.execution_cost) },
                { k: 'Arrival price', v: fmtUsd(metrics.arrival_price) },
              ]}
            />
          </div>
        </Panel>
      </div>
    </div>
  );
}
