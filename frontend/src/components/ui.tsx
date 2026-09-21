import React from 'react';

/** Semantic colour for a regime label. Colour carries meaning here, not decoration. */
export function regimeColor(label: string): string {
  switch (label.toLowerCase()) {
    case 'low volatility':
      return 'var(--pos)';
    case 'high volatility':
      return 'var(--warn)';
    case 'stress':
      return 'var(--neg)';
    default:
      return 'var(--info)';
  }
}

/** Locale is pinned so server and client renders always match. */
export const fmtInt = (v: number) => v.toLocaleString('en-US', { maximumFractionDigits: 0 });

export const fmtUsd = (v: number, digits = 2) =>
  `${v < 0 ? '-' : ''}$${Math.abs(v).toLocaleString('en-US', { minimumFractionDigits: digits, maximumFractionDigits: digits })}`;

/** Flat bordered surface. */
export function Panel({
  children,
  className = '',
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return <section className={`border border-line bg-panel rounded-[4px] ${className}`}>{children}</section>;
}

export function PanelHeader({
  title,
  note,
  right,
}: {
  title: string;
  note?: string;
  right?: React.ReactNode;
}) {
  return (
    <div className="flex flex-wrap items-end justify-between gap-3 px-5 py-4 border-b border-line">
      <div>
        <h2 className="text-[15px] font-semibold text-ink leading-tight">{title}</h2>
        {note && <p className="text-[12.5px] text-ink-3 mt-1 max-w-2xl">{note}</p>}
      </div>
      {right}
    </div>
  );
}

export function PageTitle({
  eyebrow,
  title,
  children,
}: {
  eyebrow: string;
  title: string;
  children?: React.ReactNode;
}) {
  return (
    <div className="flex flex-wrap items-end justify-between gap-4 mb-6">
      <div>
        <p className="label mb-1.5">{eyebrow}</p>
        <h1 className="text-2xl font-semibold tracking-tight text-ink">{title}</h1>
      </div>
      {children}
    </div>
  );
}

export function EmptyState({
  title,
  body,
  action,
}: {
  title: string;
  body: string;
  action?: React.ReactNode;
}) {
  return (
    <Panel className="max-w-xl">
      <div className="px-6 py-8">
        <h2 className="text-[15px] font-semibold text-ink">{title}</h2>
        <p className="text-[13px] text-ink-2 mt-1.5 mb-5">{body}</p>
        {action}
      </div>
    </Panel>
  );
}

/** A ruled row of key figures. Cells share hairlines instead of being separate cards. */
export function StatStrip({
  items,
  className = '',
}: {
  items: { label: string; value: string; sub?: string; tone?: 'pos' | 'neg' | 'warn' }[];
  className?: string;
}) {
  const toneClass = { pos: 'text-pos', neg: 'text-neg', warn: 'text-warn' };
  return (
    <div
      className={`grid grid-cols-2 lg:grid-cols-4 border border-line bg-panel rounded-[4px] divide-line ${className}`}
    >
      {items.map((it, i) => (
        <div
          key={it.label}
          className={`px-5 py-4 border-line ${i % 2 === 1 ? 'border-l' : ''} ${i > 0 ? 'lg:border-l' : ''} ${
            i >= 2 ? 'border-t lg:border-t-0' : ''
          }`}
        >
          <p className="label">{it.label}</p>
          <p className={`num text-[22px] font-medium mt-1.5 ${it.tone ? toneClass[it.tone] : 'text-ink'}`}>{it.value}</p>
          {it.sub && <p className="num text-[12px] text-ink-3 mt-0.5">{it.sub}</p>}
        </div>
      ))}
    </div>
  );
}

/** Definition-list table: label on the left, figure on the right. */
export function KeyValueTable({ rows }: { rows: { k: string; v: React.ReactNode }[] }) {
  return (
    <dl className="text-[13px]">
      {rows.map((r) => (
        <div key={r.k} className="flex items-baseline justify-between gap-4 py-2 border-b border-line last:border-b-0">
          <dt className="text-ink-2">{r.k}</dt>
          <dd className="num text-ink text-right">{r.v}</dd>
        </div>
      ))}
    </dl>
  );
}

/* ---- Recharts shared styling ---- */

export const AXIS_TICK = { fontSize: 11, fill: '#6b727a', fontFamily: 'var(--font-mono)' };
export const GRID_STROKE = '#1c2023';
export const AXIS_LINE = { stroke: '#30353a' };

type TooltipPayload = { name?: string; value?: number | string; color?: string; dataKey?: string | number };

export function ChartTooltip({
  active,
  payload,
  label,
  labelPrefix = '',
  format,
}: {
  active?: boolean;
  payload?: TooltipPayload[];
  label?: string | number;
  labelPrefix?: string;
  format?: (v: number, key: string) => string;
}) {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-raised border border-line-strong rounded-[3px] px-2.5 py-1.5 text-[12px]">
      <p className="text-ink-3 mb-0.5">
        {labelPrefix}
        {label}
      </p>
      {payload.map((p) => (
        <p key={String(p.dataKey)} className="num text-ink">
          {format && typeof p.value === 'number' ? format(p.value, String(p.dataKey)) : p.value}
        </p>
      ))}
    </div>
  );
}
