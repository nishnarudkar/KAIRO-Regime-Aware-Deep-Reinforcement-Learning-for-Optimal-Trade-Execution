'use client';

import React, { useEffect, useState } from 'react';
import { BarChart3, TrendingDown, DollarSign, PieChart, ShieldCheck, Activity } from 'lucide-react';
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

interface ExecutionAnalyticsTabProps {
  executionRecord: ExecutionRecord | null;
}

export function ExecutionAnalyticsTab({ executionRecord }: ExecutionAnalyticsTabProps) {
  const [trajectory, setTrajectory] = useState<ExecutionTrajectory | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (executionRecord?.execution_id) {
      setLoading(true);
      getExecutionTrajectory(executionRecord.execution_id)
        .then(setTrajectory)
        .catch(() => setTrajectory(null))
        .finally(() => setLoading(false));
    }
  }, [executionRecord]);

  if (!executionRecord) {
    return (
      <div className="max-w-4xl mx-auto bg-slate-900/60 border border-slate-800 rounded-2xl p-12 text-center">
        <BarChart3 className="h-12 w-12 text-slate-600 mx-auto mb-4" />
        <h3 className="text-lg font-bold text-white mb-2">No Analytics Data Available</h3>
        <p className="text-sm text-slate-400">Run a simulation to generate detailed implementation shortfall and trajectory charts.</p>
      </div>
    );
  }

  const { metrics } = executionRecord;

  // Format Recharts data
  const trajectoryData = (trajectory?.inventory_trajectory || []).map((inv, idx) => ({
    step: idx,
    inventory: inv,
    price: trajectory?.price_trajectory[idx] || metrics.arrival_price,
    action: trajectory?.action_trajectory[idx] || 0,
  }));

  const actionDistributionData = [
    { name: '0% Fill', count: metrics.action_counts['0'] || 0, fill: '#64748b' },
    { name: '10% Fill', count: metrics.action_counts['1'] || 0, fill: '#3b82f6' },
    { name: '25% Fill', count: metrics.action_counts['2'] || 0, fill: '#6366f1' },
    { name: '50% Fill', count: metrics.action_counts['3'] || 0, fill: '#10b981' },
  ];

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      {/* Header Summary */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6 shadow-xl">
        <div className="flex flex-wrap items-center justify-between gap-4 mb-6">
          <div>
            <h2 className="text-xl font-bold text-white">Execution Analytics & Cost Breakdown</h2>
            <p className="text-xs text-slate-400">
              Policy: <span className="text-indigo-400 font-semibold">{executionRecord.policy}</span> • Symbol: <span className="text-white">{executionRecord.symbol}</span> • Scenario: <span className="text-slate-300">{executionRecord.scenario}</span>
            </p>
          </div>
          <div className="text-right">
            <p className="text-xs text-slate-400">Implementation Shortfall</p>
            <p className="text-2xl font-black text-indigo-400">{metrics.implementation_shortfall_bps.toFixed(2)} bps</p>
          </div>
        </div>

        {/* Detailed Grid */}
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
          <div className="bg-slate-950/60 border border-slate-800 p-3 rounded-xl">
            <span className="text-[11px] text-slate-400 block">Fill Rate</span>
            <span className="text-base font-bold text-emerald-400">{(metrics.completion_rate * 100).toFixed(1)}%</span>
          </div>

          <div className="bg-slate-950/60 border border-slate-800 p-3 rounded-xl">
            <span className="text-[11px] text-slate-400 block">Executed Shares</span>
            <span className="text-base font-bold text-white">{executionRecord.executed_inventory.toLocaleString()}</span>
          </div>

          <div className="bg-slate-950/60 border border-slate-800 p-3 rounded-xl">
            <span className="text-[11px] text-slate-400 block">Avg Fill Price</span>
            <span className="text-base font-bold text-white">${metrics.average_execution_price.toFixed(2)}</span>
          </div>

          <div className="bg-slate-950/60 border border-slate-800 p-3 rounded-xl">
            <span className="text-[11px] text-slate-400 block">Market Impact</span>
            <span className="text-base font-bold text-amber-400">${metrics.market_impact_cost.toFixed(2)}</span>
          </div>

          <div className="bg-slate-950/60 border border-slate-800 p-3 rounded-xl">
            <span className="text-[11px] text-slate-400 block">Transaction Fees</span>
            <span className="text-base font-bold text-white">${metrics.total_transaction_fees.toFixed(2)}</span>
          </div>

          <div className="bg-slate-950/60 border border-slate-800 p-3 rounded-xl">
            <span className="text-[11px] text-slate-400 block">VWAP Slippage</span>
            <span className="text-base font-bold text-blue-400">{metrics.vwap_slippage_bps.toFixed(2)} bps</span>
          </div>
        </div>
      </div>

      {/* Trajectory Charts */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Inventory Decay Chart */}
        <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-5 shadow-xl">
          <h3 className="text-sm font-bold text-white mb-1">Remaining Inventory Decay Trajectory</h3>
          <p className="text-xs text-slate-400 mb-4">Remaining shares ($I_t$) over time steps</p>

          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={trajectoryData}>
                <defs>
                  <linearGradient id="invGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#6366f1" stopOpacity={0.4} />
                    <stop offset="95%" stopColor="#6366f1" stopOpacity={0.0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="step" stroke="#64748b" tick={{ fontSize: 11 }} />
                <YAxis stroke="#64748b" tick={{ fontSize: 11 }} />
                <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155' }} />
                <Area type="monotone" dataKey="inventory" stroke="#6366f1" fillOpacity={1} fill="url(#invGrad)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Price Trajectory Chart */}
        <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-5 shadow-xl">
          <h3 className="text-sm font-bold text-white mb-1">Market Price Trajectory</h3>
          <p className="text-xs text-slate-400 mb-4">Asset price movements over the execution horizon</p>

          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={trajectoryData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="step" stroke="#64748b" tick={{ fontSize: 11 }} />
                <YAxis domain={['auto', 'auto']} stroke="#64748b" tick={{ fontSize: 11 }} />
                <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155' }} />
                <Line type="monotone" dataKey="price" stroke="#10b981" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Action Distribution Chart */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-5 shadow-xl">
        <h3 className="text-sm font-bold text-white mb-1">RL Action Choice Distribution</h3>
        <p className="text-xs text-slate-400 mb-4">Frequency of inventory fill percentage actions selected by policy</p>

        <div className="h-56 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={actionDistributionData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="name" stroke="#64748b" tick={{ fontSize: 11 }} />
              <YAxis stroke="#64748b" tick={{ fontSize: 11 }} />
              <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155' }} />
              <Bar dataKey="count" fill="#3b82f6" radius={[6, 6, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
