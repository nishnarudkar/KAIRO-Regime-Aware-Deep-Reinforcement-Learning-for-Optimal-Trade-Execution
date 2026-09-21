'use client';

import React from 'react';
import { Activity, Clock, ShieldCheck, DollarSign, Layers, PieChart, ArrowUpRight, Sparkles } from 'lucide-react';
import { ExecutionRecord } from '../lib/api';

interface ExecutionMonitorTabProps {
  executionRecord: ExecutionRecord | null;
  onNavigateToAnalytics: () => void;
}

export function ExecutionMonitorTab({
  executionRecord,
  onNavigateToAnalytics,
}: ExecutionMonitorTabProps) {
  if (!executionRecord) {
    return (
      <div className="max-w-4xl mx-auto bg-slate-900/60 border border-slate-800 rounded-2xl p-12 text-center">
        <Activity className="h-12 w-12 text-slate-600 mx-auto mb-4 animate-pulse" />
        <h3 className="text-lg font-bold text-white mb-2">No Execution Active</h3>
        <p className="text-sm text-slate-400 mb-6">Launch a new simulation from the New Execution tab to monitor live order progress.</p>
      </div>
    );
  }

  const { metrics } = executionRecord;
  const fillPercentage = (metrics.completion_rate * 100).toFixed(1);

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      {/* Top Banner Status Card */}
      <div className="bg-gradient-to-r from-slate-900 via-indigo-950/40 to-slate-900 border border-indigo-500/20 rounded-2xl p-6 shadow-xl relative overflow-hidden">
        <div className="absolute top-0 right-0 p-8 opacity-10 pointer-events-none">
          <Activity className="h-48 w-48 text-indigo-400" />
        </div>

        <div className="relative z-10 flex flex-wrap items-center justify-between gap-4 mb-6">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <span className="text-xs px-2.5 py-0.5 rounded-full bg-emerald-500/20 text-emerald-400 font-semibold border border-emerald-500/30">
                ACTIVE EXECUTION
              </span>
              <span className="text-xs text-slate-400 font-mono">ID: {executionRecord.execution_id.slice(0, 8)}...</span>
            </div>
            <h2 className="text-2xl font-extrabold text-white">
              {executionRecord.side} {executionRecord.target_inventory.toLocaleString()} {executionRecord.symbol}
            </h2>
            <p className="text-xs text-slate-400">
              Policy: <span className="text-indigo-300 font-semibold">{executionRecord.policy}</span> • Scenario: <span className="text-slate-300">{executionRecord.scenario}</span>
            </p>
          </div>

          <button
            onClick={onNavigateToAnalytics}
            className="px-4 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white font-medium text-xs flex items-center gap-2 transition-all shadow-md shadow-indigo-500/20"
          >
            <span>View Full Analytics</span>
            <ArrowUpRight className="h-3.5 w-3.5" />
          </button>
        </div>

        {/* Progress Bar */}
        <div className="space-y-2">
          <div className="flex justify-between text-xs font-semibold">
            <span className="text-slate-300">Execution Fill Progress</span>
            <span className="text-emerald-400">{fillPercentage}% Completed</span>
          </div>
          <div className="w-full bg-slate-950 rounded-full h-3 border border-slate-800 p-0.5 overflow-hidden">
            <div
              className="bg-gradient-to-r from-indigo-500 via-blue-500 to-emerald-400 h-full rounded-full transition-all duration-500 shadow-sm shadow-emerald-500/30"
              style={{ width: `${Math.min(100, metrics.completion_rate * 100)}%` }}
            />
          </div>
        </div>
      </div>

      {/* Monitor Metrics Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-4">
          <div className="flex items-center justify-between text-xs text-slate-400 mb-1">
            <span>Implementation Shortfall</span>
            <DollarSign className="h-3.5 w-3.5 text-indigo-400" />
          </div>
          <p className="text-xl font-bold text-white">{metrics.implementation_shortfall_bps.toFixed(2)} bps</p>
          <p className="text-xs text-slate-400 mt-1">${metrics.implementation_shortfall.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })} total</p>
        </div>

        <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-4">
          <div className="flex items-center justify-between text-xs text-slate-400 mb-1">
            <span>Avg Fill Price</span>
            <Activity className="h-3.5 w-3.5 text-blue-400" />
          </div>
          <p className="text-xl font-bold text-white">${metrics.average_execution_price.toFixed(2)}</p>
          <p className="text-xs text-slate-400 mt-1">Arrival: ${metrics.arrival_price.toFixed(2)}</p>
        </div>

        <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-4">
          <div className="flex items-center justify-between text-xs text-slate-400 mb-1">
            <span>Market VWAP Slippage</span>
            <Clock className="h-3.5 w-3.5 text-emerald-400" />
          </div>
          <p className="text-xl font-bold text-white">{metrics.vwap_slippage_bps.toFixed(2)} bps</p>
          <p className="text-xs text-slate-400 mt-1">VWAP: ${metrics.market_vwap_price.toFixed(2)}</p>
        </div>

        <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-4">
          <div className="flex items-center justify-between text-xs text-slate-400 mb-1">
            <span>Total Execution Cost</span>
            <PieChart className="h-3.5 w-3.5 text-amber-400" />
          </div>
          <p className="text-xl font-bold text-white">${metrics.execution_cost.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</p>
          <p className="text-xs text-slate-400 mt-1">Impact: ${metrics.market_impact_cost.toFixed(2)}</p>
        </div>
      </div>
    </div>
  );
}
