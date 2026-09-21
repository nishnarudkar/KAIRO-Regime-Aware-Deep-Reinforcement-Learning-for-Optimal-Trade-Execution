'use client';

import React from 'react';
import { Activity, ShieldCheck, Wifi, WifiOff, BarChart3, Sliders, Layers, Sparkles, Database } from 'lucide-react';

interface HeaderProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
  isBackendConnected: boolean;
  currentRegimeLabel?: string;
}

export function Header({
  activeTab,
  setActiveTab,
  isBackendConnected,
  currentRegimeLabel = 'Normal',
}: HeaderProps) {
  const tabs = [
    { id: 'new', label: 'New Execution', icon: Sliders },
    { id: 'monitor', label: 'Execution Monitor', icon: Activity },
    { id: 'analytics', label: 'Execution Analytics', icon: BarChart3 },
    { id: 'comparison', label: 'Strategy Comparison', icon: Layers },
    { id: 'research', label: 'Research & Regimes', icon: Database },
  ];

  const getRegimeColor = (label: string) => {
    switch (label.toLowerCase()) {
      case 'low volatility': return 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20';
      case 'high volatility': return 'bg-amber-500/10 text-amber-400 border-amber-500/20';
      case 'stress': return 'bg-rose-500/10 text-rose-400 border-rose-500/20';
      default: return 'bg-blue-500/10 text-blue-400 border-blue-500/20';
    }
  };

  return (
    <header className="border-b border-slate-800 bg-slate-950/80 backdrop-blur-md sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Brand Header */}
          <div className="flex items-center gap-3">
            <div className="h-10 w-10 rounded-xl bg-gradient-to-tr from-indigo-600 via-blue-600 to-emerald-400 p-[1px] shadow-lg shadow-indigo-500/20">
              <div className="h-full w-full bg-slate-950 rounded-[11px] flex items-center justify-center">
                <Sparkles className="h-5 w-5 text-indigo-400" />
              </div>
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-bold tracking-tight text-white text-lg">KAIRO</span>
                <span className="text-xs px-2 py-0.5 rounded-full bg-indigo-500/10 text-indigo-400 border border-indigo-500/20 font-medium">v1.0 DRL</span>
              </div>
              <p className="text-xs text-slate-400">Adaptive Execution Intelligence</p>
            </div>
          </div>

          {/* Center Navigation Tabs */}
          <nav className="hidden md:flex items-center gap-1 bg-slate-900/60 p-1 rounded-xl border border-slate-800">
            {tabs.map((tab) => {
              const Icon = tab.icon;
              const isActive = activeTab === tab.id;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-medium transition-all ${
                    isActive
                      ? 'bg-gradient-to-r from-indigo-600 to-blue-600 text-white shadow-md shadow-indigo-500/20'
                      : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
                  }`}
                >
                  <Icon className="h-3.5 w-3.5" />
                  {tab.label}
                </button>
              );
            })}
          </nav>

          {/* Right Status Badges */}
          <div className="flex items-center gap-3">
            {/* Regime Badge */}
            <div className={`hidden sm:flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium border ${getRegimeColor(currentRegimeLabel)}`}>
              <span className="h-1.5 w-1.5 rounded-full bg-current animate-pulse" />
              <span>Regime: {currentRegimeLabel}</span>
            </div>

            {/* Backend Connection Indicator */}
            <div className={`flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium border ${
              isBackendConnected
                ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20'
                : 'bg-rose-500/10 text-rose-400 border-rose-500/20'
            }`}>
              {isBackendConnected ? (
                <>
                  <Wifi className="h-3.5 w-3.5" />
                  <span>API Online</span>
                </>
              ) : (
                <>
                  <WifiOff className="h-3.5 w-3.5" />
                  <span>API Offline</span>
                </>
              )}
            </div>
          </div>
        </div>

        {/* Mobile Tab Navigation */}
        <div className="md:hidden flex items-center justify-between overflow-x-auto py-2 border-t border-slate-800/50">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-1 px-3 py-1 rounded-md text-xs whitespace-nowrap ${
                  isActive ? 'bg-indigo-600 text-white font-medium' : 'text-slate-400'
                }`}
              >
                <Icon className="h-3 w-3" />
                {tab.label}
              </button>
            );
          })}
        </div>
      </div>
    </header>
  );
}
