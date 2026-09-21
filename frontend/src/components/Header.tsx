'use client';

import React from 'react';
import { regimeColor } from './ui';

interface HeaderProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
  isBackendConnected: boolean;
  currentRegimeLabel?: string;
}

export const TABS = [
  { id: 'new', label: 'Order Ticket' },
  { id: 'monitor', label: 'Monitor' },
  { id: 'analytics', label: 'Analytics' },
  { id: 'comparison', label: 'Comparison' },
  { id: 'research', label: 'Research' },
];

export function Header({
  activeTab,
  setActiveTab,
  isBackendConnected,
  currentRegimeLabel = 'Normal',
}: HeaderProps) {
  return (
    <header className="border-b border-line bg-bg sticky top-0 z-50">
      <div className="max-w-[1200px] mx-auto px-5 sm:px-8">
        <div className="flex items-center justify-between h-14 gap-6">
          <div className="flex items-baseline gap-3 shrink-0">
            <span className="font-mono font-medium tracking-[0.22em] text-ink text-[15px]">KAIRO</span>
            <span className="hidden lg:inline text-[12px] text-ink-3">Adaptive execution</span>
          </div>

          <nav aria-label="Primary" className="hidden md:flex items-stretch h-full gap-1 -mb-px">
            {TABS.map((tab) => {
              const isActive = activeTab === tab.id;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  aria-current={isActive ? 'page' : undefined}
                  className={`px-3.5 text-[13px] border-b-2 transition-colors ${
                    isActive
                      ? 'text-ink border-accent'
                      : 'text-ink-3 border-transparent hover:text-ink-2'
                  }`}
                >
                  {tab.label}
                </button>
              );
            })}
          </nav>

          <div className="flex items-center gap-5 text-[12px] shrink-0">
            <span className="hidden sm:flex items-center gap-2 text-ink-2">
              <span className="label">Regime</span>
              <span className="flex items-center gap-1.5 text-ink">
                <span className="h-1.5 w-1.5 rounded-full" style={{ background: regimeColor(currentRegimeLabel) }} />
                {currentRegimeLabel}
              </span>
            </span>
            <span className="flex items-center gap-1.5 text-ink-2">
              <span
                className={`h-1.5 w-1.5 rounded-full ${isBackendConnected ? 'bg-pos' : 'bg-neg'}`}
                aria-hidden
              />
              {isBackendConnected ? 'API connected' : 'API offline'}
            </span>
          </div>
        </div>

        {/* Mobile navigation */}
        <nav aria-label="Primary" className="md:hidden flex overflow-x-auto -mx-5 px-5 border-t border-line [scrollbar-width:none]">
          {TABS.map((tab) => {
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                aria-current={isActive ? 'page' : undefined}
                className={`px-3 py-2.5 text-[13px] whitespace-nowrap border-b-2 -mb-px ${
                  isActive ? 'text-ink border-accent' : 'text-ink-3 border-transparent'
                }`}
              >
                {tab.label}
              </button>
            );
          })}
        </nav>
      </div>
    </header>
  );
}
