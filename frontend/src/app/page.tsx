'use client';

import React, { useState, useEffect } from 'react';
import { Header } from '../components/Header';
import { NewExecutionTab } from '../components/NewExecutionTab';
import { ExecutionMonitorTab } from '../components/ExecutionMonitorTab';
import { ExecutionAnalyticsTab } from '../components/ExecutionAnalyticsTab';
import { StrategyComparisonTab } from '../components/StrategyComparisonTab';
import { ResearchTab } from '../components/ResearchTab';
import { checkBackendHealth, getCurrentRegime, ExecutionRecord } from '../lib/api';

export default function Home() {
  const [activeTab, setActiveTab] = useState('new');
  const [isBackendConnected, setIsBackendConnected] = useState(false);
  const [currentRegimeLabel, setCurrentRegimeLabel] = useState('Normal');
  const [executionRecord, setExecutionRecord] = useState<ExecutionRecord | null>(null);

  useEffect(() => {
    // Check backend health & regime on load and poll every 10s
    const checkStatus = async () => {
      try {
        await checkBackendHealth();
        setIsBackendConnected(true);
        const regimeData = await getCurrentRegime().catch(() => null);
        if (regimeData) {
          setCurrentRegimeLabel(regimeData.regime_label);
        }
      } catch {
        setIsBackendConnected(false);
      }
    };

    checkStatus();
    const interval = setInterval(checkStatus, 10000);
    return () => clearInterval(interval);
  }, []);

  const handleExecutionCreated = (record: ExecutionRecord) => {
    setExecutionRecord(record);
    setActiveTab('monitor');
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 font-sans selection:bg-indigo-500 selection:text-white">
      {/* Top Header & Navigation */}
      <Header
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        isBackendConnected={isBackendConnected}
        currentRegimeLabel={currentRegimeLabel}
      />

      {/* Main Tab Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {activeTab === 'new' && (
          <NewExecutionTab onExecutionCreated={handleExecutionCreated} />
        )}

        {activeTab === 'monitor' && (
          <ExecutionMonitorTab
            executionRecord={executionRecord}
            onNavigateToAnalytics={() => setActiveTab('analytics')}
          />
        )}

        {activeTab === 'analytics' && (
          <ExecutionAnalyticsTab executionRecord={executionRecord} />
        )}

        {activeTab === 'comparison' && (
          <StrategyComparisonTab />
        )}

        {activeTab === 'research' && (
          <ResearchTab />
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-900 bg-slate-950 py-6 mt-12 text-center text-xs text-slate-500">
        <p>KAIRO — Regime-Aware Deep Reinforcement Learning for Optimal Trade Execution Platform</p>
      </footer>
    </div>
  );
}
