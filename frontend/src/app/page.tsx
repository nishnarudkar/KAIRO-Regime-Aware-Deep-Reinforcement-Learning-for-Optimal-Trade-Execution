'use client';

import React, { useState, useEffect } from 'react';
import { Header } from '../components/Header';
import { NewExecutionTab } from '../components/NewExecutionTab';
import { ExecutionMonitorTab } from '../components/ExecutionMonitorTab';
import { ExecutionAnalyticsTab } from '../components/ExecutionAnalyticsTab';
import { StrategyComparisonTab } from '../components/StrategyComparisonTab';
import { ResearchTab } from '../components/ResearchTab';
import { checkBackendHealth, getCurrentRegime, ExecutionRecord, API_BASE_URL } from '../lib/api';

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
    <div className="min-h-screen flex flex-col">
      <Header
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        isBackendConnected={isBackendConnected}
        currentRegimeLabel={currentRegimeLabel}
      />

      <main className="flex-1 w-full max-w-[1200px] mx-auto px-5 sm:px-8 py-8">
        {activeTab === 'new' && <NewExecutionTab onExecutionCreated={handleExecutionCreated} />}

        {activeTab === 'monitor' && (
          <ExecutionMonitorTab
            executionRecord={executionRecord}
            onNavigateToAnalytics={() => setActiveTab('analytics')}
            onNavigateToNew={() => setActiveTab('new')}
          />
        )}

        {activeTab === 'analytics' && (
          <ExecutionAnalyticsTab executionRecord={executionRecord} onNavigateToNew={() => setActiveTab('new')} />
        )}

        {activeTab === 'comparison' && <StrategyComparisonTab />}

        {activeTab === 'research' && <ResearchTab />}
      </main>

      <footer className="border-t border-line">
        <div className="max-w-[1200px] mx-auto px-5 sm:px-8 py-4 flex flex-wrap justify-between gap-2 text-[12px] text-ink-3">
          <span>Synthetic market data · simulated fills · regime-aware deep RL for trade execution</span>
          <span className="num">{API_BASE_URL}</span>
        </div>
      </footer>
    </div>
  );
}
