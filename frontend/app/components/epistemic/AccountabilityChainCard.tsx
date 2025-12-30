import React, { useState } from 'react';

export interface ChainStep {
  id: string;
  label: string;  // 'PROMISE', 'ACTION', 'OUTCOME', 'VERIFICATION'
  text: string;
  status: 'completed' | 'pending' | 'unknown';
  source?: string;
  date?: string;
}

export interface AccountabilityChain {
  id: string;
  title: string;
  type: 'relief_fund' | 'investigation' | 'prosecution' | 'policy' | 'other';
  steps: ChainStep[];
  lastUpdated: string;
}

interface AccountabilityChainCardProps {
  chains: AccountabilityChain[];
  onAddEvidence?: (chainId: string, stepId: string) => void;
  onSetReminder?: (chainId: string, stepId: string) => void;
  compact?: boolean;
}

const AccountabilityChainCard: React.FC<AccountabilityChainCardProps> = ({
  chains,
  onAddEvidence,
  onSetReminder,
  compact = false
}) => {
  const [expandedChain, setExpandedChain] = useState<string | null>(null);

  const getTypeIcon = (type: AccountabilityChain['type']) => {
    switch (type) {
      case 'relief_fund': return '💰';
      case 'investigation': return '🔍';
      case 'prosecution': return '⚖️';
      case 'policy': return '📋';
      default: return '📌';
    }
  };

  const getStatusColor = (status: ChainStep['status']) => {
    switch (status) {
      case 'completed': return 'bg-green-500';
      case 'pending': return 'bg-amber-400';
      case 'unknown': return 'bg-slate-300';
    }
  };

  const pendingCount = chains.reduce(
    (acc, chain) => acc + chain.steps.filter(s => s.status === 'pending' || s.status === 'unknown').length,
    0
  );

  if (compact) {
    return (
      <div className="bg-gradient-to-r from-orange-50 to-amber-50 rounded-lg border border-orange-200 p-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="text-orange-500 text-lg">📋</span>
            <span className="text-sm font-medium text-slate-700">
              {chains.length} Accountability Chain{chains.length !== 1 ? 's' : ''}
            </span>
          </div>
          <span className="text-xs text-orange-600 font-medium">
            {pendingCount} pending
          </span>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg border border-slate-200 shadow-sm overflow-hidden">
      {/* Header */}
      <div className="px-4 py-3 bg-gradient-to-r from-orange-50 to-amber-50 border-b border-orange-100">
        <div className="flex items-center justify-between">
          <h3 className="font-semibold text-slate-800 flex items-center gap-2">
            <span>📋</span>
            Accountability Tracking
          </h3>
          <span className="px-2 py-0.5 bg-orange-100 text-orange-700 text-xs font-semibold rounded-full">
            {pendingCount} pending
          </span>
        </div>
        <p className="text-xs text-slate-500 mt-1">
          Track promises from announcement to fulfillment
        </p>
      </div>

      <div className="divide-y divide-slate-100">
        {chains.map(chain => {
          const isExpanded = expandedChain === chain.id;
          const completedSteps = chain.steps.filter(s => s.status === 'completed').length;
          const progress = (completedSteps / chain.steps.length) * 100;

          return (
            <div key={chain.id} className="p-4">
              {/* Chain header */}
              <button
                onClick={() => setExpandedChain(isExpanded ? null : chain.id)}
                className="w-full flex items-center gap-3 text-left"
              >
                <span className="text-xl">{getTypeIcon(chain.type)}</span>
                <div className="flex-1">
                  <div className="font-medium text-slate-800">{chain.title}</div>
                  <div className="flex items-center gap-2 mt-1">
                    {/* Progress bar */}
                    <div className="flex-1 h-1.5 bg-slate-100 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-gradient-to-r from-green-400 to-green-500 rounded-full transition-all duration-300"
                        style={{ width: `${progress}%` }}
                      />
                    </div>
                    <span className="text-xs text-slate-500">
                      {completedSteps}/{chain.steps.length}
                    </span>
                  </div>
                </div>
                <svg
                  className={`w-5 h-5 text-slate-400 transition-transform ${isExpanded ? 'rotate-180' : ''}`}
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                </svg>
              </button>

              {/* Expanded chain steps */}
              {isExpanded && (
                <div className="mt-4 pl-3">
                  {chain.steps.map((step, idx) => (
                    <div key={step.id} className="flex gap-3">
                      {/* Connector */}
                      <div className="flex flex-col items-center">
                        <div className={`w-3 h-3 rounded-full ${getStatusColor(step.status)} flex-shrink-0`} />
                        {idx < chain.steps.length - 1 && (
                          <div className="w-0.5 flex-1 bg-slate-200 min-h-[40px]" />
                        )}
                      </div>

                      {/* Step content */}
                      <div className="flex-1 pb-4">
                        <div className="text-[10px] uppercase tracking-wider text-slate-400 font-semibold">
                          {step.label}
                        </div>
                        <div className={`text-sm mt-0.5 ${
                          step.status === 'completed' ? 'text-slate-700' : 'text-slate-400'
                        }`}>
                          {step.text}
                        </div>

                        {/* Source/date info for completed steps */}
                        {step.status === 'completed' && step.source && (
                          <div className="text-xs text-slate-400 mt-1">
                            {step.source}
                            {step.date && <span className="ml-2">{step.date}</span>}
                          </div>
                        )}

                        {/* Action buttons for pending/unknown steps */}
                        {(step.status === 'pending' || step.status === 'unknown') && (
                          <div className="flex gap-2 mt-2">
                            <button
                              onClick={() => onAddEvidence?.(chain.id, step.id)}
                              className="px-2.5 py-1 bg-indigo-50 text-indigo-600 text-xs font-medium rounded-lg hover:bg-indigo-100 transition-colors"
                            >
                              + Add Evidence
                            </button>
                            {step.date && (
                              <button
                                onClick={() => onSetReminder?.(chain.id, step.id)}
                                className="px-2.5 py-1 bg-amber-50 text-amber-600 text-xs font-medium rounded-lg hover:bg-amber-100 transition-colors"
                              >
                                Set Reminder
                              </button>
                            )}
                          </div>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Footer */}
      <div className="px-4 py-3 bg-slate-50 border-t border-slate-100">
        <div className="text-xs text-slate-500 text-center">
          Help track these promises to completion
        </div>
      </div>
    </div>
  );
};

export default AccountabilityChainCard;
