import React from 'react';

export interface SourceDiversity {
  wire: number;      // Reuters, AP, AFP
  international: number;  // BBC, Guardian, DW
  local: number;     // SCMP, local papers
  official: number;  // Government sources
  ngo: number;       // Amnesty, RSF, etc.
  other: number;
}

export interface EpistemicGap {
  type: 'missing_source' | 'perspective_gap' | 'unverified' | 'stale';
  description: string;
  priority: 'high' | 'medium' | 'low';
  bounty?: number;
}

export interface EpistemicState {
  source_count: number;
  source_diversity: SourceDiversity;
  claim_count: number;
  coverage: number;  // 0-1, percentage of expected aspects covered
  heat: number;      // 0-1, recency factor
  has_contradiction: boolean;
  perspective_balance?: {
    [key: string]: number;  // e.g., "prosecution": 0.6, "defense": 0.2
  };
  gaps: EpistemicGap[];
  last_updated: string;
}

interface EpistemicStateCardProps {
  state: EpistemicState | null;
  loading?: boolean;
  compact?: boolean;
}

const EpistemicStateCard: React.FC<EpistemicStateCardProps> = ({
  state,
  loading = false,
  compact = false
}) => {
  if (loading) {
    return (
      <div className="bg-white rounded-lg border border-slate-200 shadow-sm p-4 animate-pulse">
        <div className="h-4 bg-slate-200 rounded w-1/3 mb-3"></div>
        <div className="h-8 bg-slate-200 rounded mb-3"></div>
        <div className="h-4 bg-slate-200 rounded w-2/3"></div>
      </div>
    );
  }

  if (!state) {
    return null;
  }

  const coveragePercent = Math.round(state.coverage * 100);
  const heatPercent = Math.round(state.heat * 100);
  const openQuestions = state.gaps.length;
  const totalSources = Object.values(state.source_diversity).reduce((a, b) => a + b, 0);

  // Heat indicator color
  const heatColor = heatPercent > 70 ? 'text-red-500' :
                    heatPercent > 40 ? 'text-amber-500' : 'text-slate-400';

  const heatLabel = heatPercent > 70 ? 'Hot' :
                    heatPercent > 40 ? 'Warm' : 'Cooling';

  // Source diversity breakdown
  const sourceTypes = [
    { key: 'wire', label: 'Wire', icon: '📡', color: 'bg-blue-100 text-blue-700' },
    { key: 'international', label: 'Intl', icon: '🌍', color: 'bg-green-100 text-green-700' },
    { key: 'local', label: 'Local', icon: '📰', color: 'bg-purple-100 text-purple-700' },
    { key: 'official', label: 'Official', icon: '🏛️', color: 'bg-amber-100 text-amber-700' },
    { key: 'ngo', label: 'NGO', icon: '⚖️', color: 'bg-rose-100 text-rose-700' },
  ];

  if (compact) {
    return (
      <div className="bg-gradient-to-r from-indigo-50 to-purple-50 rounded-lg border border-indigo-100 p-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-1.5">
              <span className="text-sm font-medium text-slate-600">Coverage:</span>
              <span className="text-sm font-bold text-indigo-600">{coveragePercent}%</span>
            </div>
            <div className="w-px h-4 bg-slate-200"></div>
            <div className="flex items-center gap-1.5">
              <span className="text-sm font-medium text-slate-600">Sources:</span>
              <span className="text-sm font-bold text-purple-600">{totalSources}</span>
            </div>
            <div className="w-px h-4 bg-slate-200"></div>
            <div className={`flex items-center gap-1 ${heatColor}`}>
              <span className="text-sm">🔥</span>
              <span className="text-sm font-medium">{heatLabel}</span>
            </div>
          </div>
          {openQuestions > 0 && (
            <div className="flex items-center gap-1 text-amber-600">
              <span className="text-sm">❓</span>
              <span className="text-sm font-medium">{openQuestions} open</span>
            </div>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg border border-slate-200 shadow-sm overflow-hidden">
      {/* Header */}
      <div className="px-4 py-3 bg-gradient-to-r from-indigo-50 to-purple-50 border-b border-indigo-100">
        <div className="flex items-center justify-between">
          <h3 className="font-semibold text-slate-800 flex items-center gap-2">
            <span>📊</span>
            Epistemic State
          </h3>
          <div className={`flex items-center gap-1 ${heatColor}`}>
            <span>🔥</span>
            <span className="text-sm font-medium">{heatLabel}</span>
          </div>
        </div>
      </div>

      <div className="p-4 space-y-4">
        {/* Coverage bar */}
        <div>
          <div className="flex items-center justify-between text-sm mb-1.5">
            <span className="text-slate-600">Story Coverage</span>
            <span className="font-bold text-indigo-600">{coveragePercent}%</span>
          </div>
          <div className="h-2.5 bg-slate-100 rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-indigo-500 to-purple-500 rounded-full transition-all duration-500"
              style={{ width: `${coveragePercent}%` }}
            />
          </div>
          {openQuestions > 0 && (
            <div className="mt-1.5 text-xs text-amber-600 flex items-center gap-1">
              <span>❓</span>
              <span>{openQuestions} open question{openQuestions > 1 ? 's' : ''}</span>
            </div>
          )}
        </div>

        {/* Source diversity */}
        <div>
          <div className="flex items-center justify-between text-sm mb-2">
            <span className="text-slate-600">Source Diversity</span>
            <span className="text-slate-500">{totalSources} sources</span>
          </div>
          <div className="flex flex-wrap gap-1.5">
            {sourceTypes.map(st => {
              const count = state.source_diversity[st.key as keyof SourceDiversity] || 0;
              if (count === 0) return null;
              return (
                <span
                  key={st.key}
                  className={`inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium ${st.color}`}
                >
                  <span>{st.icon}</span>
                  <span>{st.label}</span>
                  <span className="font-bold">{count}</span>
                </span>
              );
            })}
          </div>

          {/* Missing source types */}
          {sourceTypes.some(st => (state.source_diversity[st.key as keyof SourceDiversity] || 0) === 0) && (
            <div className="mt-2 text-xs text-slate-400">
              Missing: {sourceTypes
                .filter(st => (state.source_diversity[st.key as keyof SourceDiversity] || 0) === 0)
                .map(st => st.label)
                .join(', ')}
            </div>
          )}
        </div>

        {/* Contradiction warning */}
        {state.has_contradiction && (
          <div className="flex items-start gap-2 p-2.5 bg-amber-50 border border-amber-200 rounded-lg">
            <span className="text-amber-500 flex-shrink-0">⚡</span>
            <div className="text-sm text-amber-800">
              <span className="font-medium">Contradictions detected</span>
              <span className="text-amber-600"> — sources disagree on some facts</span>
            </div>
          </div>
        )}

        {/* Perspective balance (if available) */}
        {state.perspective_balance && Object.keys(state.perspective_balance).length > 0 && (
          <div>
            <div className="text-sm text-slate-600 mb-2">Perspective Balance</div>
            <div className="space-y-1.5">
              {Object.entries(state.perspective_balance)
                .sort((a, b) => b[1] - a[1])
                .map(([perspective, ratio]) => (
                  <div key={perspective} className="flex items-center gap-2">
                    <span className="text-xs text-slate-500 w-20 truncate capitalize">
                      {perspective.replace(/_/g, ' ')}
                    </span>
                    <div className="flex-1 h-1.5 bg-slate-100 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-indigo-400 rounded-full"
                        style={{ width: `${ratio * 100}%` }}
                      />
                    </div>
                    <span className="text-xs text-slate-400 w-10 text-right">
                      {Math.round(ratio * 100)}%
                    </span>
                  </div>
                ))}
            </div>
          </div>
        )}

        {/* Last updated */}
        <div className="pt-2 border-t border-slate-100 text-xs text-slate-400 text-center">
          Updated {new Date(state.last_updated).toLocaleString()}
        </div>
      </div>
    </div>
  );
};

export default EpistemicStateCard;
