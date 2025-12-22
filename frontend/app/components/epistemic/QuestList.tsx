import React, { useState } from 'react';
import { EpistemicGap } from './EpistemicStateCard';

interface QuestListProps {
  gaps: EpistemicGap[];
  eventId: string;
  onContribute?: (gap: EpistemicGap) => void;
  compact?: boolean;
}

const QuestList: React.FC<QuestListProps> = ({
  gaps,
  eventId: _eventId,
  onContribute,
  compact = false
}) => {
  const [expandedQuest, setExpandedQuest] = useState<string | null>(null);

  if (!gaps || gaps.length === 0) {
    return null;
  }

  // Sort by priority and bounty
  const sortedGaps = [...gaps].sort((a, b) => {
    const priorityOrder = { high: 0, medium: 1, low: 2 };
    const priorityDiff = priorityOrder[a.priority] - priorityOrder[b.priority];
    if (priorityDiff !== 0) return priorityDiff;
    return (b.bounty || 0) - (a.bounty || 0);
  });

  const getGapIcon = (type: EpistemicGap['type']) => {
    switch (type) {
      case 'missing_source': return '📎';
      case 'perspective_gap': return '⚖️';
      case 'unverified': return '❓';
      case 'stale': return '🕐';
      default: return '🎯';
    }
  };

  const getPriorityColor = (priority: EpistemicGap['priority']) => {
    switch (priority) {
      case 'high': return 'text-red-600 bg-red-50 border-red-200';
      case 'medium': return 'text-amber-600 bg-amber-50 border-amber-200';
      case 'low': return 'text-slate-600 bg-slate-50 border-slate-200';
      default: return 'text-slate-600 bg-slate-50 border-slate-200';
    }
  };

  const getTypeLabel = (type: EpistemicGap['type']) => {
    switch (type) {
      case 'missing_source': return 'Source needed';
      case 'perspective_gap': return 'Perspective needed';
      case 'unverified': return 'Verification needed';
      case 'stale': return 'Update needed';
      default: return 'Help needed';
    }
  };

  if (compact) {
    return (
      <div className="bg-amber-50 border border-amber-200 rounded-lg p-3">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm font-medium text-amber-800 flex items-center gap-1.5">
            <span>❓</span>
            Help improve this story
          </span>
          <span className="text-xs text-amber-600">{gaps.length} open</span>
        </div>
        <div className="flex flex-wrap gap-1.5">
          {sortedGaps.slice(0, 3).map((gap, idx) => (
            <button
              key={idx}
              onClick={() => onContribute?.(gap)}
              className="inline-flex items-center gap-1 px-2 py-1 bg-white border border-amber-200 rounded-full text-xs text-amber-700 hover:bg-amber-100 transition-colors"
            >
              <span>{getGapIcon(gap.type)}</span>
              <span className="max-w-[100px] truncate">{gap.description}</span>
              {gap.bounty && (
                <span className="font-bold text-amber-600">+{gap.bounty}</span>
              )}
            </button>
          ))}
          {gaps.length > 3 && (
            <span className="text-xs text-amber-500 self-center">
              +{gaps.length - 3} more
            </span>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg border border-slate-200 shadow-sm overflow-hidden">
      {/* Header */}
      <div className="px-4 py-3 bg-gradient-to-r from-amber-50 to-yellow-50 border-b border-amber-100">
        <div className="flex items-center justify-between">
          <h3 className="font-semibold text-slate-800 flex items-center gap-2">
            <span>❓</span>
            Help Improve This Story
          </h3>
          <span className="text-xs text-amber-600 font-medium">
            {gaps.length} open quest{gaps.length > 1 ? 's' : ''}
          </span>
        </div>
        <p className="text-xs text-slate-500 mt-1">
          Contribute sources to earn credits and improve coverage
        </p>
      </div>

      {/* Quest list */}
      <div className="divide-y divide-slate-100">
        {sortedGaps.map((gap, idx) => {
          const questKey = `${gap.type}-${idx}`;
          const isExpanded = expandedQuest === questKey;

          return (
            <div
              key={questKey}
              className={`p-4 transition-colors ${isExpanded ? 'bg-amber-50/50' : 'hover:bg-slate-50'}`}
            >
              <div
                className="flex items-start gap-3 cursor-pointer"
                onClick={() => setExpandedQuest(isExpanded ? null : questKey)}
              >
                {/* Icon */}
                <div className="text-xl flex-shrink-0 mt-0.5">
                  {getGapIcon(gap.type)}
                </div>

                {/* Content */}
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1">
                    <span className={`px-2 py-0.5 rounded-full text-xs font-medium border ${getPriorityColor(gap.priority)}`}>
                      {gap.priority}
                    </span>
                    <span className="text-xs text-slate-400">
                      {getTypeLabel(gap.type)}
                    </span>
                  </div>
                  <p className="text-sm text-slate-700 leading-snug">
                    {gap.description}
                  </p>
                </div>

                {/* Bounty */}
                {gap.bounty && (
                  <div className="flex-shrink-0 text-right">
                    <div className="text-lg font-bold text-amber-600">
                      +{gap.bounty}
                    </div>
                    <div className="text-xs text-slate-400">credits</div>
                  </div>
                )}
              </div>

              {/* Expanded content */}
              {isExpanded && (
                <div className="mt-4 pl-9">
                  <div className="bg-white border border-amber-200 rounded-lg p-3">
                    <p className="text-sm text-slate-600 mb-3">
                      {gap.type === 'missing_source' && (
                        <>Find and submit a relevant source from this category to earn credits.</>
                      )}
                      {gap.type === 'perspective_gap' && (
                        <>This story needs more balanced perspectives. Submit sources that provide alternative viewpoints.</>
                      )}
                      {gap.type === 'unverified' && (
                        <>Help verify this claim by finding supporting or contradicting sources.</>
                      )}
                      {gap.type === 'stale' && (
                        <>This story may have new developments. Submit recent sources to keep it current.</>
                      )}
                    </p>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        onContribute?.(gap);
                      }}
                      className="w-full py-2.5 bg-gradient-to-r from-amber-500 to-yellow-500 hover:from-amber-600 hover:to-yellow-600 text-white rounded-lg font-medium transition-all shadow-sm hover:shadow flex items-center justify-center gap-2"
                    >
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                      </svg>
                      Submit a Source
                    </button>
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Footer */}
      <div className="px-4 py-3 bg-slate-50 border-t border-slate-100">
        <div className="flex items-center justify-between">
          <span className="text-xs text-slate-500">
            Total available: <span className="font-bold text-amber-600">
              {gaps.reduce((sum, g) => sum + (g.bounty || 0), 0)} credits
            </span>
          </span>
          <button className="text-xs text-indigo-600 hover:text-indigo-800 font-medium">
            Learn how credits work →
          </button>
        </div>
      </div>
    </div>
  );
};

export default QuestList;
