import React, { useState } from 'react';

export interface DivergentValue {
  value: string;
  source: string;
  date?: string;
  votes: number;
  isLatest?: boolean;
}

export interface DivergentTopic {
  id: string;
  topic: string;
  question: string; // "How many people died?"
  values: DivergentValue[];
  status: 'needs_resolution' | 'temporal_update' | 'resolved';
  resolution?: {
    value: string;
    explanation: string;
    resolvedBy: string;
    resolvedAt: string;
  };
}

interface DivergentValuesCardProps {
  topics: DivergentTopic[];
  onVote?: (topicId: string, value: string) => Promise<void>;
  onResolve?: (topicId: string, value: string, explanation: string, method: string) => Promise<void>;
  compact?: boolean;
}

const DivergentValuesCard: React.FC<DivergentValuesCardProps> = ({
  topics,
  onVote,
  onResolve,
  compact = false
}) => {
  const [expandedTopic, setExpandedTopic] = useState<string | null>(null);
  const [selectedValues, setSelectedValues] = useState<Record<string, string>>({});
  const [resolutionText, setResolutionText] = useState<Record<string, string>>({});
  const [votedValues, setVotedValues] = useState<Set<string>>(new Set());

  const handleVote = async (topicId: string, value: string) => {
    const key = `${topicId}:${value}`;
    if (votedValues.has(key)) return;

    setVotedValues(new Set([...votedValues, key]));
    if (onVote) {
      await onVote(topicId, value);
    }
  };

  const handleResolve = async (topicId: string) => {
    const value = selectedValues[topicId];
    const explanation = resolutionText[topicId] || '';
    if (!value) return;

    if (onResolve) {
      await onResolve(topicId, value, explanation, 'temporal_update');
    }
  };

  const unresolvedTopics = topics.filter(t => t.status === 'needs_resolution');

  if (compact) {
    return (
      <div className="bg-gradient-to-r from-red-50 to-orange-50 rounded-lg border border-red-200 p-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="text-red-500 text-lg">⚡</span>
            <span className="text-sm font-medium text-slate-700">
              {unresolvedTopics.length} Divergent Value{unresolvedTopics.length !== 1 ? 's' : ''}
            </span>
          </div>
          <button className="text-xs text-red-600 hover:text-red-700 font-medium">
            Resolve
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg border border-slate-200 shadow-sm overflow-hidden">
      {/* Header */}
      <div className="px-4 py-3 bg-gradient-to-r from-red-50 to-orange-50 border-b border-red-100">
        <div className="flex items-center justify-between">
          <h3 className="font-semibold text-slate-800 flex items-center gap-2">
            <span>⚡</span>
            Divergent Values
          </h3>
          <span className="px-2 py-0.5 bg-red-100 text-red-700 text-xs font-semibold rounded-full">
            {unresolvedTopics.length} need resolution
          </span>
        </div>
      </div>

      <div className="divide-y divide-slate-100">
        {topics.map(topic => {
          const isExpanded = expandedTopic === topic.id;
          const sortedValues = [...topic.values].sort((a, b) => b.votes - a.votes);
          const topValue = sortedValues[0];

          return (
            <div key={topic.id} className="p-4">
              {/* Topic header */}
              <button
                onClick={() => setExpandedTopic(isExpanded ? null : topic.id)}
                className="w-full flex items-center justify-between text-left"
              >
                <div>
                  <div className="font-medium text-slate-800">{topic.topic}</div>
                  <div className="text-sm text-slate-500 mt-0.5">{topic.question}</div>
                </div>
                <div className="flex items-center gap-2">
                  {topic.status === 'resolved' ? (
                    <span className="px-2 py-0.5 bg-green-100 text-green-700 text-xs font-medium rounded-full">
                      Resolved
                    </span>
                  ) : topic.status === 'temporal_update' ? (
                    <span className="px-2 py-0.5 bg-blue-100 text-blue-700 text-xs font-medium rounded-full">
                      Temporal
                    </span>
                  ) : (
                    <span className="px-2 py-0.5 bg-amber-100 text-amber-700 text-xs font-medium rounded-full">
                      Open
                    </span>
                  )}
                  <svg
                    className={`w-5 h-5 text-slate-400 transition-transform ${isExpanded ? 'rotate-180' : ''}`}
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                  </svg>
                </div>
              </button>

              {/* Quick preview when collapsed */}
              {!isExpanded && (
                <div className="mt-2 flex items-center gap-3">
                  <span className="text-2xl font-bold text-slate-800">{topValue?.value}</span>
                  <span className="text-slate-400">vs</span>
                  {sortedValues.slice(1, 3).map((v, idx) => (
                    <span key={idx} className="text-lg text-slate-500">{v.value}</span>
                  ))}
                  {sortedValues.length > 3 && (
                    <span className="text-sm text-slate-400">+{sortedValues.length - 3} more</span>
                  )}
                </div>
              )}

              {/* Expanded view */}
              {isExpanded && (
                <div className="mt-4 space-y-3">
                  {sortedValues.map((val, idx) => {
                    const isVoted = votedValues.has(`${topic.id}:${val.value}`);
                    const isSelected = selectedValues[topic.id] === val.value;

                    return (
                      <div
                        key={idx}
                        onClick={() => setSelectedValues({ ...selectedValues, [topic.id]: val.value })}
                        className={`flex items-center gap-4 p-3 rounded-lg border cursor-pointer transition-all ${
                          isSelected
                            ? 'bg-green-50 border-green-300'
                            : 'bg-slate-50 border-slate-200 hover:border-slate-300'
                        }`}
                      >
                        {/* Value */}
                        <div className={`text-2xl font-bold min-w-[60px] ${
                          val.isLatest ? 'text-green-600' : 'text-slate-600'
                        }`}>
                          {val.value}
                        </div>

                        {/* Source info */}
                        <div className="flex-1">
                          <div className="text-sm text-slate-600">{val.source}</div>
                          {val.date && (
                            <div className="text-xs text-slate-400">{val.date}</div>
                          )}
                          {val.isLatest && (
                            <div className="text-xs text-green-600 font-medium mt-0.5">
                              Most recent
                            </div>
                          )}
                        </div>

                        {/* Vote button */}
                        <div className="flex flex-col items-center gap-1">
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              handleVote(topic.id, val.value);
                            }}
                            disabled={isVoted}
                            className={`w-8 h-8 rounded-full flex items-center justify-center transition-all ${
                              isVoted
                                ? 'bg-green-500 text-white'
                                : 'bg-slate-200 hover:bg-green-100 text-slate-500'
                            }`}
                          >
                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 15l7-7 7 7" />
                            </svg>
                          </button>
                          <span className="text-xs text-slate-500">{val.votes + (isVoted ? 1 : 0)}</span>
                        </div>
                      </div>
                    );
                  })}

                  {/* Resolution input */}
                  {topic.status === 'needs_resolution' && (
                    <div className="mt-4 p-3 bg-slate-50 rounded-lg">
                      <div className="text-sm text-slate-600 mb-2">
                        Provide temporal context to resolve this:
                      </div>
                      <textarea
                        value={resolutionText[topic.id] || ''}
                        onChange={(e) => setResolutionText({ ...resolutionText, [topic.id]: e.target.value })}
                        placeholder="e.g., 'The 160 figure was confirmed after DNA testing on Dec 2. Earlier figures were preliminary.'"
                        className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm resize-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
                        rows={2}
                      />
                      <div className="flex gap-2 mt-2">
                        <button
                          onClick={() => handleResolve(topic.id)}
                          disabled={!selectedValues[topic.id]}
                          className="px-3 py-1.5 bg-indigo-500 text-white text-sm font-medium rounded-lg hover:bg-indigo-600 disabled:bg-slate-300 disabled:cursor-not-allowed"
                        >
                          Submit Resolution
                        </button>
                        <button className="px-3 py-1.5 border border-slate-300 text-slate-600 text-sm font-medium rounded-lg hover:bg-slate-50">
                          Mark as Temporal Update
                        </button>
                      </div>
                    </div>
                  )}

                  {/* Resolved state */}
                  {topic.status === 'resolved' && topic.resolution && (
                    <div className="mt-3 p-3 bg-green-50 border border-green-200 rounded-lg">
                      <div className="flex items-center gap-2 text-green-700 text-sm">
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                        </svg>
                        <span className="font-medium">Resolved: {topic.resolution.value}</span>
                      </div>
                      <div className="text-sm text-green-600 mt-1">
                        {topic.resolution.explanation}
                      </div>
                      <div className="text-xs text-green-500 mt-1">
                        by {topic.resolution.resolvedBy}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default DivergentValuesCard;
