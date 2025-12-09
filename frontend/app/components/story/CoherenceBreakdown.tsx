import React from 'react'
import { CoherenceBreakdown as CoherenceBreakdownType } from '../../types/story'

interface CoherenceBreakdownProps {
  breakdown: CoherenceBreakdownType
}

function CoherenceBreakdown({ breakdown }: CoherenceBreakdownProps) {
  const metrics = [
    { label: 'Entity Overlap', value: breakdown.entity_overlap, color: 'indigo' },
    { label: 'Centrality', value: breakdown.centrality, color: 'purple' },
    { label: 'Claim Density', value: breakdown.claim_density, color: 'pink' }
  ]

  const getColorClasses = (color: string) => {
    const colors = {
      indigo: 'bg-indigo-500',
      purple: 'bg-purple-500',
      pink: 'bg-pink-500'
    }
    return colors[color as keyof typeof colors] || 'bg-slate-500'
  }

  return (
    <div className="bg-white rounded-xl border border-slate-200 p-6">
      <h3 className="text-lg font-semibold text-slate-900 mb-4">Coherence Breakdown</h3>

      {/* Overall Score */}
      <div className="mb-6 pb-6 border-b border-slate-200">
        <div className="flex justify-between items-center mb-2">
          <span className="text-sm font-medium text-slate-600">Overall Coherence</span>
          <span className="text-2xl font-bold text-indigo-600">
            {breakdown.score.toFixed(1)}/100
          </span>
        </div>
        <div className="w-full bg-slate-200 rounded-full h-3">
          <div
            className="bg-gradient-to-r from-indigo-500 to-purple-500 h-3 rounded-full transition-all"
            style={{ width: `${breakdown.score}%` }}
          />
        </div>
      </div>

      {/* Component Metrics */}
      <div className="space-y-4 mb-6">
        {metrics.map((metric) => (
          <div key={metric.label}>
            <div className="flex justify-between items-center mb-2">
              <span className="text-sm text-slate-600">{metric.label}</span>
              <span className="text-sm font-semibold text-slate-900">
                {metric.value.toFixed(1)}%
              </span>
            </div>
            <div className="w-full bg-slate-200 rounded-full h-2">
              <div
                className={`${getColorClasses(metric.color)} h-2 rounded-full transition-all`}
                style={{ width: `${metric.value}%` }}
              />
            </div>
          </div>
        ))}
      </div>

      {/* Detailed Stats */}
      <div className="grid grid-cols-2 gap-4 pt-4 border-t border-slate-200">
        <div className="text-center">
          <div className="text-2xl font-bold text-slate-900">
            {breakdown.breakdown.shared_entities}
          </div>
          <div className="text-xs text-slate-500">Shared Entities</div>
        </div>
        <div className="text-center">
          <div className="text-2xl font-bold text-slate-900">
            {breakdown.breakdown.total_entities}
          </div>
          <div className="text-xs text-slate-500">Total Entities</div>
        </div>
        <div className="text-center">
          <div className="text-2xl font-bold text-slate-900">
            {breakdown.breakdown.avg_entity_stories.toFixed(1)}
          </div>
          <div className="text-xs text-slate-500">Avg Stories/Entity</div>
        </div>
        <div className="text-center">
          <div className="text-2xl font-bold text-slate-900">
            {breakdown.breakdown.claim_count}
          </div>
          <div className="text-xs text-slate-500">Claims</div>
        </div>
      </div>
    </div>
  )
}

export default CoherenceBreakdown
