import React from 'react'
import { InquirySummary } from '../../types/inquiry'

export type { InquirySummary } from '../../types/inquiry'

interface InquiryCardProps {
  inquiry: InquirySummary
  onClick: () => void
  variant?: 'default' | 'resolved' | 'bounty' | 'contested'
}

// Confidence ring SVG component
function ConfidenceRing({ probability, size = 56 }: { probability: number; size?: number }) {
  const radius = (size - 8) / 2
  const circumference = 2 * Math.PI * radius
  const offset = circumference - (circumference * probability)

  const color = probability > 0.8 ? '#22c55e' : probability > 0.5 ? '#f59e0b' : '#94a3b8'

  return (
    <div className="relative" style={{ width: size, height: size }}>
      <svg className="transform -rotate-90" style={{ width: size, height: size }}>
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          stroke="#e5e7eb"
          strokeWidth="4"
          fill="none"
        />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          stroke={color}
          strokeWidth="4"
          fill="none"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          className="transition-all duration-500"
        />
      </svg>
      <div className="absolute inset-0 flex items-center justify-center">
        <span className="text-xs font-bold text-slate-700">
          {Math.round(probability * 100)}%
        </span>
      </div>
    </div>
  )
}

function InquiryCard({ inquiry, onClick, variant = 'default' }: InquiryCardProps) {
  const isResolved = inquiry.status === 'resolved'
  const isContested = inquiry.entropy_bits > 3
  const isHighBounty = inquiry.stake > 100

  // Default cover image based on schema type
  const getCoverPlaceholder = () => {
    switch (inquiry.schema_type) {
      case 'monotone_count':
        return '🔢'
      case 'boolean':
        return '✓✗'
      case 'forecast':
        return '🔮'
      default:
        return '📋'
    }
  }

  // Card styling based on variant
  const getCardClass = () => {
    const base = "rounded-2xl p-5 cursor-pointer transition-all duration-200 group"
    switch (variant) {
      case 'resolved':
        return `${base} bg-gradient-to-br from-green-50 to-white border border-green-200 hover:shadow-lg`
      case 'bounty':
        return `${base} bg-white border border-slate-100 hover:shadow-md hover:border-amber-300`
      case 'contested':
        return `${base} bg-white border-2 border-red-100 hover:shadow-md hover:border-red-300`
      default:
        return `${base} bg-white border border-slate-100 shadow-sm hover:shadow-md hover:border-indigo-200`
    }
  }

  return (
    <div onClick={onClick} className={getCardClass()}>
      <div className="flex gap-4">
        {/* Cover Image / Placeholder */}
        <div className="flex-shrink-0">
          {inquiry.cover_image ? (
            <img
              src={inquiry.cover_image}
              alt=""
              className="w-28 h-28 rounded-xl object-cover"
              onError={(e) => {
                (e.target as HTMLImageElement).style.display = 'none'
              }}
            />
          ) : isResolved ? (
            <div className="w-14 h-14 rounded-full bg-green-500 flex items-center justify-center text-white font-bold text-lg">
              {typeof inquiry.posterior_map === 'number'
                ? inquiry.posterior_map
                : inquiry.posterior_map === 'true' ? '✓' : '✗'}
            </div>
          ) : (
            <ConfidenceRing probability={inquiry.posterior_probability} />
          )}
        </div>

        {/* Content */}
        <div className="flex-1 min-w-0">
          {/* Status badges */}
          <div className="flex items-center gap-2 mb-1.5">
            {isResolved && (
              <span className="text-xs bg-green-500 text-white px-2 py-0.5 rounded-full">
                RESOLVED
              </span>
            )}
            {inquiry.schema_type === 'forecast' && (
              <span className="text-xs bg-purple-100 text-purple-700 px-2 py-0.5 rounded-full">
                Forecast
              </span>
            )}
            <span className={`text-xs px-1.5 py-0.5 rounded border ${
              inquiry.rigor === 'A' ? 'bg-green-50 text-green-700 border-green-200' :
              inquiry.rigor === 'B' ? 'bg-indigo-50 text-indigo-700 border-indigo-200' :
              'bg-slate-50 text-slate-600 border-slate-200'
            }`}>
              Rigor {inquiry.rigor}
            </span>
            {inquiry.resolved_ago && (
              <span className="text-xs text-slate-400">{inquiry.resolved_ago}</span>
            )}
          </div>

          {/* Title */}
          <h3 className="font-semibold text-slate-800 mb-1.5 line-clamp-2 group-hover:text-indigo-700 transition-colors">
            {inquiry.title}
          </h3>

          {/* Scope entities */}
          {inquiry.scope_entities && inquiry.scope_entities.length > 0 && (
            <div className="flex flex-wrap gap-1 mb-2">
              {inquiry.scope_entities.slice(0, 3).map((entity, i) => (
                <span key={i} className="text-xs bg-purple-100 text-purple-700 px-2 py-0.5 rounded-full">
                  {entity}
                </span>
              ))}
            </div>
          )}

          {/* Stats row */}
          <div className="flex items-center gap-3 text-sm flex-wrap">
            {/* Best answer */}
            {inquiry.posterior_map !== null && !isResolved && (
              <span className="text-slate-500">
                Best: <strong className="text-slate-700">{inquiry.posterior_map}</strong>
              </span>
            )}

            {/* Contributions */}
            <span className="flex items-center gap-1 text-slate-500">
              <span className="text-xs">📊</span>
              {inquiry.contributions}
            </span>

            {/* Stake */}
            {inquiry.stake > 0 && (
              <span className="flex items-center gap-1 text-amber-600 font-medium">
                <span className="text-xs">💰</span>
                ${inquiry.stake.toFixed(2)}
              </span>
            )}

            {/* Open tasks */}
            {inquiry.open_tasks > 0 && (
              <span className="flex items-center gap-1 text-indigo-500">
                <span className="text-xs">⚡</span>
                {inquiry.open_tasks}
              </span>
            )}

            {/* Entropy (for contested) */}
            {isContested && (
              <span className="text-red-500 font-medium">
                🔥 {inquiry.entropy_bits.toFixed(1)} bits
              </span>
            )}

            {/* Confidence for resolved */}
            {isResolved && (
              <span className="text-green-600 font-medium ml-auto">
                {Math.round(inquiry.posterior_probability * 100)}% confidence
              </span>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default InquiryCard
