import React, { useState, useEffect, useRef } from 'react'
import { Contribution, BeliefSnapshot } from '../../types/inquiry'

interface ReplayTimelineProps {
  contributions: Contribution[]
  snapshots: BeliefSnapshot[]
  currentIndex: number
  onIndexChange: (index: number) => void
  isPlaying: boolean
  onPlayPause: () => void
}

function getObservationIcon(kind?: string): string {
  switch (kind) {
    case 'point': return '='
    case 'lower_bound': return '≥'
    case 'upper_bound': return '≤'
    case 'interval': return '↔'
    case 'approximate': return '~'
    default: return '?'
  }
}

function getContributionIcon(type: string): string {
  switch (type) {
    case 'evidence': return '📊'
    case 'refutation': return '❌'
    case 'attribution': return '🗣️'
    case 'scope_correction': return '🚫'
    case 'disambiguation': return '🔍'
    default: return '📝'
  }
}

function formatDelta(delta: number, prefix: string = ''): string {
  if (Math.abs(delta) < 0.001) return ''
  const sign = delta > 0 ? '+' : ''
  return `${prefix}${sign}${delta.toFixed(2)}`
}

function ReplayTimeline({
  contributions,
  snapshots,
  currentIndex,
  onIndexChange,
  isPlaying,
  onPlayPause
}: ReplayTimelineProps) {
  const timelineRef = useRef<HTMLDivElement>(null)

  // Auto-scroll to current item
  useEffect(() => {
    if (timelineRef.current && currentIndex >= 0) {
      const item = timelineRef.current.children[currentIndex] as HTMLElement
      if (item) {
        item.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
      }
    }
  }, [currentIndex])

  if (contributions.length === 0) {
    return (
      <div className="bg-white rounded-2xl p-5 shadow-sm border border-slate-100">
        <h3 className="font-semibold text-slate-800 mb-4 flex items-center gap-2">
          <span>🎬</span> Replay Timeline
        </h3>
        <div className="text-center py-8 text-slate-400">
          <div className="text-3xl mb-2">📭</div>
          <p>No contributions yet</p>
          <p className="text-xs mt-1">Be the first to add evidence!</p>
        </div>
      </div>
    )
  }

  return (
    <div className="bg-white rounded-2xl p-5 shadow-sm border border-slate-100">
      <div className="flex items-center justify-between mb-4">
        <h3 className="font-semibold text-slate-800 flex items-center gap-2">
          <span>🎬</span> Replay Timeline
        </h3>
        <div className="flex items-center gap-2">
          <button
            onClick={() => onIndexChange(0)}
            disabled={currentIndex === 0}
            className="p-1.5 rounded hover:bg-slate-100 disabled:opacity-30 disabled:cursor-not-allowed"
            title="Reset to start"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 19l-7-7 7-7m8 14l-7-7 7-7" />
            </svg>
          </button>
          <button
            onClick={onPlayPause}
            className={`p-2 rounded-lg font-medium text-sm ${
              isPlaying
                ? 'bg-red-100 text-red-600 hover:bg-red-200'
                : 'bg-indigo-100 text-indigo-600 hover:bg-indigo-200'
            }`}
          >
            {isPlaying ? (
              <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                <path d="M6 4h4v16H6V4zm8 0h4v16h-4V4z" />
              </svg>
            ) : (
              <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                <path d="M8 5v14l11-7z" />
              </svg>
            )}
          </button>
          <button
            onClick={() => onIndexChange(contributions.length - 1)}
            disabled={currentIndex === contributions.length - 1}
            className="p-1.5 rounded hover:bg-slate-100 disabled:opacity-30 disabled:cursor-not-allowed"
            title="Jump to end"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 5l7 7-7 7M5 5l7 7-7 7" />
            </svg>
          </button>
        </div>
      </div>

      {/* Progress bar */}
      <div className="mb-4">
        <div className="flex justify-between text-xs text-slate-400 mb-1">
          <span>Start</span>
          <span>{currentIndex + 1} / {contributions.length}</span>
          <span>Now</span>
        </div>
        <div className="h-1.5 bg-slate-100 rounded-full overflow-hidden">
          <div
            className="h-full bg-indigo-500 rounded-full transition-all duration-300"
            style={{ width: `${((currentIndex + 1) / contributions.length) * 100}%` }}
          />
        </div>
      </div>

      {/* Timeline items */}
      <div ref={timelineRef} className="space-y-2 max-h-80 overflow-y-auto pr-1">
        {contributions.map((contrib, i) => {
          const snapshot = snapshots[i]
          const isActive = i === currentIndex
          const isPast = i < currentIndex
          const isFuture = i > currentIndex

          return (
            <div
              key={contrib.id}
              onClick={() => onIndexChange(i)}
              className={`
                relative flex gap-3 p-2 rounded-lg cursor-pointer transition-all
                ${isActive ? 'bg-indigo-50 border border-indigo-200 shadow-sm' : ''}
                ${isPast ? 'opacity-60' : ''}
                ${isFuture ? 'opacity-40' : ''}
                hover:opacity-100 hover:bg-slate-50
              `}
            >
              {/* Timeline connector */}
              <div className="flex flex-col items-center">
                <div className={`
                  w-8 h-8 rounded-full flex items-center justify-center text-sm
                  ${isActive ? 'bg-indigo-500 text-white' : isPast ? 'bg-green-100 text-green-600' : 'bg-slate-100 text-slate-400'}
                `}>
                  {isPast ? '✓' : getContributionIcon(contrib.type)}
                </div>
                {i < contributions.length - 1 && (
                  <div className={`w-0.5 flex-1 mt-1 ${isPast ? 'bg-green-200' : 'bg-slate-200'}`} />
                )}
              </div>

              {/* Content */}
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-1">
                  <span className="text-xs font-medium text-slate-600">{contrib.source_name || 'Unknown'}</span>
                  <span className="text-xs text-slate-400">
                    {contrib.observation_kind && (
                      <span className="font-mono bg-slate-100 px-1 rounded">
                        {getObservationIcon(contrib.observation_kind)}{contrib.extracted_value}
                      </span>
                    )}
                  </span>
                </div>
                <p className="text-xs text-slate-600 line-clamp-2">{contrib.text}</p>

                {/* Impact metrics */}
                {snapshot && (
                  <div className="flex items-center gap-3 mt-1">
                    {snapshot.probability_delta !== 0 && (
                      <span className={`text-xs font-medium ${snapshot.probability_delta > 0 ? 'text-green-600' : 'text-red-500'}`}>
                        {formatDelta(snapshot.probability_delta * 100, '')}% conf
                      </span>
                    )}
                    {snapshot.entropy_delta !== 0 && (
                      <span className={`text-xs ${snapshot.entropy_delta < 0 ? 'text-green-600' : 'text-amber-500'}`}>
                        {formatDelta(snapshot.entropy_delta, '')} bits
                      </span>
                    )}
                    {(contrib.posterior_impact || contrib.impact || 0) > 0.01 && (
                      <span className="text-xs text-indigo-500">
                        +{Math.round((contrib.posterior_impact || contrib.impact || 0) * 100)}% impact
                      </span>
                    )}
                  </div>
                )}
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

export default ReplayTimeline
