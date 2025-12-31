import React, { useEffect, useState } from 'react'
import { BeliefState } from '../../types/inquiry'

interface BeliefStatePanelProps {
  beliefState: BeliefState
  posteriorTop10: Array<{ value: any; probability: number }>
  credibleInterval?: [number, number]
  schemaType: string
  animated?: boolean
  prevBeliefState?: BeliefState
}

function BeliefStatePanel({
  beliefState,
  posteriorTop10,
  credibleInterval,
  schemaType,
  animated = true,
  prevBeliefState
}: BeliefStatePanelProps) {
  const [displayProb, setDisplayProb] = useState(beliefState.map_probability)
  const [displayEntropy, setDisplayEntropy] = useState(beliefState.entropy_bits)

  // Animate changes
  useEffect(() => {
    if (!animated) {
      setDisplayProb(beliefState.map_probability)
      setDisplayEntropy(beliefState.entropy_bits)
      return
    }

    const duration = 500
    const steps = 20
    const interval = duration / steps

    const probDelta = (beliefState.map_probability - displayProb) / steps
    const entropyDelta = (beliefState.entropy_bits - displayEntropy) / steps

    let step = 0
    const timer = setInterval(() => {
      step++
      setDisplayProb(prev => prev + probDelta)
      setDisplayEntropy(prev => prev + entropyDelta)
      if (step >= steps) {
        clearInterval(timer)
        setDisplayProb(beliefState.map_probability)
        setDisplayEntropy(beliefState.entropy_bits)
      }
    }, interval)

    return () => clearInterval(timer)
  }, [beliefState.map_probability, beliefState.entropy_bits])

  // Determine confidence color
  const getConfidenceColor = (prob: number) => {
    if (prob >= 0.95) return 'text-green-600'
    if (prob >= 0.8) return 'text-green-500'
    if (prob >= 0.5) return 'text-amber-500'
    return 'text-slate-400'
  }

  const getBarColor = (prob: number) => {
    if (prob >= 0.95) return 'bg-green-500'
    if (prob >= 0.8) return 'bg-green-400'
    if (prob >= 0.5) return 'bg-amber-500'
    return 'bg-slate-400'
  }

  // Format MAP value for display
  const formatMapValue = (value: any) => {
    if (typeof value === 'number') {
      return value.toLocaleString()
    }
    if (typeof value === 'boolean') {
      return value ? 'Yes' : 'No'
    }
    if (value === 'true') return 'Yes'
    if (value === 'false') return 'No'
    return String(value)
  }

  // Calculate delta indicators
  const probDelta = prevBeliefState
    ? beliefState.map_probability - prevBeliefState.map_probability
    : 0
  const entropyDelta = prevBeliefState
    ? beliefState.entropy_bits - prevBeliefState.entropy_bits
    : 0

  return (
    <div className="bg-white rounded-2xl p-5 shadow-sm border border-slate-100">
      <h3 className="font-semibold text-slate-800 mb-4 flex items-center gap-2">
        <span>🎯</span> Belief State
      </h3>

      {/* Big MAP Display */}
      <div className="bg-gradient-to-br from-slate-50 to-slate-100 rounded-xl p-6 mb-4">
        <div className="text-center">
          <div className="text-xs text-slate-500 uppercase tracking-wide mb-1">
            Most Likely Answer
          </div>
          <div className="text-4xl font-bold text-slate-800 mb-2">
            {formatMapValue(beliefState.map)}
          </div>
          <div className="flex items-center justify-center gap-2">
            <span className={`text-lg font-semibold ${getConfidenceColor(displayProb)}`}>
              {Math.round(displayProb * 100)}%
            </span>
            <span className="text-slate-400">confident</span>
            {probDelta !== 0 && (
              <span className={`text-xs ${probDelta > 0 ? 'text-green-500' : 'text-red-500'}`}>
                {probDelta > 0 ? '+' : ''}{Math.round(probDelta * 100)}%
              </span>
            )}
          </div>
        </div>

        {/* Confidence bar */}
        <div className="mt-4">
          <div className="h-3 bg-slate-200 rounded-full overflow-hidden">
            <div
              className={`h-full rounded-full transition-all duration-500 ${getBarColor(displayProb)}`}
              style={{ width: `${displayProb * 100}%` }}
            />
          </div>
          <div className="flex justify-between text-xs text-slate-400 mt-1">
            <span>0%</span>
            <span className="text-slate-500">95% = resolvable</span>
            <span>100%</span>
          </div>
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-3 gap-3 mb-4">
        <div className="bg-slate-50 rounded-lg p-3 text-center">
          <div className="text-xs text-slate-500">Entropy</div>
          <div className="font-semibold text-slate-700">
            {displayEntropy.toFixed(2)}
            <span className="text-xs text-slate-400 ml-1">bits</span>
          </div>
          {entropyDelta !== 0 && (
            <div className={`text-xs ${entropyDelta < 0 ? 'text-green-500' : 'text-amber-500'}`}>
              {entropyDelta > 0 ? '+' : ''}{entropyDelta.toFixed(2)}
            </div>
          )}
        </div>
        <div className="bg-slate-50 rounded-lg p-3 text-center">
          <div className="text-xs text-slate-500">Observations</div>
          <div className="font-semibold text-slate-700">{beliefState.observation_count}</div>
        </div>
        <div className="bg-slate-50 rounded-lg p-3 text-center">
          <div className="text-xs text-slate-500">Uncertainty</div>
          <div className="font-semibold text-slate-700">
            {Math.round(beliefState.normalized_entropy * 100)}%
          </div>
        </div>
      </div>

      {/* Credible Interval */}
      {credibleInterval && schemaType === 'monotone_count' && (
        <div className="bg-indigo-50 rounded-lg p-3 mb-4">
          <div className="flex items-center justify-between">
            <span className="text-xs text-indigo-600">95% Credible Interval</span>
            <span className="font-mono text-sm text-indigo-700">
              [{credibleInterval[0].toLocaleString()}, {credibleInterval[1].toLocaleString()}]
            </span>
          </div>
        </div>
      )}

      {/* Posterior Distribution */}
      {posteriorTop10 && posteriorTop10.length > 0 && (
        <div>
          <div className="text-xs text-slate-500 mb-2">Probability Distribution</div>
          <div className="flex items-end gap-1 h-24 bg-slate-50 rounded-lg p-3">
            {posteriorTop10.slice(0, 8).map((item, i) => {
              const maxProb = posteriorTop10[0].probability
              const height = (item.probability / maxProb) * 100
              const isMap = item.value === beliefState.map

              return (
                <div key={i} className="flex-1 flex flex-col items-center">
                  <div className="text-xs text-slate-500 mb-1">
                    {Math.round(item.probability * 100)}%
                  </div>
                  <div
                    className={`w-full rounded-t transition-all duration-500 ${
                      isMap ? 'bg-indigo-500' : 'bg-indigo-200'
                    }`}
                    style={{ height: `${height * 0.6}px`, minHeight: '4px' }}
                  />
                  <div className={`text-xs font-mono mt-1 ${isMap ? 'text-indigo-600 font-bold' : 'text-slate-500'}`}>
                    {typeof item.value === 'number' ? item.value.toLocaleString() : String(item.value)}
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* Log Score (for nerds) */}
      <div className="mt-3 pt-3 border-t border-slate-100">
        <div className="flex justify-between text-xs text-slate-400">
          <span>Total Log Score</span>
          <span className="font-mono">{beliefState.total_log_score.toFixed(2)}</span>
        </div>
      </div>
    </div>
  )
}

export default BeliefStatePanel
