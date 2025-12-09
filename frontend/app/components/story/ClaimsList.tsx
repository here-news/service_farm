import React from 'react'
import { Claim } from '../../types/story'

interface ClaimsListProps {
  claims: Claim[]
}

function ClaimsList({ claims }: ClaimsListProps) {
  if (claims.length === 0) {
    return (
      <div className="text-center py-8 text-slate-400">
        No claims extracted yet
      </div>
    )
  }

  return (
    <div className="space-y-3">
      {claims.map((claim, index) => (
        <div
          key={claim.id || index}
          className="border border-slate-200 rounded-lg p-4 hover:bg-slate-50 transition"
        >
          <div className="flex gap-3">
            <div className="w-8 h-8 rounded-full bg-indigo-100 text-indigo-600 flex items-center justify-center font-semibold text-sm flex-shrink-0">
              {index + 1}
            </div>
            <div className="flex-1">
              <p className="text-slate-800 leading-relaxed">{claim.text}</p>
              {claim.confidence !== undefined && (
                <div className="mt-2 flex items-center gap-2">
                  <div className="flex-1 bg-slate-200 rounded-full h-2">
                    <div
                      className="bg-gradient-to-r from-indigo-500 to-purple-500 h-2 rounded-full transition-all"
                      style={{ width: `${claim.confidence * 100}%` }}
                    />
                  </div>
                  <span className="text-xs text-slate-500 font-medium">
                    {(claim.confidence * 100).toFixed(0)}%
                  </span>
                </div>
              )}
            </div>
          </div>
        </div>
      ))}
    </div>
  )
}

export default ClaimsList
