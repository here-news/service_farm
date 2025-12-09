import React from 'react'
import { Artifact } from '../../types/story'
import { formatTime } from '../../utils/timeFormat'

interface ArtifactsListProps {
  artifacts: Artifact[]
}

function ArtifactsList({ artifacts }: ArtifactsListProps) {
  if (artifacts.length === 0) {
    return (
      <div className="text-center py-8 text-slate-400">
        No source articles found
      </div>
    )
  }

  return (
    <div className="space-y-3">
      {artifacts.map((artifact, index) => {
        const timeInfo = formatTime(artifact.published_at)

        return (
          <a
            key={index}
            href={artifact.url}
            target="_blank"
            rel="noopener noreferrer"
            className="block border border-slate-200 rounded-lg p-4 hover:shadow-md hover:border-indigo-300 transition"
          >
            <div className="flex items-start gap-3">
              <div className="w-10 h-10 rounded bg-slate-100 flex items-center justify-center text-slate-400 flex-shrink-0">
                📄
              </div>
              <div className="flex-1 min-w-0">
                <h4 className="font-medium text-slate-900 mb-1 line-clamp-2">
                  {artifact.title || 'Untitled Article'}
                </h4>
                <div className="flex items-center gap-3 text-xs text-slate-500">
                  <span className="font-medium text-indigo-600">{artifact.domain}</span>
                  {artifact.published_at && timeInfo.absolute !== 'Date unknown' && (
                    <span
                      className={`cursor-help ${!timeInfo.reliable ? 'opacity-50' : ''}`}
                      title={timeInfo.fullDateTime}
                    >
                      {timeInfo.absolute}
                      {!timeInfo.reliable && <span className="ml-1">⚠️</span>}
                    </span>
                  )}
                </div>
              </div>
              <svg className="w-5 h-5 text-slate-400 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
              </svg>
            </div>
          </a>
        )
      })}
    </div>
  )
}

export default ArtifactsList
