import React, { useState } from 'react'
import { formatTime } from '../../utils/timeFormat'
import URLPreview from '../comment/URLPreview'

export interface StoryMatch {
  story_id: string
  is_new: boolean
  match_score: number
  matched_story_title: string
}

export interface PreviewMeta {
  title?: string
  description?: string
  thumbnail_url?: string
  site_name?: string
}

export interface ExtractionResult {
  title?: string
  author?: string
  publish_date?: string
  meta_description?: string
  content_text?: string
  screenshot_url?: string
  word_count?: number
  reading_time_minutes?: number
  language?: string
}

export interface SemanticData {
  claims?: Array<{
    text?: string
    claim_text?: string
    claim_type?: string
  }>
  entities?: {
    people?: Array<{ name: string } | string>
    organizations?: Array<{ name: string } | string>
    locations?: Array<{ name: string } | string>
  }
}

export interface EventSubmission {
  id: string
  user_id: string
  user_name: string
  user_picture?: string
  content: string
  urls?: string
  status: 'pending' | 'extracting' | 'processing' | 'completed' | 'failed' | 'blocked'
  task_id?: string
  story_match?: StoryMatch
  preview_meta?: PreviewMeta
  created_at: string
  current_stage?: string
  error?: string
  block_reason?: string
  result?: ExtractionResult
  semantic_data?: SemanticData
}

interface PendingSubmissionProps {
  submission: EventSubmission
}

function PendingSubmission({ submission }: PendingSubmissionProps) {
  const [expanded, setExpanded] = useState(false)
  const timeInfo = formatTime(submission.created_at)
  const urlList = submission.urls ? submission.urls.split(',').filter(u => u.trim()) : []

  // Get status info
  const getStatusInfo = () => {
    switch (submission.status) {
      case 'pending':
        return { icon: '⏱️', text: 'queued', color: 'text-slate-600', borderColor: 'border-slate-300' }
      case 'extracting':
      case 'processing':
        return { icon: '🔍', text: 'processing', color: 'text-blue-600', borderColor: 'border-blue-300' }
      case 'failed':
        return { icon: '❌', text: 'failed', color: 'text-red-600', borderColor: 'border-red-300' }
      case 'blocked':
        return { icon: '⚠️', text: 'blocked', color: 'text-amber-600', borderColor: 'border-amber-300' }
      case 'completed':
        if (submission.story_match) {
          const isNew = submission.story_match.is_new
          const matchScore = Math.round(submission.story_match.match_score * 100)
          return {
            icon: '✓',
            text: isNew ? 'created new story' : `merged (${matchScore}% match)`,
            color: 'text-green-600',
            borderColor: 'border-green-300',
            storyLink: {
              id: submission.story_match.story_id,
              title: submission.story_match.matched_story_title
            }
          }
        }
        return { icon: '✓', text: 'completed', color: 'text-green-600', borderColor: 'border-green-300' }
      default:
        return { icon: '⏳', text: 'processing', color: 'text-slate-600', borderColor: 'border-slate-300' }
    }
  }

  const statusInfo = getStatusInfo()
  const urlDomain = urlList.length > 0 ? new URL(urlList[0]).hostname : null

  // Collapsed view (default)
  if (!expanded) {
    return (
      <div
        onClick={() => setExpanded(true)}
        className={`py-2 px-3 bg-slate-50 border-l-2 ${statusInfo.borderColor} text-xs text-slate-500 cursor-pointer hover:bg-slate-100 transition-colors flex items-center gap-2`}
      >
        {/* Small avatar */}
        {submission.user_picture ? (
          <img
            src={submission.user_picture}
            alt={submission.user_name}
            className="w-6 h-6 rounded-full flex-shrink-0"
          />
        ) : (
          <div className="w-6 h-6 rounded-full bg-gradient-to-br from-indigo-400 to-purple-400 flex items-center justify-center text-white text-xs font-semibold flex-shrink-0">
            {submission.user_name.charAt(0).toUpperCase()}
          </div>
        )}

        <div className="flex-1 min-w-0">
          <span className="font-medium text-slate-600">{submission.user_name}</span>
          {' '}<span className={statusInfo.color}>({statusInfo.icon} {statusInfo.text})</span>
          {' '}at {timeInfo.absolute}
          {urlDomain && (
            <>
              {' '}• <span className="text-slate-600">{urlDomain}</span>
            </>
          )}
          {statusInfo.storyLink && (
            <>
              {' '}→ <a
                href={`/story/${statusInfo.storyLink.id}`}
                className="text-indigo-600 hover:text-indigo-800 underline"
                onClick={(e) => e.stopPropagation()}
              >
                {statusInfo.storyLink.title}
              </a>
            </>
          )}
        </div>

        <span className="text-slate-400 flex-shrink-0">▼</span>
      </div>
    )
  }

  // Expanded view (like classic_app)
  return (
    <div className={`rounded-lg border-2 ${statusInfo.borderColor} p-4 bg-slate-50 transition-all`}>
      {/* Header with collapse button */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-3">
          {submission.user_picture ? (
            <img
              src={submission.user_picture}
              alt={submission.user_name}
              className="w-8 h-8 rounded-full flex-shrink-0"
            />
          ) : (
            <div className="w-8 h-8 rounded-full bg-gradient-to-br from-indigo-400 to-purple-400 flex items-center justify-center text-white text-sm font-semibold flex-shrink-0">
              {submission.user_name.charAt(0).toUpperCase()}
            </div>
          )}
          <div>
            <div className="font-semibold text-slate-900">{submission.user_name}</div>
            <div className="text-xs text-slate-500">{timeInfo.absolute}</div>
          </div>
        </div>
        <button
          onClick={() => setExpanded(false)}
          className="text-slate-400 hover:text-slate-600 text-lg font-bold"
        >
          ×
        </button>
      </div>

      {/* Status Badge */}
      <div className={`inline-flex items-center gap-2 px-3 py-1 rounded-full ${statusInfo.color} bg-white border ${statusInfo.borderColor} text-sm font-medium mb-3`}>
        <span>{statusInfo.icon}</span>
        <span>{statusInfo.text}</span>
      </div>

      {/* Extraction Pipeline Progress */}
      {(submission.status === 'extracting' || submission.status === 'processing' || submission.status === 'completed') && submission.current_stage && (
        <div className="mb-4 p-3 bg-white rounded-lg border border-slate-200">
          <div className="flex items-center justify-between gap-2">
            {[
              { id: 'extraction', label: 'Metadata', icon: '📋' },
              { id: 'cleaning', label: 'Metadata+', icon: '✨' },
              { id: 'content', label: 'Content', icon: '📄' },
              { id: 'semantization', label: 'Semantic', icon: '🧠' }
            ].map((stage, index, stages) => {
              const currentStageIndex = stages.findIndex(s => s.id === submission.current_stage)
              const isCompleted = index < currentStageIndex || submission.status === 'completed'
              const isActive = index === currentStageIndex && (submission.status === 'extracting' || submission.status === 'processing')

              return (
                <React.Fragment key={stage.id}>
                  <div className="flex flex-col items-center gap-1 flex-1">
                    <div
                      className={`w-10 h-10 rounded-full flex items-center justify-center text-lg font-semibold transition-all ${
                        isCompleted
                          ? 'bg-green-500 text-white'
                          : isActive
                          ? 'bg-blue-500 text-white animate-pulse'
                          : 'bg-slate-200 text-slate-400'
                      }`}
                    >
                      {stage.icon}
                    </div>
                    <div
                      className={`text-xs font-medium text-center ${
                        isCompleted
                          ? 'text-green-600'
                          : isActive
                          ? 'text-blue-600 font-semibold'
                          : 'text-slate-400'
                      }`}
                    >
                      {stage.label}
                    </div>
                  </div>
                  {index < stages.length - 1 && (
                    <div className="text-slate-300 text-lg mb-6">→</div>
                  )}
                </React.Fragment>
              )
            })}
          </div>
        </div>
      )}

      {/* Content */}
      {submission.content && (
        <p className="text-slate-700 text-sm leading-relaxed mb-3 whitespace-pre-wrap">
          {submission.content}
        </p>
      )}

      {/* URL Previews */}
      {urlList.length > 0 && (
        <div className="space-y-2 mt-3">
          {urlList.map((url, idx) => (
            <div key={idx}>
              {submission.preview_meta ? (
                <a
                  href={url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="mt-2 flex gap-3 border border-slate-200 rounded-lg overflow-hidden bg-white hover:border-indigo-300 hover:shadow-md transition-all p-3"
                >
                  {submission.preview_meta.thumbnail_url && (
                    <img
                      src={submission.preview_meta.thumbnail_url}
                      alt={submission.preview_meta.title || url}
                      className="w-20 h-20 rounded object-cover flex-shrink-0"
                      onError={(e) => {
                        e.currentTarget.style.display = 'none'
                      }}
                    />
                  )}
                  <div className="flex-1 min-w-0">
                    {submission.preview_meta.site_name && (
                      <div className="text-xs text-slate-500 mb-1">{submission.preview_meta.site_name}</div>
                    )}
                    <div className="font-semibold text-slate-900 text-sm line-clamp-2 mb-1">
                      {submission.preview_meta.title || url}
                    </div>
                    {submission.preview_meta.description && (
                      <div className="text-xs text-slate-600 line-clamp-2 mb-1">
                        {submission.preview_meta.description}
                      </div>
                    )}
                    <div className="text-xs text-indigo-600 flex items-center gap-1">
                      <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                      </svg>
                      {new URL(url).hostname}
                    </div>
                  </div>
                </a>
              ) : (
                <a
                  href={url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="mt-2 flex items-center gap-2 text-sm text-indigo-600 hover:text-indigo-800 underline"
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                  </svg>
                  {new URL(url).hostname}
                </a>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Story Link */}
      {statusInfo.storyLink && (
        <a
          href={`/story/${statusInfo.storyLink.id}`}
          className="mt-3 inline-flex items-center gap-2 px-4 py-2 bg-indigo-600 text-white font-semibold rounded-lg hover:bg-indigo-700 transition-colors text-sm"
        >
          <span>View Story: {statusInfo.storyLink.title}</span>
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
          </svg>
        </a>
      )}

      {/* Task Info Grid */}
      {(submission.task_id || submission.current_stage) && (
        <div className="mt-4 grid grid-cols-2 gap-3 text-xs">
          {submission.task_id && (
            <div className="bg-white p-2 rounded border border-slate-200">
              <div className="text-slate-500 uppercase font-medium mb-1">Task ID</div>
              <div className="text-slate-700 font-mono">{submission.task_id.substring(0, 8)}...</div>
            </div>
          )}
          {submission.current_stage && (
            <div className="bg-white p-2 rounded border border-slate-200">
              <div className="text-slate-500 uppercase font-medium mb-1">Current Stage</div>
              <div className="text-slate-700">{submission.current_stage}</div>
            </div>
          )}
          {submission.result?.word_count && (
            <div className="bg-white p-2 rounded border border-slate-200">
              <div className="text-slate-500 uppercase font-medium mb-1">Word Count</div>
              <div className="text-slate-700">{submission.result.word_count}</div>
            </div>
          )}
          {submission.result?.reading_time_minutes && (
            <div className="bg-white p-2 rounded border border-slate-200">
              <div className="text-slate-500 uppercase font-medium mb-1">Reading Time</div>
              <div className="text-slate-700">{submission.result.reading_time_minutes} min</div>
            </div>
          )}
        </div>
      )}

      {/* Screenshot */}
      {submission.result?.screenshot_url && (
        <div className="mt-4">
          <h4 className="font-semibold text-slate-900 mb-2 text-sm">Screenshot</h4>
          <img
            src={submission.result.screenshot_url}
            alt="Page screenshot"
            className="w-full max-w-2xl rounded-lg shadow-md border border-slate-200"
          />
        </div>
      )}

      {/* Extracted Content */}
      {submission.status === 'completed' && submission.result && (
        <div className="mt-4 bg-white p-4 rounded-lg border border-slate-200">
          <h3 className="text-lg font-bold text-slate-900 mb-2">
            {submission.result.title || 'Untitled'}
          </h3>

          {submission.result.author && (
            <p className="text-sm text-slate-600 mb-1">
              <strong>Author:</strong> {submission.result.author}
            </p>
          )}

          {submission.result.publish_date && (
            <p className="text-sm text-slate-600 mb-2">
              <strong>Published:</strong> {new Date(submission.result.publish_date).toLocaleDateString()}
            </p>
          )}

          {submission.result.meta_description && (
            <p className="text-sm text-slate-700 leading-relaxed mb-2">
              {submission.result.meta_description}
            </p>
          )}

          {/* Full Content (Collapsible) */}
          {submission.result.content_text && (
            <details className="mt-3">
              <summary className="cursor-pointer font-semibold text-indigo-600 hover:text-indigo-800 text-sm">
                Show Full Content ({submission.result.word_count || 0} words)
              </summary>
              <div className="mt-2 p-3 bg-slate-50 rounded text-sm text-slate-700 leading-relaxed max-h-96 overflow-y-auto whitespace-pre-wrap">
                {submission.result.content_text.substring(0, 5000)}
                {submission.result.content_text.length > 5000 && '...'}
              </div>
            </details>
          )}

          {/* Meta Tags */}
          {submission.result.language && (
            <div className="flex flex-wrap gap-2 mt-3">
              <span className="px-2 py-1 bg-slate-100 text-slate-700 rounded text-xs">
                Language: {submission.result.language}
              </span>
              {submission.result.word_count && (
                <span className="px-2 py-1 bg-slate-100 text-slate-700 rounded text-xs">
                  📝 {submission.result.word_count} words
                </span>
              )}
              {submission.result.reading_time_minutes && (
                <span className="px-2 py-1 bg-slate-100 text-slate-700 rounded text-xs">
                  ⏱️ {submission.result.reading_time_minutes} min read
                </span>
              )}
            </div>
          )}
        </div>
      )}

      {/* Semantic Analysis */}
      {submission.semantic_data && (submission.semantic_data.claims || submission.semantic_data.entities) && (
        <div className="mt-4 bg-white p-4 rounded-lg border border-slate-200">
          <h4 className="font-semibold text-slate-900 mb-3 flex items-center gap-2 text-sm">
            🧠 Semantic Analysis
          </h4>

          {/* Claims */}
          {submission.semantic_data.claims && submission.semantic_data.claims.length > 0 && (
            <details open className="mb-3">
              <summary className="cursor-pointer font-medium text-indigo-600 hover:text-indigo-800 text-sm mb-2">
                Claims ({submission.semantic_data.claims.length})
              </summary>
              <div className="space-y-2">
                {submission.semantic_data.claims.slice(0, 10).map((claim, idx) => (
                  <div key={idx} className="p-2 bg-slate-50 border-l-2 border-indigo-500 rounded text-sm">
                    <div className="font-medium text-slate-900">
                      {claim.text || claim.claim_text || ''}
                    </div>
                    {claim.claim_type && (
                      <div className="text-xs text-slate-500 mt-1">Type: {claim.claim_type}</div>
                    )}
                  </div>
                ))}
              </div>
            </details>
          )}

          {/* Entities */}
          {submission.semantic_data.entities && (
            <details className="mb-2">
              <summary className="cursor-pointer font-medium text-indigo-600 hover:text-indigo-800 text-sm mb-2">
                Entities
              </summary>
              <div className="space-y-2">
                {submission.semantic_data.entities.people && submission.semantic_data.entities.people.length > 0 && (
                  <div>
                    <strong className="text-xs text-slate-600">People:</strong>
                    <div className="flex flex-wrap gap-1 mt-1">
                      {submission.semantic_data.entities.people.slice(0, 10).map((p, idx) => (
                        <span key={idx} className="px-2 py-1 bg-blue-50 text-blue-700 rounded text-xs">
                          {typeof p === 'string' ? p : p.name}
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                {submission.semantic_data.entities.organizations && submission.semantic_data.entities.organizations.length > 0 && (
                  <div>
                    <strong className="text-xs text-slate-600">Organizations:</strong>
                    <div className="flex flex-wrap gap-1 mt-1">
                      {submission.semantic_data.entities.organizations.slice(0, 10).map((o, idx) => (
                        <span key={idx} className="px-2 py-1 bg-purple-50 text-purple-700 rounded text-xs">
                          {typeof o === 'string' ? o : o.name}
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                {submission.semantic_data.entities.locations && submission.semantic_data.entities.locations.length > 0 && (
                  <div>
                    <strong className="text-xs text-slate-600">Locations:</strong>
                    <div className="flex flex-wrap gap-1 mt-1">
                      {submission.semantic_data.entities.locations.slice(0, 10).map((l, idx) => (
                        <span key={idx} className="px-2 py-1 bg-green-50 text-green-700 rounded text-xs">
                          {typeof l === 'string' ? l : l.name}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </details>
          )}
        </div>
      )}

      {/* Error Display */}
      {submission.status === 'failed' && submission.error && (
        <div className="mt-4 p-3 bg-red-50 border-l-4 border-red-500 rounded text-sm">
          <strong className="text-red-900">Error:</strong>
          <p className="text-red-700 mt-1">{submission.error}</p>
        </div>
      )}

      {/* Block Reason */}
      {submission.status === 'blocked' && submission.block_reason && (
        <div className="mt-4 p-3 bg-amber-50 border-l-4 border-amber-500 rounded text-sm">
          <strong className="text-amber-900">Blocked:</strong>
          <p className="text-amber-700 mt-1">{submission.block_reason}</p>
        </div>
      )}
    </div>
  )
}

export default PendingSubmission
