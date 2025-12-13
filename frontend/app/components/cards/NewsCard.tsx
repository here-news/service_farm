import React, { useState } from 'react'
import { Story, PageThumbnail } from '../../types/story'
import { formatTime } from '../../utils/timeFormat'

interface NewsCardProps {
  story: Story
  onClick: () => void
}

// Fanning thumbnails component - displays multiple source thumbnails with slide-on-hover
function FanningThumbnails({ thumbnails }: { thumbnails: PageThumbnail[] }) {
  const [isHovered, setIsHovered] = useState(false)

  if (!thumbnails || thumbnails.length === 0) return null

  // Show up to 4 thumbnails in a fanning stack
  const displayThumbnails = thumbnails.slice(0, 4)
  const extraCount = thumbnails.length - 4

  return (
    <div
      className="relative w-32 h-28 flex-shrink-0"
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      {displayThumbnails.map((thumb, index) => {
        // Default stacked position
        const stackedRotation = (index - 1.5) * 5
        const stackedX = index * 4
        const stackedY = index * 2

        // Spread out position on hover - fan out horizontally
        const spreadX = index * 28
        const spreadRotation = 0

        const zIndex = isHovered ? index + 1 : displayThumbnails.length - index

        return (
          <div
            key={thumb.page_id}
            className="absolute top-0 left-0 w-24 h-24 rounded-lg overflow-hidden shadow-md border border-slate-200 bg-white transition-all duration-300 ease-out hover:shadow-lg"
            style={{
              transform: isHovered
                ? `translateX(${spreadX}px) translateY(0px) rotate(${spreadRotation}deg) scale(1.02)`
                : `translateX(${stackedX}px) translateY(${stackedY}px) rotate(${stackedRotation}deg)`,
              zIndex,
              transitionDelay: isHovered ? `${index * 50}ms` : `${(displayThumbnails.length - index) * 30}ms`,
            }}
            title={`${thumb.title} (${thumb.domain})`}
          >
            <img
              src={thumb.thumbnail_url}
              alt={thumb.title}
              className="w-full h-full object-cover"
              onError={(e) => {
                // Hide broken images
                (e.target as HTMLImageElement).style.display = 'none'
              }}
            />
            {/* Show domain label on hover */}
            {isHovered && thumb.domain && (
              <div className="absolute bottom-0 left-0 right-0 bg-black/70 text-white text-xs px-1 py-0.5 truncate">
                {thumb.domain}
              </div>
            )}
          </div>
        )
      })}
      {extraCount > 0 && (
        <div
          className="absolute bottom-0 right-0 bg-indigo-600 text-white text-xs font-semibold px-1.5 py-0.5 rounded-full shadow transition-opacity duration-200"
          style={{
            zIndex: 10,
            opacity: isHovered ? 0 : 1
          }}
        >
          +{extraCount}
        </div>
      )}
    </div>
  )
}

// Get thought styling based on type
function getThoughtStyle(type: string): { icon: string; bgClass: string; textClass: string; borderClass: string } {
  switch (type) {
    case 'contradiction':
    case 'contradiction_detected':
      return {
        icon: '⚡',
        bgClass: 'bg-gradient-to-r from-red-50 to-orange-50',
        textClass: 'text-red-700',
        borderClass: 'border-l-red-400'
      }
    case 'coherence_drop':
      return {
        icon: '📉',
        bgClass: 'bg-gradient-to-r from-amber-50 to-yellow-50',
        textClass: 'text-amber-700',
        borderClass: 'border-l-amber-400'
      }
    case 'anomaly':
      return {
        icon: '🔍',
        bgClass: 'bg-gradient-to-r from-purple-50 to-indigo-50',
        textClass: 'text-purple-700',
        borderClass: 'border-l-purple-400'
      }
    case 'progress':
      return {
        icon: '📈',
        bgClass: 'bg-gradient-to-r from-green-50 to-emerald-50',
        textClass: 'text-green-700',
        borderClass: 'border-l-green-400'
      }
    case 'emergence':
      return {
        icon: '🌱',
        bgClass: 'bg-gradient-to-r from-teal-50 to-cyan-50',
        textClass: 'text-teal-700',
        borderClass: 'border-l-teal-400'
      }
    case 'state_change':
      return {
        icon: '🔄',
        bgClass: 'bg-gradient-to-r from-blue-50 to-sky-50',
        textClass: 'text-blue-700',
        borderClass: 'border-l-blue-400'
      }
    case 'question':
      return {
        icon: '❓',
        bgClass: 'bg-gradient-to-r from-indigo-50 to-violet-50',
        textClass: 'text-indigo-700',
        borderClass: 'border-l-indigo-400'
      }
    default:
      return {
        icon: '💭',
        bgClass: 'bg-gradient-to-r from-slate-50 to-gray-50',
        textClass: 'text-slate-700',
        borderClass: 'border-l-slate-400'
      }
  }
}

// Extract first sentence or meaningful snippet from summary
function getSummarySnippet(summary: string | undefined, maxLength: number = 150): string | null {
  if (!summary) return null

  // Remove claim/entity references like [cl_xxx] and [en_xxx]
  const cleaned = summary.replace(/\[cl_[a-z0-9]+\]/g, '').replace(/\[en_[a-z0-9]+\]/g, '').trim()

  // Get first sentence
  const firstSentence = cleaned.split(/[.!?]/)[0]
  if (firstSentence && firstSentence.length > 20) {
    return firstSentence.length > maxLength
      ? firstSentence.substring(0, maxLength) + '...'
      : firstSentence + '.'
  }

  // Fallback to truncated text
  return cleaned.length > maxLength
    ? cleaned.substring(0, maxLength) + '...'
    : cleaned
}

function NewsCard({ story, onClick }: NewsCardProps) {
  const timeInfo = formatTime(story.last_updated || story.created_at)
  const hasThumbnails = story.page_thumbnails && story.page_thumbnails.length > 0
  const summarySnippet = getSummarySnippet(story.summary || story.description)
  const thoughtStyle = story.thought ? getThoughtStyle(story.thought.type) : null

  return (
    <div
      onClick={onClick}
      className="border border-slate-200 rounded-xl p-5 hover:shadow-lg hover:border-indigo-200 transition-all cursor-pointer bg-white group"
    >
      <div className="flex gap-4">
        {/* Fanning thumbnails from source pages */}
        {hasThumbnails ? (
          <FanningThumbnails thumbnails={story.page_thumbnails!} />
        ) : story.cover_image ? (
          <img
            src={story.cover_image}
            alt={story.title}
            className="w-28 h-28 rounded-lg object-cover flex-shrink-0"
          />
        ) : (
          // Placeholder when no images
          <div className="w-28 h-28 rounded-lg bg-gradient-to-br from-slate-100 to-slate-200 flex-shrink-0 flex items-center justify-center">
            <span className="text-4xl text-slate-300">📰</span>
          </div>
        )}

        <div className="flex-1 min-w-0">
          {/* Title */}
          <h3 className="text-lg font-semibold text-slate-900 mb-1.5 line-clamp-2 group-hover:text-indigo-700 transition-colors">
            {story.title}
          </h3>

          {/* Summary snippet - always show if available */}
          {summarySnippet && (
            <p className="text-sm text-slate-600 mb-2 line-clamp-2 leading-relaxed">
              {summarySnippet}
            </p>
          )}

          {/* Thought byline - stimulating observation from the event organism */}
          {story.thought && thoughtStyle && (
            <div className={`${thoughtStyle.bgClass} ${thoughtStyle.textClass} border-l-4 ${thoughtStyle.borderClass} px-3 py-2 rounded-r-lg mb-3 flex items-start gap-2 animate-pulse-subtle`}>
              <span className="text-base flex-shrink-0">{thoughtStyle.icon}</span>
              <p className="text-sm italic line-clamp-2 leading-snug">
                {story.thought.content}
              </p>
            </div>
          )}

          {/* Metadata row */}
          <div className="flex items-center gap-3 text-sm flex-wrap">
            {/* Coherence indicator */}
            {story.coherence !== undefined && story.coherence !== null && (
              <div className="flex items-center gap-1.5" title="Event coherence score">
                <div className={`w-2.5 h-2.5 rounded-full ${
                  story.coherence >= 0.7 ? 'bg-green-500' :
                  story.coherence >= 0.4 ? 'bg-yellow-500' : 'bg-red-500'
                }`} />
                <span className="text-slate-600 font-medium">
                  {(story.coherence * 100).toFixed(0)}%
                </span>
              </div>
            )}

            {/* Claims count */}
            {story.claim_count !== undefined && story.claim_count > 0 && (
              <div className="flex items-center gap-1 text-slate-500" title="Number of verified claims">
                <span className="text-xs">📋</span>
                <span>{story.claim_count} claims</span>
              </div>
            )}

            {/* Source count */}
            {story.page_count !== undefined && story.page_count > 0 && (
              <div className="flex items-center gap-1 text-slate-500" title="Number of source articles">
                <span className="text-xs">📰</span>
                <span>{story.page_count} sources</span>
              </div>
            )}

            {/* Time Information */}
            {timeInfo.absolute !== 'Date unknown' && (
              <div
                className="flex items-center gap-1 cursor-help text-slate-400 ml-auto"
                title={timeInfo.fullDateTime}
              >
                <span className="text-xs">🕐</span>
                <span>{timeInfo.relative}</span>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default NewsCard
