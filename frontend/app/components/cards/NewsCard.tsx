import React from 'react'
import { Story } from '../../types/story'
import { formatTime } from '../../utils/timeFormat'

interface NewsCardProps {
  story: Story
  onClick: () => void
}

function NewsCard({ story, onClick }: NewsCardProps) {
  const timeInfo = formatTime(story.last_updated || story.created_at)

  return (
    <div
      onClick={onClick}
      className="border border-slate-200 rounded-xl p-6 hover:shadow-lg transition cursor-pointer bg-white"
    >
      <div className="flex gap-4">
        {story.cover_image && (
          <img
            src={story.cover_image}
            alt={story.title}
            className="w-24 h-24 rounded-lg object-cover flex-shrink-0"
          />
        )}

        <div className="flex-1 min-w-0">
          <h3 className="text-xl font-semibold text-slate-900 mb-2 line-clamp-2">
            {story.title}
          </h3>

          {story.description && (
            <p className="text-sm text-slate-600 mb-3 line-clamp-2">
              {story.description}
            </p>
          )}

          <div className="flex items-center gap-4 text-sm flex-wrap">
            {story.coherence !== undefined && (
              <div className="flex items-center gap-1">
                <span className="text-slate-500">Coherence:</span>
                <span className="font-semibold text-indigo-600">
                  {(story.coherence * 100).toFixed(0)}%
                </span>
              </div>
            )}

            {story.claim_count !== undefined && (
              <div className="flex items-center gap-1">
                <span className="text-slate-500">Claims:</span>
                <span className="font-semibold text-slate-700">
                  {story.claim_count}
                </span>
              </div>
            )}

            {/* Time Information */}
            {timeInfo.absolute !== 'Date unknown' && (
              <div
                className="flex items-center gap-1 cursor-help"
                title={timeInfo.fullDateTime}
              >
                <span className="text-slate-400">•</span>
                <span className="text-slate-500">{timeInfo.relative}</span>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default NewsCard
