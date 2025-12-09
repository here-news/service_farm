import React from 'react'

function StoryCardSkeleton() {
  return (
    <div className="border border-slate-200 rounded-xl p-6 animate-pulse">
      <div className="flex gap-4">
        {/* Image skeleton */}
        <div className="w-24 h-24 bg-slate-200 rounded-lg flex-shrink-0" />

        <div className="flex-1 min-w-0">
          {/* Title skeleton */}
          <div className="h-6 bg-slate-200 rounded w-3/4 mb-3" />

          {/* Description skeleton */}
          <div className="space-y-2 mb-3">
            <div className="h-4 bg-slate-200 rounded w-full" />
            <div className="h-4 bg-slate-200 rounded w-5/6" />
          </div>

          {/* Metadata skeleton */}
          <div className="flex gap-4">
            <div className="h-4 bg-slate-200 rounded w-24" />
            <div className="h-4 bg-slate-200 rounded w-20" />
          </div>
        </div>
      </div>
    </div>
  )
}

export default StoryCardSkeleton
