import React from 'react'
import { useNavigate } from 'react-router-dom'
import { RelatedStory } from '../../types/story'

interface RelatedStoriesProps {
  stories: RelatedStory[]
}

function RelatedStories({ stories }: RelatedStoriesProps) {
  const navigate = useNavigate()

  if (stories.length === 0) {
    return (
      <div className="text-center py-8 text-slate-400">
        No related stories found
      </div>
    )
  }

  return (
    <div className="space-y-3">
      {stories.map((story) => (
        <div
          key={story.id}
          onClick={() => navigate(`/story/${story.id}`)}
          className="border border-slate-200 rounded-lg p-4 hover:shadow-md hover:border-indigo-300 transition cursor-pointer"
        >
          <div className="flex items-start justify-between gap-3">
            <div className="flex-1">
              <h4 className="font-medium text-slate-900 mb-2 line-clamp-2">
                {story.title}
              </h4>
              <div className="flex items-center gap-2 text-xs text-slate-500">
                <span className="inline-flex items-center gap-1">
                  <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
                    <path d="M13 6a3 3 0 11-6 0 3 3 0 016 0zM18 8a2 2 0 11-4 0 2 2 0 014 0zM14 15a4 4 0 00-8 0v3h8v-3zM6 8a2 2 0 11-4 0 2 2 0 014 0zM16 18v-3a5.972 5.972 0 00-.75-2.906A3.005 3.005 0 0119 15v3h-3zM4.75 12.094A5.973 5.973 0 004 15v3H1v-3a3 3 0 013.75-2.906z" />
                  </svg>
                  {story.shared_entities} shared {story.shared_entities === 1 ? 'entity' : 'entities'}
                </span>
              </div>
            </div>
            <svg className="w-5 h-5 text-slate-400 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
            </svg>
          </div>
        </div>
      ))}
    </div>
  )
}

export default RelatedStories
