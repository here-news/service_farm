import React, { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'

interface StorySummary {
  id: string
  scale: 'incident' | 'case'
  title: string
  description: string
  primary_entity: string
  time_start?: string
  source_count: number
  surface_count: number
  incident_count?: number
}

interface EntitySummary {
  id: string
  canonical_name: string
  entity_type?: string
  narrative?: string
  profile_summary?: string
  image_url?: string
  source_count: number
  surface_count: number
  claim_count?: number
  related_events?: string[]
  last_active?: string
}

export default function TempHomePage() {
  const [stories, setStories] = useState<StorySummary[]>([])
  const [entities, setEntities] = useState<EntitySummary[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const loadData = async () => {
      try {
        const [storiesRes, entitiesRes] = await Promise.all([
          fetch('/api/stories?scale=case&limit=50'),
          fetch('/api/entities?limit=50')
        ])

        if (storiesRes.ok) {
          const data = await storiesRes.json()
          setStories(data.stories || [])
        }

        if (entitiesRes.ok) {
          const data = await entitiesRes.json()
          setEntities(data.entities || [])
        }
      } catch (err) {
        console.error('Failed to load data:', err)
      } finally {
        setLoading(false)
      }
    }

    loadData()
  }, [])

  if (loading) {
    return (
      <div className="flex items-center justify-center py-20">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600" />
      </div>
    )
  }

  return (
    <div className="max-w-7xl mx-auto px-4 py-6">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-slate-800">Browse Data</h1>
        <p className="text-slate-500 text-sm mt-1">
          Temporary index page - click to view event or entity details
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Stories Column */}
        <div>
          <div className="flex items-center gap-2 mb-4">
            <div className="w-3 h-3 rounded-full bg-red-500" />
            <h2 className="text-lg font-semibold text-slate-700">Stories ({stories.length})</h2>
          </div>
          <div className="space-y-2">
            {stories.map(story => (
              <Link
                key={story.id}
                to={`/story/${story.id}`}
                className="block p-4 bg-white rounded-lg border border-slate-200 hover:border-indigo-300 hover:shadow-sm transition"
              >
                <div className="flex items-center gap-2 mb-1">
                  <span className="font-medium text-slate-800">{story.title}</span>
                  <span className="text-xs px-1.5 py-0.5 bg-slate-100 text-slate-500 rounded">
                    {story.scale}
                  </span>
                </div>
                <div className="text-sm text-slate-500 line-clamp-2 mb-2">
                  {story.description}
                </div>
                <div className="flex items-center gap-4 text-xs text-slate-400">
                  <span>{story.primary_entity}</span>
                  <span>{story.source_count} sources</span>
                  <span>{story.surface_count} surfaces</span>
                  {story.incident_count && <span>{story.incident_count} incidents</span>}
                  {story.time_start && (
                    <span>{new Date(story.time_start).toLocaleDateString()}</span>
                  )}
                </div>
              </Link>
            ))}
            {stories.length === 0 && (
              <div className="text-slate-400 text-center py-8">No stories found</div>
            )}
          </div>
        </div>

        {/* Entities Column */}
        <div>
          <div className="flex items-center gap-2 mb-4">
            <div className="w-3 h-3 rounded-full bg-indigo-500" />
            <h2 className="text-lg font-semibold text-slate-700">Entities ({entities.length})</h2>
          </div>
          <div className="space-y-2">
            {entities.map(entity => (
              <Link
                key={entity.id}
                to={`/entity/${entity.id}`}
                className="block p-4 bg-white rounded-lg border border-slate-200 hover:border-indigo-300 hover:shadow-sm transition"
              >
                <div className="flex gap-3">
                  {entity.image_url && (
                    <img
                      src={entity.image_url}
                      alt={entity.canonical_name}
                      className="w-12 h-12 rounded-full object-cover flex-shrink-0"
                      onError={(e) => (e.currentTarget.style.display = 'none')}
                    />
                  )}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      <span className="font-medium text-slate-800">{entity.canonical_name}</span>
                      {entity.entity_type && (
                        <span className="text-xs px-1.5 py-0.5 bg-slate-100 text-slate-500 rounded">
                          {entity.entity_type}
                        </span>
                      )}
                    </div>
                    {entity.profile_summary && (
                      <div className="text-sm text-slate-600 mb-1">{entity.profile_summary}</div>
                    )}
                    {entity.narrative && !entity.profile_summary && (
                      <div className="text-sm text-slate-500 line-clamp-2 mb-1">
                        {entity.narrative}
                      </div>
                    )}
                    <div className="flex items-center gap-3 text-xs text-slate-400">
                      <span>{entity.source_count} sources</span>
                      <span>{entity.claim_count || entity.surface_count} claims</span>
                      {entity.related_events && entity.related_events.length > 0 && (
                        <span className="text-indigo-500">{entity.related_events.length} events</span>
                      )}
                    </div>
                  </div>
                </div>
              </Link>
            ))}
            {entities.length === 0 && (
              <div className="text-slate-400 text-center py-8">No entities found</div>
            )}
          </div>
        </div>
      </div>

      {/* Demo links */}
      <div className="mt-8 p-4 bg-slate-100 rounded-lg">
        <h3 className="text-sm font-medium text-slate-600 mb-2">Demo Pages (simulated data)</h3>
        <div className="flex gap-4">
          <Link to="/event/wang-fuk-court-fire" className="text-indigo-600 hover:underline text-sm">
            Wang Fuk Court Fire
          </Link>
          <Link to="/entity/elon-musk" className="text-indigo-600 hover:underline text-sm">
            Elon Musk
          </Link>
        </div>
      </div>
    </div>
  )
}
