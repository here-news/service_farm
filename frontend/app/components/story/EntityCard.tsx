import React from 'react'
import { Entity } from '../../types/story'

interface EntityCardProps {
  entity: Entity
  type: 'person' | 'organization' | 'location'
}

function EntityCard({ entity, type }: EntityCardProps) {
  const typeIcons = {
    person: '👤',
    organization: '🏢',
    location: '📍'
  }

  const typeColors = {
    person: 'from-blue-500 to-indigo-500',
    organization: 'from-purple-500 to-pink-500',
    location: 'from-green-500 to-teal-500'
  }

  return (
    <div className="border border-slate-200 rounded-lg p-4 hover:shadow-md transition">
      <div className="flex gap-3">
        {entity.wikidata_thumbnail ? (
          <img
            src={entity.wikidata_thumbnail}
            alt={entity.name}
            className="w-16 h-16 rounded-lg object-cover flex-shrink-0"
          />
        ) : (
          <div className={`w-16 h-16 rounded-lg bg-gradient-to-br ${typeColors[type]} flex items-center justify-center text-2xl flex-shrink-0`}>
            {typeIcons[type]}
          </div>
        )}

        <div className="flex-1 min-w-0">
          <h4 className="font-semibold text-slate-900 mb-1">{entity.name}</h4>
          {(entity.description || entity.wikidata_description) && (
            <p className="text-sm text-slate-600 line-clamp-2">
              {entity.description || entity.wikidata_description}
            </p>
          )}
          {entity.wikidata_qid && (
            <a
              href={`https://www.wikidata.org/wiki/${entity.wikidata_qid}`}
              target="_blank"
              rel="noopener noreferrer"
              className="text-xs text-indigo-600 hover:text-indigo-700 mt-1 inline-block"
            >
              View on Wikidata →
            </a>
          )}
        </div>
      </div>
    </div>
  )
}

export default EntityCard
