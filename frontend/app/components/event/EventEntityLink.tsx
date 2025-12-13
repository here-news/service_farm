import React, { useState, useEffect, useContext } from 'react'
import { useNavigate } from 'react-router-dom'
import { ParagraphVisibilityContext } from './EventNarrativeContent'

export interface EventEntityData {
  id: string
  canonical_name: string
  entity_type: string
  aliases?: string[]
  mention_count?: number
  profile_summary?: string
  wikidata_qid?: string
  wikidata_label?: string
  wikidata_description?: string
  image_url?: string
  status?: string
  confidence?: number
}

interface EventEntityLinkProps {
  entityId: string
  displayName: string
  isFirstMention?: boolean
  // Pre-loaded entity data from parent (optional)
  preloadedEntity?: EventEntityData
  // Entities map from event data for quick lookup
  eventEntities?: Array<{
    id: string
    canonical_name: string
    entity_type: string
    wikidata_qid?: string
    wikidata_description?: string
    image_url?: string
    mention_count?: number
  }>
}

const entityTypeColors: Record<string, string> = {
  person: '#805ad5',
  organization: '#3182ce',
  location: '#38a169',
  event: '#d69e2e',
  default: '#718096'
}

function EventEntityLink({
  entityId,
  displayName,
  isFirstMention = false,
  preloadedEntity,
  eventEntities
}: EventEntityLinkProps) {
  const navigate = useNavigate()
  const [showTooltip, setShowTooltip] = useState(false)
  const [entityData, setEntityData] = useState<EventEntityData | null>(preloadedEntity || null)
  const [loading, setLoading] = useState(false)
  const [imageError, setImageError] = useState(false)

  // Get visibility from parent paragraph context
  const isInView = useContext(ParagraphVisibilityContext)

  const handleClick = () => {
    if (entityId && entityId.startsWith('en_')) {
      navigate(`/entity/${entityId}`)
    }
  }

  // Try to find entity in eventEntities first
  useEffect(() => {
    if (preloadedEntity) {
      setEntityData(preloadedEntity)
      return
    }

    if (eventEntities && entityId) {
      const found = eventEntities.find(e => e.id === entityId)
      if (found) {
        setEntityData({
          id: found.id,
          canonical_name: found.canonical_name,
          entity_type: found.entity_type,
          wikidata_qid: found.wikidata_qid,
          wikidata_description: found.wikidata_description,
          image_url: found.image_url,
          mention_count: found.mention_count,
        })
      }
    }
  }, [entityId, eventEntities, preloadedEntity])

  // Fetch full entity data on hover if not loaded
  const handleMouseEnter = async () => {
    setShowTooltip(true)

    // If we already have rich data, skip fetch
    if (entityData?.wikidata_description || entityData?.profile_summary) {
      return
    }

    // Only fetch if we have an entity ID
    if (!entityId || !entityId.startsWith('en_')) return

    setLoading(true)
    try {
      const response = await fetch(`/api/entities/${entityId}`)
      if (response.ok) {
        const data = await response.json()
        setEntityData(data.entity)
      }
    } catch (err) {
      console.error('Failed to fetch entity:', err)
    } finally {
      setLoading(false)
    }
  }

  const entityColor = entityData?.entity_type
    ? (entityTypeColors[entityData.entity_type.toLowerCase()] || entityTypeColors.default)
    : entityTypeColors.default

  const isPerson = entityData?.entity_type?.toLowerCase() === 'person'
  const hasImage = entityData?.image_url && !imageError

  // Only show headshot when paragraph is in view
  const showHeadshot = isFirstMention && hasImage && isInView

  return (
    <span style={{ position: 'relative', display: 'inline' }}>
      <span
        onClick={handleClick}
        onMouseEnter={handleMouseEnter}
        onMouseLeave={() => setShowTooltip(false)}
        style={{
          cursor: 'pointer',
          display: 'inline-flex',
          alignItems: 'center',
          gap: '6px',
          verticalAlign: 'baseline',
        }}
        title={entityId ? `View ${displayName} details` : displayName}
      >
        {/* Larger headshot - only shows when paragraph scrolls into view */}
        <span
          style={{
            display: 'inline-flex',
            alignItems: 'center',
            width: showHeadshot ? '32px' : '0px',
            height: showHeadshot ? '32px' : '0px',
            opacity: showHeadshot ? 1 : 0,
            transition: 'all 0.3s ease-out',
            overflow: 'hidden',
            flexShrink: 0,
          }}
        >
          {isFirstMention && hasImage && (
            <img
              src={entityData.image_url}
              alt={displayName}
              onError={() => setImageError(true)}
              style={{
                width: '32px',
                height: '32px',
                borderRadius: '50%',
                objectFit: 'cover',
                border: `2px solid ${entityColor}`,
                boxShadow: '0 2px 8px rgba(0,0,0,0.15)',
              }}
            />
          )}
        </span>
        <span
          style={{
            color: entityColor,
            fontWeight: 600,
            backgroundColor: isPerson ? 'rgba(128, 90, 213, 0.1)' : 'transparent',
            padding: isPerson ? '1px 6px' : '0',
            borderRadius: isPerson ? '4px' : '0',
            borderBottom: isPerson ? 'none' : `2px solid ${entityColor}`,
            transition: 'opacity 0.2s'
          }}
          className="hover:opacity-70"
        >
          {displayName}
        </span>
      </span>

      {/* Tooltip */}
      {showTooltip && (
        <div
          style={{
            position: 'absolute',
            bottom: '100%',
            left: '50%',
            transform: 'translateX(-50%)',
            marginBottom: '8px',
            backgroundColor: '#1a202c',
            color: 'white',
            padding: '12px',
            borderRadius: '8px',
            minWidth: '280px',
            maxWidth: '380px',
            zIndex: 1000,
            boxShadow: '0 4px 20px rgba(0,0,0,0.5)',
            pointerEvents: 'none',
            border: `1px solid ${entityColor}40`
          }}
        >
          {/* Arrow pointing down */}
          <div
            style={{
              position: 'absolute',
              top: '100%',
              left: '50%',
              transform: 'translateX(-50%)',
              width: 0,
              height: 0,
              borderLeft: '8px solid transparent',
              borderRight: '8px solid transparent',
              borderTop: '8px solid #1a202c'
            }}
          />

          {loading ? (
            <div className="flex items-center gap-2 text-gray-400">
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-500"></div>
              <span>Loading...</span>
            </div>
          ) : entityData ? (
            <div className="flex gap-3">
              {/* Larger image in tooltip */}
              {entityData.image_url && (
                <img
                  src={entityData.image_url}
                  alt={entityData.canonical_name}
                  className="w-16 h-16 rounded-lg object-cover flex-shrink-0 border-2"
                  style={{ borderColor: entityColor }}
                  onError={(e) => {
                    (e.target as HTMLImageElement).style.display = 'none'
                  }}
                />
              )}

              <div className="flex-1 min-w-0">
                {/* Name and type */}
                <div className="flex items-center gap-2 mb-1 flex-wrap">
                  <span className="font-semibold text-white">
                    {entityData.canonical_name}
                  </span>
                  {entityData.entity_type && (
                    <span
                      className="px-2 py-0.5 rounded text-xs font-medium text-white flex-shrink-0"
                      style={{ backgroundColor: entityColor }}
                    >
                      {entityData.entity_type}
                    </span>
                  )}
                </div>

                {/* Metadata */}
                <div className="text-xs text-gray-400 mb-1 flex flex-wrap gap-2">
                  {entityData.wikidata_qid && (
                    <span className="bg-gray-700 px-1.5 py-0.5 rounded">
                      {entityData.wikidata_qid}
                    </span>
                  )}
                  {entityData.mention_count !== undefined && entityData.mention_count > 0 && (
                    <span>{entityData.mention_count} mentions</span>
                  )}
                </div>

                {/* Description */}
                {(entityData.wikidata_description || entityData.profile_summary) && (
                  <div className="text-xs text-gray-300 line-clamp-3 leading-relaxed">
                    {entityData.wikidata_description || entityData.profile_summary}
                  </div>
                )}

                {/* Aliases */}
                {entityData.aliases && entityData.aliases.length > 0 && (
                  <div className="text-xs text-gray-500 mt-1 italic">
                    Also known as: {entityData.aliases.slice(0, 3).join(', ')}
                  </div>
                )}
              </div>
            </div>
          ) : (
            <div className="text-gray-400 text-sm">
              <span className="font-medium text-white">{displayName}</span>
              <p className="text-xs mt-1">No additional information available</p>
            </div>
          )}
        </div>
      )}
    </span>
  )
}

export default EventEntityLink
