import React, { useState } from 'react'

export interface EntityMetadata {
  name: string
  canonical_id?: string
  qid?: string
  wikidata_qid?: string
  description?: string
  wikidata_description?: string
  wikidata_thumbnail?: string
  entity_type?: string
  claim_count?: number
}

interface EntityLinkProps {
  name: string
  canonicalId?: string
  metadata?: EntityMetadata
}

const entityTypeColors: Record<string, string> = {
  person: '#805ad5',
  organization: '#3182ce',
  location: '#38a169',
  event: '#d69e2e',
  default: '#718096'
}

function EntityLink({ name, canonicalId, metadata }: EntityLinkProps) {
  const [showTooltip, setShowTooltip] = useState(false)

  const entityColor = metadata?.entity_type
    ? (entityTypeColors[metadata.entity_type.toLowerCase()] || entityTypeColors.default)
    : entityTypeColors.default

  const handleClick = (e: React.MouseEvent) => {
    e.preventDefault()
    // TODO: Navigate to entity page or show entity details modal
    console.log('Entity clicked:', name, canonicalId)
  }

  const isPerson = metadata?.entity_type?.toLowerCase() === 'person'

  return (
    <span style={{ position: 'relative', display: 'inline-block' }}>
      <span
        onClick={handleClick}
        onMouseEnter={() => setShowTooltip(true)}
        onMouseLeave={() => setShowTooltip(false)}
        style={{
          cursor: 'pointer',
          color: isPerson ? '#92400e' : entityColor,
          fontWeight: 500,
          backgroundColor: isPerson ? '#fef3c7' : 'transparent',
          padding: isPerson ? '2px 4px' : '0',
          borderRadius: isPerson ? '3px' : '0',
          borderBottom: isPerson ? 'none' : `2px solid ${entityColor}`,
          transition: 'opacity 0.2s'
        }}
        className="hover:opacity-70"
      >
        {name}
      </span>

      {/* Tooltip */}
      {showTooltip && metadata && (
        <div
          style={{
            position: 'absolute',
            bottom: '100%',
            left: '50%',
            transform: 'translateX(-50%)',
            marginBottom: '8px',
            backgroundColor: '#2d3748',
            color: 'white',
            padding: '12px',
            borderRadius: '8px',
            minWidth: '250px',
            maxWidth: '350px',
            zIndex: 1000,
            boxShadow: '0 4px 12px rgba(0,0,0,0.3)',
            pointerEvents: 'none'
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
              borderLeft: '6px solid transparent',
              borderRight: '6px solid transparent',
              borderTop: '6px solid #2d3748'
            }}
          />

          <div className="flex gap-3">
            {/* Thumbnail */}
            {metadata.wikidata_thumbnail && (
              <img
                src={metadata.wikidata_thumbnail}
                alt={metadata.name}
                className="w-16 h-16 rounded-lg object-cover flex-shrink-0 border-2"
                style={{ borderColor: entityColor }}
              />
            )}

            <div className="flex-1 min-w-0">
              {/* Name and type */}
              <div className="flex items-center gap-2 mb-1">
                <span className="font-semibold text-blue-300 truncate">
                  {metadata.name}
                </span>
                {metadata.entity_type && (
                  <span
                    className="px-2 py-0.5 rounded text-xs font-medium text-white flex-shrink-0"
                    style={{ backgroundColor: entityColor }}
                  >
                    {metadata.entity_type}
                  </span>
                )}
              </div>

              {/* Metadata */}
              <div className="text-xs text-gray-300 mb-1">
                {metadata.wikidata_qid && (
                  <span>{metadata.wikidata_qid}</span>
                )}
                {metadata.claim_count !== undefined && (
                  <span className="ml-2">• {metadata.claim_count} claims</span>
                )}
              </div>

              {/* Description */}
              {(metadata.description || metadata.wikidata_description) && (
                <div className="text-xs text-gray-200 line-clamp-3">
                  {metadata.description || metadata.wikidata_description}
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </span>
  )
}

export default EntityLink
