import React, { useState } from 'react'

export interface ClaimSource {
  page_id: string
  url: string
  title: string
  site_name?: string
  domain?: string
}

export interface ClaimEntity {
  id: string
  canonical_name: string
  entity_type: string
  wikidata_qid?: string
}

export interface EventClaimData {
  id: string
  text: string
  event_time?: string | null
  confidence?: number
  modality?: string
  topic_key?: string
  page_id?: string
  source?: ClaimSource | null
  entities?: ClaimEntity[]
}

interface EventClaimLinkProps {
  claimId: string
  displayText?: string
  // Pre-loaded claim data from parent (optional)
  preloadedClaim?: EventClaimData
  // Claims list from event data for quick lookup
  eventClaims?: Array<{
    id: string
    text: string
    event_time?: string
    confidence?: number
  }>
}

const modalityColors: Record<string, { bg: string; text: string }> = {
  observation: { bg: '#065f46', text: '#a7f3d0' },
  prediction: { bg: '#1e40af', text: '#bfdbfe' },
  speculation: { bg: '#92400e', text: '#fde68a' },
  opinion: { bg: '#7c3aed', text: '#ddd6fe' },
  default: { bg: '#374151', text: '#d1d5db' }
}

function EventClaimLink({
  claimId,
  displayText,
  preloadedClaim,
  eventClaims
}: EventClaimLinkProps) {
  const [showPopup, setShowPopup] = useState(false)
  const [claimData, setClaimData] = useState<EventClaimData | null>(preloadedClaim || null)
  const [loading, setLoading] = useState(false)

  // Find claim in eventClaims on mount
  React.useEffect(() => {
    if (preloadedClaim) {
      setClaimData(preloadedClaim)
      return
    }

    if (eventClaims && claimId) {
      const found = eventClaims.find(c => c.id === claimId)
      if (found) {
        setClaimData({
          id: found.id,
          text: found.text,
          event_time: found.event_time,
          confidence: found.confidence,
        })
      }
    }
  }, [claimId, eventClaims, preloadedClaim])

  // Fetch full claim data (with source) on click
  const handleClick = async (e: React.MouseEvent) => {
    e.preventDefault()
    e.stopPropagation()

    setShowPopup(!showPopup)

    // If we already have full data with source, skip fetch
    if (claimData?.source !== undefined) {
      return
    }

    if (!claimId || !claimId.startsWith('cl_')) return

    setLoading(true)
    try {
      const response = await fetch(`/api/claims/${claimId}`)
      if (response.ok) {
        const data = await response.json()
        setClaimData({
          ...data.claim,
          source: data.source,
          entities: data.entities
        })
      }
    } catch (err) {
      console.error('Failed to fetch claim:', err)
    } finally {
      setLoading(false)
    }
  }

  // Close popup when clicking outside
  React.useEffect(() => {
    if (!showPopup) return

    const handleClickOutside = (e: MouseEvent) => {
      const target = e.target as HTMLElement
      if (!target.closest('.claim-popup-container')) {
        setShowPopup(false)
      }
    }

    document.addEventListener('click', handleClickOutside)
    return () => document.removeEventListener('click', handleClickOutside)
  }, [showPopup])

  const formatDateTime = (dateStr: string | null | undefined) => {
    if (!dateStr) return null
    try {
      const date = new Date(dateStr)
      return date.toLocaleString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
      })
    } catch {
      return dateStr
    }
  }

  const modalityStyle = claimData?.modality
    ? modalityColors[claimData.modality] || modalityColors.default
    : modalityColors.default

  // Display as superscript citation marker
  const markerText = displayText || `[${claimId.replace('cl_', '').slice(0, 4)}]`

  return (
    <span className="claim-popup-container" style={{ position: 'relative', display: 'inline' }}>
      <sup
        onClick={handleClick}
        style={{
          cursor: 'pointer',
          color: '#60a5fa',
          fontWeight: 600,
          fontSize: '0.7em',
          marginLeft: '1px',
          padding: '0 2px',
          borderRadius: '2px',
          backgroundColor: showPopup ? 'rgba(96, 165, 250, 0.2)' : 'transparent',
          transition: 'all 0.2s'
        }}
        className="hover:bg-blue-500/20"
        title={claimData?.text || 'Click to view claim'}
      >
        {markerText}
      </sup>

      {/* Popup */}
      {showPopup && (
        <div
          style={{
            position: 'absolute',
            bottom: '100%',
            left: '50%',
            transform: 'translateX(-50%)',
            marginBottom: '12px',
            backgroundColor: '#111827',
            color: 'white',
            padding: '16px',
            borderRadius: '12px',
            minWidth: '320px',
            maxWidth: '450px',
            zIndex: 1001,
            boxShadow: '0 8px 30px rgba(0,0,0,0.6)',
            border: '1px solid #374151'
          }}
          onClick={(e) => e.stopPropagation()}
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
              borderLeft: '10px solid transparent',
              borderRight: '10px solid transparent',
              borderTop: '10px solid #111827'
            }}
          />

          {/* Close button */}
          <button
            onClick={() => setShowPopup(false)}
            style={{
              position: 'absolute',
              top: '8px',
              right: '8px',
              background: 'none',
              border: 'none',
              color: '#9ca3af',
              cursor: 'pointer',
              fontSize: '16px',
              lineHeight: 1,
              padding: '4px'
            }}
            className="hover:text-white"
          >
            x
          </button>

          {loading ? (
            <div className="flex items-center gap-2 text-gray-400 py-4">
              <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-blue-500"></div>
              <span>Loading claim details...</span>
            </div>
          ) : claimData ? (
            <div>
              {/* Claim text */}
              <div className="text-gray-200 text-sm leading-relaxed mb-3 pr-6">
                "{claimData.text}"
              </div>

              {/* Metadata row */}
              <div className="flex flex-wrap gap-2 mb-3">
                {/* Event time */}
                {claimData.event_time && (
                  <div className="flex items-center gap-1 text-xs bg-gray-800 px-2 py-1 rounded">
                    <span className="text-gray-500">When:</span>
                    <span className="text-gray-300">{formatDateTime(claimData.event_time)}</span>
                  </div>
                )}

                {/* Confidence */}
                {claimData.confidence !== undefined && (
                  <div className="flex items-center gap-1 text-xs bg-gray-800 px-2 py-1 rounded">
                    <span className="text-gray-500">Confidence:</span>
                    <span className="text-green-400">{Math.round(claimData.confidence * 100)}%</span>
                  </div>
                )}

                {/* Modality */}
                {claimData.modality && (
                  <span
                    className="text-xs px-2 py-1 rounded font-medium"
                    style={{
                      backgroundColor: modalityStyle.bg,
                      color: modalityStyle.text
                    }}
                  >
                    {claimData.modality}
                  </span>
                )}

                {/* Topic key */}
                {claimData.topic_key && (
                  <span className="text-xs bg-purple-900/50 text-purple-300 px-2 py-1 rounded">
                    {claimData.topic_key}
                  </span>
                )}
              </div>

              {/* Source */}
              {claimData.source && (
                <div className="border-t border-gray-700 pt-3 mt-3">
                  <div className="text-xs text-gray-500 mb-1">Source:</div>
                  <a
                    href={claimData.source.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-sm text-blue-400 hover:text-blue-300 hover:underline block truncate"
                    title={claimData.source.url}
                  >
                    {claimData.source.title || claimData.source.site_name || claimData.source.domain || claimData.source.url}
                  </a>
                  {claimData.source.site_name && claimData.source.title && (
                    <div className="text-xs text-gray-500 mt-1">
                      {claimData.source.site_name}
                    </div>
                  )}
                </div>
              )}

              {/* Mentioned entities */}
              {claimData.entities && claimData.entities.length > 0 && (
                <div className="border-t border-gray-700 pt-3 mt-3">
                  <div className="text-xs text-gray-500 mb-2">Mentioned entities:</div>
                  <div className="flex flex-wrap gap-1">
                    {claimData.entities.map(entity => (
                      <span
                        key={entity.id}
                        className="text-xs px-2 py-0.5 rounded bg-gray-800 text-gray-300"
                        title={entity.entity_type}
                      >
                        {entity.canonical_name}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {/* Claim ID */}
              <div className="text-xs text-gray-600 mt-3 pt-2 border-t border-gray-800">
                ID: {claimData.id}
              </div>
            </div>
          ) : (
            <div className="text-gray-400 text-sm py-2">
              Unable to load claim details
            </div>
          )}
        </div>
      )}
    </span>
  )
}

export default EventClaimLink
