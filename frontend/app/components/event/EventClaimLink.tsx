import React, { useState, useRef } from 'react'
import { useNavigate } from 'react-router-dom'

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
    page_id?: string
  }>
}

function EventClaimLink({
  claimId,
  displayText: _displayText, // Reserved for future use
  preloadedClaim,
  eventClaims
}: EventClaimLinkProps) {
  const navigate = useNavigate()
  const [showPopup, setShowPopup] = useState(false)
  const [claimData, setClaimData] = useState<EventClaimData | null>(preloadedClaim || null)
  const [loading, setLoading] = useState(false)
  const hoverTimeoutRef = useRef<NodeJS.Timeout | null>(null)

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
          page_id: found.page_id,
        })
      }
    }
  }, [claimId, eventClaims, preloadedClaim])

  // Fetch full claim data on hover (for popup)
  const fetchClaimData = async () => {
    if (claimData?.source !== undefined) return
    if (!claimId || !claimId.startsWith('cl_')) return

    setLoading(true)
    try {
      const response = await fetch(`/api/claim/${claimId}`)
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

  // Handle hover - show popup after short delay
  const handleMouseEnter = () => {
    hoverTimeoutRef.current = setTimeout(() => {
      setShowPopup(true)
      fetchClaimData()
    }, 300) // 300ms delay before showing popup
  }

  const handleMouseLeave = () => {
    if (hoverTimeoutRef.current) {
      clearTimeout(hoverTimeoutRef.current)
      hoverTimeoutRef.current = null
    }
    setShowPopup(false)
  }

  // Handle click - navigate to page with claim anchor
  const handleClick = (e: React.MouseEvent) => {
    e.preventDefault()
    e.stopPropagation()

    const pageId = claimData?.source?.page_id || claimData?.page_id
    if (pageId) {
      navigate(`/page/${pageId}#${claimId}`)
    } else {
      // If no page_id yet, fetch it first
      fetchClaimData().then(() => {
        const pid = claimData?.source?.page_id || claimData?.page_id
        if (pid) {
          navigate(`/page/${pid}#${claimId}`)
        }
      })
    }
  }

  // Cleanup timeout on unmount
  React.useEffect(() => {
    return () => {
      if (hoverTimeoutRef.current) {
        clearTimeout(hoverTimeoutRef.current)
      }
    }
  }, [])


  // Display as a subtle but visible citation dot
  // Hover: shows popup with claim preview
  // Click: navigates to /page/{pg_id}/#cl_id
  return (
    <span
      className="claim-popup-container"
      style={{ position: 'relative', display: 'inline' }}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
    >
      <span
        onClick={handleClick}
        style={{
          cursor: 'pointer',
          display: 'inline-block',
          width: '8px',
          height: '8px',
          borderRadius: '50%',
          backgroundColor: showPopup ? '#4f46e5' : '#818cf8',
          marginLeft: '2px',
          marginRight: '1px',
          verticalAlign: 'super',
          transition: 'all 0.15s',
          boxShadow: showPopup ? '0 0 0 3px rgba(99, 102, 241, 0.3)' : 'none'
        }}
        className="hover:bg-indigo-600 hover:scale-125"
        title="Click to view source"
      />

      {/* Hover Popup - compact preview */}
      {showPopup && (
        <div
          style={{
            position: 'absolute',
            bottom: '100%',
            left: '50%',
            transform: 'translateX(-50%)',
            marginBottom: '8px',
            backgroundColor: '#1f2937',
            color: 'white',
            padding: '10px 12px',
            borderRadius: '8px',
            minWidth: '250px',
            maxWidth: '350px',
            zIndex: 1001,
            boxShadow: '0 4px 20px rgba(0,0,0,0.4)',
            border: '1px solid #374151',
            pointerEvents: 'none' // Popup doesn't capture mouse - allows click through
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
              borderTop: '6px solid #1f2937'
            }}
          />

          {loading ? (
            <div className="flex items-center gap-2 text-gray-400 text-sm">
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-indigo-400"></div>
              <span>Loading...</span>
            </div>
          ) : claimData ? (
            <div>
              {/* Claim text - compact */}
              <div className="text-gray-200 text-sm leading-snug line-clamp-3">
                "{claimData.text}"
              </div>

              {/* Compact metadata row */}
              <div className="flex items-center gap-2 mt-2 text-xs text-gray-400">
                {claimData.confidence !== undefined && (
                  <span className="text-green-400">
                    {Math.round(claimData.confidence * 100)}%
                  </span>
                )}
                {claimData.source?.site_name && (
                  <span className="truncate">
                    {claimData.source.site_name}
                  </span>
                )}
                <span className="text-indigo-400 ml-auto">Click to view</span>
              </div>
            </div>
          ) : (
            <div className="text-gray-400 text-sm">
              Click to view source
            </div>
          )}
        </div>
      )}
    </span>
  )
}

export default EventClaimLink
