import React, { useState, useEffect } from 'react'

interface URLPreviewProps {
  url: string
}

interface URLMetadata {
  task_id: string
  url: string
  status: string
  title?: string
  description?: string
  image?: string
  domain?: string
  favicon?: string
  site_name?: string
}

function URLPreview({ url }: URLPreviewProps) {
  const [metadata, setMetadata] = useState<URLMetadata | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(false)

  useEffect(() => {
    fetchMetadata()
  }, [url])

  const fetchMetadata = async () => {
    try {
      setLoading(true)
      setError(false)

      // Submit URL for preview
      const response = await fetch('/api/preview', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url })
      })

      if (!response.ok) {
        setError(true)
        setLoading(false)
        return
      }

      const data = await response.json()
      setMetadata(data)

      // Handle different statuses
      if (data.status === 'fetching') {
        // Still fetching, poll for status
        pollForStatus(data.task_id)
      } else if (data.status === 'failed' || data.status === 'timeout') {
        // Preview failed, show fallback
        setError(true)
        setLoading(false)
      } else {
        // Completed or extracting with preview data
        setLoading(false)
      }
    } catch (err) {
      console.error('Failed to fetch URL metadata:', err)
      setError(true)
      setLoading(false)
    }
  }

  const pollForStatus = async (taskId: string) => {
    let attempts = 0
    const maxAttempts = 10
    const interval = 2000 // 2 seconds

    const poll = async () => {
      try {
        const response = await fetch(`/api/preview/status/${taskId}`)

        if (response.ok) {
          const data = await response.json()
          setMetadata(data)

          if (data.status === 'completed') {
            setLoading(false)
            return
          }

          if (data.status === 'failed') {
            setError(true)
            setLoading(false)
            return
          }
        }

        attempts++
        if (attempts < maxAttempts) {
          setTimeout(poll, interval)
        } else {
          setLoading(false)
        }
      } catch (err) {
        console.error('Polling error:', err)
        setLoading(false)
      }
    }

    setTimeout(poll, interval)
  }

  if (loading) {
    return (
      <div className="mt-2 border border-slate-200 rounded-lg p-3 bg-slate-50 animate-pulse">
        <div className="h-4 bg-slate-200 rounded w-3/4 mb-2"></div>
        <div className="h-3 bg-slate-200 rounded w-1/2"></div>
      </div>
    )
  }

  if (error || !metadata) {
    // Fallback to simple link if preview fails
    return (
      <a
        href={url}
        target="_blank"
        rel="noopener noreferrer"
        className="mt-2 flex items-center gap-2 text-sm text-indigo-600 hover:text-indigo-800 underline"
      >
        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
        </svg>
        {new URL(url).hostname}
      </a>
    )
  }

  return (
    <a
      href={url}
      target="_blank"
      rel="noopener noreferrer"
      className="mt-2 flex gap-3 border border-slate-200 rounded-lg overflow-hidden bg-white hover:border-indigo-300 hover:shadow-md transition-all p-3"
    >
      {metadata.image && (
        <img
          src={metadata.image}
          alt={metadata.title || url}
          className="w-20 h-20 rounded object-cover flex-shrink-0"
          onError={(e) => {
            e.currentTarget.style.display = 'none'
          }}
        />
      )}

      <div className="flex-1 min-w-0">
        {metadata.site_name && (
          <div className="text-xs text-slate-500 mb-1">{metadata.site_name}</div>
        )}

        <div className="font-semibold text-slate-900 text-sm line-clamp-2 mb-1">
          {metadata.title || url}
        </div>

        {metadata.description && (
          <div className="text-xs text-slate-600 line-clamp-2 mb-1">
            {metadata.description}
          </div>
        )}

        <div className="text-xs text-indigo-600 flex items-center gap-1">
          <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
          </svg>
          {new URL(url).hostname}
        </div>
      </div>
    </a>
  )
}

export default URLPreview
