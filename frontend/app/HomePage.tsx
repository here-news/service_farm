import React, { useState, useEffect, useRef, useCallback } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import Header from './components/layout/Header'
import NewsCard from './components/cards/NewsCard'
import StoryCardSkeleton from './components/cards/StoryCardSkeleton'
import ShareBox from './components/home/ShareBox'
import PendingSubmission, { EventSubmission } from './components/home/PendingSubmission'
import { Story as Event, FeedResponse } from './types/story'
import { useSubmissionPolling } from './hooks/useSubmissionPolling'

// Union type for feed items
type FeedItem =
  | { type: 'event'; data: Event; timestamp: string }
  | { type: 'submission'; data: EventSubmission; timestamp: string }

function HomePage() {
  const [events, setEvents] = useState<Event[]>([])
  const [loading, setLoading] = useState(true)
  const [loadingMore, setLoadingMore] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [hasMore, setHasMore] = useState(true)
  const [newEventsCount, setNewEventsCount] = useState(0)

  // ShareBox state
  const [showShareBox, setShowShareBox] = useState(false)
  const [pendingSubmissions, setPendingSubmissions] = useState<EventSubmission[]>([])

  // Merged feed (stories + submissions)
  const [mergedFeed, setMergedFeed] = useState<FeedItem[]>([])

  // Filters
  const [minCoherence, setMinCoherence] = useState(0.0)
  const [debouncedCoherence, setDebouncedCoherence] = useState(0.0)

  const navigate = useNavigate()
  const [searchParams, setSearchParams] = useSearchParams()
  const pageSize = 12
  const scrollThreshold = 1000
  const refreshIntervalRef = useRef<NodeJS.Timeout | null>(null)
  const coherenceDebounceRef = useRef<NodeJS.Timeout | null>(null)

  // Handle submission updates from polling
  const handleSubmissionUpdate = useCallback((updatedSubmissions: EventSubmission[]) => {
    setPendingSubmissions(updatedSubmissions)
  }, [])

  // Enable polling for active submissions
  useSubmissionPolling({
    submissions: pendingSubmissions,
    onUpdate: handleSubmissionUpdate,
    enabled: true
  })

  useEffect(() => {
    loadPreferences()
    loadFeed(true)
    loadPendingSubmissions()
    startBackgroundRefresh()
    window.addEventListener('scroll', handleScroll)

    return () => {
      stopBackgroundRefresh()
      window.removeEventListener('scroll', handleScroll)
    }
  }, [])

  // Check for share URL parameter
  useEffect(() => {
    if (searchParams.get('share') === 'true') {
      setShowShareBox(true)
      // Remove the parameter from URL
      searchParams.delete('share')
      setSearchParams(searchParams, { replace: true })
    }
  }, [searchParams, setSearchParams])

  useEffect(() => {
    if (debouncedCoherence !== minCoherence) {
      loadFeed(true)
    }
  }, [debouncedCoherence])

  // Merge events and submissions by timestamp
  useEffect(() => {
    // Defensive checks to ensure arrays
    const safeEvents = Array.isArray(events) ? events : []
    const safePendingSubmissions = Array.isArray(pendingSubmissions) ? pendingSubmissions : []

    const merged: FeedItem[] = [
      // Convert events to feed items
      ...safeEvents.map(event => ({
        type: 'event' as const,
        data: event,
        timestamp: event.last_updated || event.created_at || ''
      })),
      // Convert submissions to feed items
      ...safePendingSubmissions.map(submission => ({
        type: 'submission' as const,
        data: submission,
        timestamp: submission.created_at
      }))
    ]

    // Sort by timestamp (newest first)
    merged.sort((a, b) => {
      const dateA = new Date(a.timestamp).getTime()
      const dateB = new Date(b.timestamp).getTime()
      return dateB - dateA
    })

    setMergedFeed(merged)
  }, [events, pendingSubmissions])

  const loadPreferences = () => {
    const saved = localStorage.getItem('story_min_coherence')
    if (saved) {
      const coherence = parseFloat(saved)
      setMinCoherence(coherence)
      setDebouncedCoherence(coherence)
    }
  }

  const savePreferences = (coherence: number) => {
    localStorage.setItem('story_min_coherence', coherence.toString())
  }

  const handleCoherenceChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = parseFloat(e.target.value)
    setMinCoherence(value)

    if (coherenceDebounceRef.current) {
      clearTimeout(coherenceDebounceRef.current)
    }

    coherenceDebounceRef.current = setTimeout(() => {
      setDebouncedCoherence(value)
      savePreferences(value)
    }, 500)
  }

  const loadFeed = async (initial = false) => {
    try {
      if (initial) {
        setLoading(true)
        setEvents([])
      }

      const params = new URLSearchParams({
        limit: pageSize.toString(),
        min_coherence: debouncedCoherence.toString()
      })

      const response = await fetch(`/api/events?${params}`)
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`)
      }
      const data: FeedResponse = await response.json()
      const eventsArray = Array.isArray(data.events) ? data.events : []
      setEvents(eventsArray)
      setHasMore(eventsArray.length === pageSize)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load feed')
    } finally {
      setLoading(false)
    }
  }

  const loadMore = async () => {
    if (loadingMore || !hasMore) return

    setLoadingMore(true)
    try {
      const params = new URLSearchParams({
        limit: pageSize.toString(),
        offset: events.length.toString(),
        min_coherence: debouncedCoherence.toString()
      })

      const response = await fetch(`/api/events?${params}`)
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`)
      }
      const data: FeedResponse = await response.json()
      const newEvents = Array.isArray(data.events) ? data.events : []
      setEvents([...events, ...newEvents])
      setHasMore(newEvents.length === pageSize)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load more events')
    } finally {
      setLoadingMore(false)
    }
  }

  const handleScroll = () => {
    const scrollHeight = document.documentElement.scrollHeight
    const scrollTop = document.documentElement.scrollTop
    const clientHeight = document.documentElement.clientHeight
    const distanceFromBottom = scrollHeight - scrollTop - clientHeight

    if (distanceFromBottom < scrollThreshold) {
      loadMore()
    }
  }

  const startBackgroundRefresh = () => {
    refreshIntervalRef.current = setInterval(() => {
      checkNewEvents()
    }, 30000) // 30 seconds
  }

  const stopBackgroundRefresh = () => {
    if (refreshIntervalRef.current) {
      clearInterval(refreshIntervalRef.current)
    }
  }

  const checkNewEvents = async () => {
    try {
      const params = new URLSearchParams({
        limit: pageSize.toString(),
        min_coherence: debouncedCoherence.toString()
      })

      const response = await fetch(`/api/events?${params}`)
      if (!response.ok) return

      const data: FeedResponse = await response.json()
      const newEvents = data.events || []

      // Defensive check: ensure events is an array
      if (!Array.isArray(events)) {
        console.warn('events is not an array:', events)
        return
      }

      if (newEvents.length > 0 && events.length > 0) {
        const firstNewId = newEvents[0].id || newEvents[0].event_id || newEvents[0].story_id
        const hasNew = !events.some(e => {
          const eId = e.id || e.event_id || e.story_id
          return eId === firstNewId
        })

        if (hasNew) {
          const newCount = newEvents.findIndex(e => {
            const eId = e.id || e.event_id || e.story_id
            return events.some(existing => {
              const existingId = existing.id || existing.event_id || existing.story_id
              return existingId === eId
            })
          })
          setNewEventsCount(newCount === -1 ? newEvents.length : newCount)

          // Auto-hide after 5 seconds
          setTimeout(() => {
            setNewEventsCount(0)
          }, 5000)
        }
      }
    } catch (err) {
      console.error('Background refresh failed:', err)
    }
  }

  const refreshFeed = () => {
    setNewEventsCount(0)
    loadFeed(true)
  }

  const handleEventClick = (eventId: string) => {
    navigate(`/event/${eventId}`)
  }

  const loadPendingSubmissions = async () => {
    try {
      const response = await fetch('/api/events/mine', {
        credentials: 'include'
      })
      if (response.ok) {
        const data = await response.json()
        console.log('Submissions data:', data)
        if (data.length > 0) {
          console.log('First submission user_picture:', data[0].user_picture)
        }
        setPendingSubmissions(data)
      }
    } catch (err) {
      console.error('Failed to load pending submissions:', err)
    }
  }

  const handleSubmitEvent = async (content: string, urls: string[]) => {
    const response = await fetch('/api/events', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'include',
      body: JSON.stringify({
        content,
        urls: urls.join(',')
      })
    })

    if (!response.ok) {
      const error = await response.json()
      throw new Error(error.detail || 'Failed to submit event')
    }

    const data = await response.json()

    // Add to pending submissions
    setPendingSubmissions([data, ...pendingSubmissions])

    // Close sharebox
    setShowShareBox(false)
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-500 via-purple-500 to-pink-500">
      <Header />

      {/* New Events Banner */}
      {newEventsCount > 0 && (
        <div
          onClick={refreshFeed}
          className="fixed top-20 left-1/2 transform -translate-x-1/2 bg-gradient-to-r from-indigo-600 to-purple-600 text-white px-6 py-3 rounded-lg shadow-lg cursor-pointer z-50 animate-slideDown"
        >
          {newEventsCount} new {newEventsCount === 1 ? 'event' : 'events'} available - Click to refresh
        </div>
      )}

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="bg-white rounded-xl shadow-lg p-6">
          {/* ShareBox Trigger Button */}
          {!showShareBox && (
            <button
              onClick={() => setShowShareBox(true)}
              className="w-full mb-6 p-4 border-2 border-dashed border-indigo-300 rounded-xl hover:border-indigo-500 hover:bg-indigo-50 transition-all group cursor-pointer"
            >
              <div className="flex items-center gap-3 text-slate-600 group-hover:text-indigo-600">
                <div className="w-10 h-10 rounded-full bg-indigo-100 group-hover:bg-indigo-200 flex items-center justify-center text-2xl transition-colors">
                  +
                </div>
                <span className="text-base font-medium">
                  Report an event or breaking news
                </span>
              </div>
            </button>
          )}

          {/* ShareBox (expanded) */}
          {showShareBox && (
            <ShareBox
              onSubmit={handleSubmitEvent}
              onCancel={() => setShowShareBox(false)}
            />
          )}

          {/* Coherence Filter */}
          <div className="mb-6 pb-6 border-b border-slate-200">
            <label className="block text-sm font-medium text-slate-700 mb-2">
              Minimum Coherence: {(minCoherence * 100).toFixed(0)}%
            </label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.01"
              value={minCoherence}
              onChange={handleCoherenceChange}
              className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-indigo-600"
            />
            <div className="flex justify-between text-xs text-slate-500 mt-1">
              <span>0%</span>
              <span>50%</span>
              <span>100%</span>
            </div>
          </div>

          {error && (
            <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg mb-4">
              Error: {error}
            </div>
          )}

          {loading ? (
            <div className="grid gap-6">
              {Array(pageSize).fill(0).map((_, i) => (
                <StoryCardSkeleton key={i} />
              ))}
            </div>
          ) : mergedFeed.length === 0 ? (
            <div className="text-center py-16">
              <div className="text-lg text-slate-600">No events found</div>
            </div>
          ) : (
            <>
              <div className="grid gap-6">
                {mergedFeed.map((item) => {
                  if (item.type === 'event') {
                    const eventId = item.data.id || item.data.event_id || item.data.story_id
                    return (
                      <NewsCard
                        key={eventId}
                        story={item.data}
                        onClick={() => eventId && handleEventClick(eventId)}
                      />
                    )
                  } else {
                    return (
                      <PendingSubmission
                        key={item.data.id}
                        submission={item.data}
                      />
                    )
                  }
                })}
              </div>

              {loadingMore && (
                <div className="mt-6 grid gap-6">
                  {Array(3).fill(0).map((_, i) => (
                    <StoryCardSkeleton key={i} />
                  ))}
                </div>
              )}

              {!hasMore && events.length > 0 && (
                <div className="text-center py-8 text-slate-400 text-sm italic">
                  No more events
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  )
}

export default HomePage
