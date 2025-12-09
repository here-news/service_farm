import React, { useState, useRef, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'

interface SearchResult {
  id: string
  title: string
  description?: string
  category?: string
  artifact_count?: number
  claim_count?: number
  coherence?: number
  similarity?: number
}

function SearchBar() {
  const [query, setQuery] = useState('')
  const [searchResults, setSearchResults] = useState<SearchResult[]>([])
  const [isSearching, setIsSearching] = useState(false)
  const [showResults, setShowResults] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const navigate = useNavigate()

  // Global keyboard shortcuts
  useEffect(() => {
    const handleGlobalKeydown = (e: globalThis.KeyboardEvent) => {
      // Focus search on "/" key (if not already typing)
      if (e.key === '/' && document.activeElement?.tagName !== 'INPUT' && document.activeElement?.tagName !== 'TEXTAREA') {
        e.preventDefault()
        inputRef.current?.focus()
      }

      // Close on Escape
      if (e.key === 'Escape' && document.activeElement === inputRef.current) {
        inputRef.current?.blur()
        setShowResults(false)
      }
    }

    window.addEventListener('keydown', handleGlobalKeydown)
    return () => window.removeEventListener('keydown', handleGlobalKeydown)
  }, [])

  // Click outside to close
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setShowResults(false)
      }
    }

    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  // Debounced search (300ms)
  useEffect(() => {
    if (query.trim().length < 2) {
      setSearchResults([])
      setShowResults(false)
      return
    }

    setIsSearching(true)
    const timeoutId = setTimeout(async () => {
      try {
        const response = await fetch('/api/stories/search', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query: query.trim(), limit: 5 })
        })

        if (response.ok) {
          const data = await response.json()
          setSearchResults(data.matches || [])
          setShowResults(true)
        }
      } catch (err) {
        console.error('Search error:', err)
        setSearchResults([])
      } finally {
        setIsSearching(false)
      }
    }, 300)

    return () => clearTimeout(timeoutId)
  }, [query])

  const handleResultClick = (storyId: string) => {
    navigate(`/story/${storyId}`)
    setShowResults(false)
    setQuery('')
  }

  return (
    <div className="flex-1 max-w-md" ref={containerRef}>
      <div className="relative">
        <input
          ref={inputRef}
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onFocus={() => query.trim().length >= 2 && setShowResults(true)}
          placeholder="Search stories..."
          className="w-full px-4 py-2 pl-10 pr-12 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent text-sm"
        />

        {/* Search icon */}
        <svg
          className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-slate-400"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
          />
        </svg>

        {/* Loading indicator or "/" hint */}
        <div className="absolute right-3 top-1/2 transform -translate-y-1/2">
          {isSearching ? (
            <div className="animate-spin w-4 h-4 border-2 border-indigo-500 border-t-transparent rounded-full" />
          ) : (
            <kbd className="hidden sm:inline-block px-1.5 py-0.5 text-xs font-semibold text-slate-500 bg-slate-100 border border-slate-200 rounded">
              /
            </kbd>
          )}
        </div>

        {/* Search Results Dropdown */}
        {showResults && query.trim().length >= 2 && (
          <div className="absolute top-full mt-2 w-full bg-white border border-slate-200 rounded-lg shadow-lg max-h-96 overflow-y-auto z-50">
            {searchResults.length > 0 ? (
              <div className="py-2">
                {searchResults.map((result) => (
                  <button
                    key={result.id}
                    onClick={() => handleResultClick(result.id)}
                    className="w-full px-4 py-3 text-left hover:bg-slate-50 transition-colors border-b border-slate-100 last:border-b-0"
                  >
                    <div className="flex items-start gap-3">
                      <div className="flex-1 min-w-0">
                        {/* Title */}
                        <div className="font-medium text-slate-900 text-sm line-clamp-1">
                          {result.title}
                        </div>

                        {/* Description */}
                        {result.description && (
                          <div className="text-xs text-slate-600 mt-1 line-clamp-2">
                            {result.description}
                          </div>
                        )}

                        {/* Metadata */}
                        <div className="flex items-center gap-2 mt-1.5">
                          {result.category && (
                            <span className="text-xs text-slate-500 bg-slate-100 px-2 py-0.5 rounded">
                              {result.category}
                            </span>
                          )}
                          {result.artifact_count !== undefined && (
                            <span className="text-xs text-slate-500">
                              {result.artifact_count} sources
                            </span>
                          )}
                          {result.coherence !== undefined && (
                            <span className="text-xs text-indigo-600 font-medium">
                              {(result.coherence * 100).toFixed(0)}% coherence
                            </span>
                          )}
                        </div>
                      </div>

                      {/* Arrow icon */}
                      <svg className="w-4 h-4 text-slate-400 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                      </svg>
                    </div>
                  </button>
                ))}
              </div>
            ) : (
              <div className="px-4 py-6 text-center text-sm text-slate-500">
                No stories found for "{query}"
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

export default SearchBar
