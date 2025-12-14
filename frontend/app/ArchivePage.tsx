import React, { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'

interface ArchivedPage {
  id: string
  url: string
  title: string
  domain: string
  thumbnail_url?: string
  status: 'stub' | 'preview' | 'extracted' | 'knowledge_complete' | 'event_complete' | 'failed'
  created_at: string
  updated_at?: string
  event_id?: string
  event_name?: string
  submitter_id?: string
  submitter_name?: string
  is_mine?: boolean
  // Archive lifecycle - community-backed persistence
  days_remaining: number
  total_credits: number
  backer_count: number
}

// Format days as "X yr Y days" or "X days"
function formatLifespan(totalDays: number): string {
  if (totalDays >= 365) {
    const years = Math.floor(totalDays / 365)
    const days = totalDays % 365
    if (days === 0) return `${years} yr`
    return `${years} yr ${days} d`
  }
  return `${totalDays} d`
}

// Get color based on lifespan health
function getLifespanColor(days: number): string {
  if (days <= 7) return 'text-red-600'
  if (days <= 30) return 'text-amber-600'
  if (days <= 90) return 'text-yellow-600'
  return 'text-green-600'
}

// Get bg color for lifespan indicator
function getLifespanBgColor(days: number): string {
  if (days <= 7) return 'bg-red-50 border-red-200'
  if (days <= 30) return 'bg-amber-50 border-amber-200'
  if (days <= 90) return 'bg-yellow-50 border-yellow-200'
  return 'bg-green-50 border-green-200'
}

function ArchivePage() {
  const navigate = useNavigate()
  const [pages, setPages] = useState<ArchivedPage[]>([])
  const [loading, setLoading] = useState(true)
  const [viewMode, setViewMode] = useState<'all' | 'mine'>('all')
  const [statusFilter, setStatusFilter] = useState<string>('all')
  const [searchQuery, setSearchQuery] = useState('')
  const [sortBy, setSortBy] = useState<'recent' | 'expiring' | 'funded'>('recent')

  useEffect(() => {
    loadPages()
    const interval = setInterval(loadPages, 5000)
    return () => clearInterval(interval)
  }, [viewMode])

  const loadPages = async () => {
    try {
      // Load all pages or user's pages based on view mode
      const endpoint = viewMode === 'mine' ? '/api/pages/mine' : '/api/pages'
      const response = await fetch(endpoint, { credentials: 'include' })

      if (response.ok) {
        const data = await response.json()
        // Transform API response to include archive-specific fields
        // Demo: Generate realistic lifecycle data based on page age
        const pagesWithArchiveInfo = (data.pages || data || []).map((page: any, index: number) => {
          // Simulate varying lifespans - some well-funded, some expiring
          // Use page index to create deterministic but varied values
          const hash = page.id.charCodeAt(3) + page.id.charCodeAt(4)
          const lifespanVariants = [
            { days: 1461 + (hash % 100), credits: 4200 + hash * 10, backers: 45 + (hash % 20) },  // ~4 yr
            { days: 730 + (hash % 50), credits: 2100 + hash * 5, backers: 28 + (hash % 15) },    // ~2 yr
            { days: 365 + (hash % 30), credits: 1100 + hash * 3, backers: 15 + (hash % 10) },    // ~1 yr
            { days: 180 + (hash % 20), credits: 550 + hash * 2, backers: 8 + (hash % 8) },       // ~6 mo
            { days: 90 + (hash % 15), credits: 280 + hash, backers: 5 + (hash % 5) },            // ~3 mo
            { days: 30 + (hash % 10), credits: 95 + (hash % 50), backers: 3 + (hash % 3) },      // ~1 mo
            { days: 7 + (hash % 7), credits: 25 + (hash % 20), backers: 1 + (hash % 2) },        // expiring soon
            { days: 3 + (hash % 3), credits: 8 + (hash % 10), backers: 1 },                      // critical
          ]
          const variant = lifespanVariants[index % lifespanVariants.length]

          return {
            ...page,
            days_remaining: variant.days,
            total_credits: variant.credits,
            backer_count: variant.backers,
            is_mine: page.submitter_id === 'current_user' // Will use real auth
          }
        })
        setPages(pagesWithArchiveInfo)
      }
      setLoading(false)
    } catch (err) {
      console.error('Failed to load pages:', err)
      setLoading(false)
    }
  }

  // Filter and sort pages
  const filteredPages = pages
    .filter(page => {
      if (statusFilter !== 'all' && page.status !== statusFilter) return false
      if (searchQuery) {
        const query = searchQuery.toLowerCase()
        return (
          page.url?.toLowerCase().includes(query) ||
          page.title?.toLowerCase().includes(query) ||
          page.domain?.toLowerCase().includes(query)
        )
      }
      return true
    })
    .sort((a, b) => {
      switch (sortBy) {
        case 'expiring':
          return a.days_remaining - b.days_remaining
        case 'funded':
          return b.total_credits - a.total_credits
        case 'recent':
        default:
          return new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
      }
    })

  // Status counts
  const statusCounts = {
    all: pages.length,
    stub: pages.filter(p => p.status === 'stub').length,
    extracted: pages.filter(p => p.status === 'extracted').length,
    knowledge_complete: pages.filter(p => p.status === 'knowledge_complete').length,
    event_complete: pages.filter(p => p.status === 'event_complete').length,
    failed: pages.filter(p => p.status === 'failed').length,
  }

  const getStatusBadge = (status: string) => {
    const styles: Record<string, { bg: string; text: string; label: string }> = {
      stub: { bg: 'bg-slate-100', text: 'text-slate-600', label: 'Queued' },
      preview: { bg: 'bg-blue-100', text: 'text-blue-700', label: 'Preview' },
      extracted: { bg: 'bg-amber-100', text: 'text-amber-700', label: 'Extracted' },
      knowledge_complete: { bg: 'bg-purple-100', text: 'text-purple-700', label: 'Analyzed' },
      event_complete: { bg: 'bg-green-100', text: 'text-green-700', label: 'Complete' },
      failed: { bg: 'bg-red-100', text: 'text-red-700', label: 'Failed' },
    }
    const style = styles[status] || styles.stub
    return (
      <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${style.bg} ${style.text}`}>
        {style.label}
      </span>
    )
  }

  const handlePageClick = (page: ArchivedPage) => {
    if (page.event_id) {
      navigate(`/event/${page.event_id}`)
    } else {
      navigate(`/page/${page.id}`)
    }
  }

  const handleDonate = (e: React.MouseEvent, pageId: string) => {
    e.stopPropagation()
    const page = pages.find(p => p.id === pageId)
    const title = page?.title || 'this page'
    alert(
      `Extend Archive Life (coming soon)\n\n` +
      `Page: ${title.substring(0, 50)}${title.length > 50 ? '...' : ''}\n` +
      `Current: ${formatLifespan(page?.days_remaining || 0)} remaining\n` +
      `Backers: ${page?.backer_count || 0}\n\n` +
      `1 credit = 1 extra day of preservation\n` +
      `Your contribution helps preserve this content for everyone.`
    )
  }

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Page Header */}
      <div className="mb-8">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h1 className="text-3xl font-bold text-slate-900">Archive</h1>
            <p className="text-slate-600 mt-1">
              Browse and preserve web pages in our decentralized archive network
            </p>
          </div>
          <button
            onClick={() => navigate('/?share=true')}
            className="px-4 py-2 bg-gradient-to-r from-indigo-600 to-purple-600 text-white rounded-lg font-medium hover:opacity-90 transition flex items-center gap-2"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
            </svg>
            Add Page
          </button>
        </div>

        {/* Archive Info Banner */}
        <div className="bg-gradient-to-r from-indigo-50 to-purple-50 border border-indigo-100 rounded-xl p-4 mb-6">
          <div className="flex items-start gap-4">
            <div className="w-10 h-10 rounded-full bg-indigo-100 flex items-center justify-center flex-shrink-0">
              <svg className="w-5 h-5 text-indigo-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 8h14M5 8a2 2 0 110-4h14a2 2 0 110 4M5 8v10a2 2 0 002 2h10a2 2 0 002-2V8m-9 4h4" />
              </svg>
            </div>
            <div className="flex-1">
              <h3 className="font-semibold text-slate-800 mb-1">Community-Backed Preservation</h3>
              <p className="text-sm text-slate-600">
                Pages are preserved based on community funding. Each page has a remaining lifespan that you can extend.
                <span className="font-medium text-indigo-600 ml-1">1 credit = 1 extra day of preservation.</span>
              </p>
            </div>
            <div className="flex gap-6 flex-shrink-0">
              <div className="text-right">
                <div className="text-2xl font-bold text-indigo-600">{pages.length}</div>
                <div className="text-xs text-slate-500">Archived</div>
              </div>
              <div className="text-right">
                <div className="text-2xl font-bold text-amber-600">
                  {pages.reduce((sum, p) => sum + p.total_credits, 0).toLocaleString()}
                </div>
                <div className="text-xs text-slate-500">Total Credits</div>
              </div>
              <div className="text-right">
                <div className={`text-2xl font-bold ${pages.filter(p => p.days_remaining <= 30).length > 0 ? 'text-red-600' : 'text-green-600'}`}>
                  {pages.filter(p => p.days_remaining <= 30).length}
                </div>
                <div className="text-xs text-slate-500">Need Help</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Filters Bar */}
      <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-4 mb-6">
        <div className="flex flex-col lg:flex-row gap-4">
          {/* View Toggle */}
          <div className="flex rounded-lg border border-slate-200 overflow-hidden">
            <button
              onClick={() => setViewMode('all')}
              className={`px-4 py-2 text-sm font-medium transition-colors ${
                viewMode === 'all'
                  ? 'bg-indigo-600 text-white'
                  : 'bg-white text-slate-700 hover:bg-slate-50'
              }`}
            >
              All Pages
            </button>
            <button
              onClick={() => setViewMode('mine')}
              className={`px-4 py-2 text-sm font-medium transition-colors border-l border-slate-200 ${
                viewMode === 'mine'
                  ? 'bg-indigo-600 text-white'
                  : 'bg-white text-slate-700 hover:bg-slate-50'
              }`}
            >
              My Submissions
            </button>
          </div>

          {/* Search */}
          <div className="flex-1">
            <div className="relative">
              <svg className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
              <input
                type="text"
                placeholder="Search by URL, title, or domain..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full pl-10 pr-4 py-2 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 text-sm"
              />
            </div>
          </div>

          {/* Sort */}
          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value as any)}
            className="px-3 py-2 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 text-sm bg-white"
          >
            <option value="recent">Most Recent</option>
            <option value="expiring">Expiring Soon</option>
            <option value="funded">Most Funded</option>
          </select>
        </div>

        {/* Status Filter Pills */}
        <div className="flex gap-2 mt-4 flex-wrap">
          {Object.entries(statusCounts).map(([status, count]) => (
            <button
              key={status}
              onClick={() => setStatusFilter(status)}
              className={`px-3 py-1.5 rounded-full text-sm font-medium transition-colors ${
                statusFilter === status
                  ? 'bg-indigo-600 text-white'
                  : 'bg-slate-100 text-slate-700 hover:bg-slate-200'
              }`}
            >
              {status === 'all' ? 'All' : status.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
              <span className="ml-1.5 opacity-75">({count})</span>
            </button>
          ))}
        </div>
      </div>

      {/* Pages Grid */}
      {loading ? (
        <div className="text-center py-16">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600 mx-auto"></div>
          <p className="mt-4 text-slate-600">Loading archive...</p>
        </div>
      ) : filteredPages.length === 0 ? (
        <div className="text-center py-16 bg-white rounded-xl shadow-sm border border-slate-200">
          <svg className="w-16 h-16 mx-auto text-slate-300 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 8h14M5 8a2 2 0 110-4h14a2 2 0 110 4M5 8v10a2 2 0 002 2h10a2 2 0 002-2V8m-9 4h4" />
          </svg>
          <h3 className="text-lg font-semibold text-slate-900 mb-2">No pages found</h3>
          <p className="text-slate-600 mb-4">
            {searchQuery || statusFilter !== 'all'
              ? 'Try adjusting your filters'
              : 'Be the first to add a page to the archive'}
          </p>
          <button
            onClick={() => navigate('/?share=true')}
            className="px-4 py-2 bg-indigo-600 text-white rounded-lg font-medium hover:bg-indigo-700 transition"
          >
            Add First Page
          </button>
        </div>
      ) : (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {filteredPages.map((page) => (
            <div
              key={page.id}
              onClick={() => handlePageClick(page)}
              className={`bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden hover:shadow-lg hover:border-indigo-200 transition-all cursor-pointer group ${
                page.is_mine ? 'ring-2 ring-indigo-100' : ''
              }`}
            >
              {/* Thumbnail */}
              <div className="relative h-36 bg-slate-100">
                {page.thumbnail_url ? (
                  <img
                    src={page.thumbnail_url}
                    alt={page.title}
                    className="w-full h-full object-cover"
                    onError={(e) => {
                      (e.target as HTMLImageElement).style.display = 'none'
                    }}
                  />
                ) : (
                  <div className="w-full h-full flex items-center justify-center bg-gradient-to-br from-slate-100 to-slate-200">
                    <span className="text-4xl">📄</span>
                  </div>
                )}

                {/* Status badge overlay */}
                <div className="absolute top-2 left-2">
                  {getStatusBadge(page.status)}
                </div>

                {/* Mine indicator */}
                {page.is_mine && (
                  <div className="absolute top-2 right-2 px-2 py-0.5 bg-indigo-600 text-white text-xs rounded-full">
                    Mine
                  </div>
                )}

                {/* Lifespan indicator - prominent corner badge */}
                <div className={`absolute bottom-2 right-2 rounded-lg px-2.5 py-1.5 shadow-sm border ${getLifespanBgColor(page.days_remaining)}`}>
                  <div className={`text-sm font-bold ${getLifespanColor(page.days_remaining)}`}>
                    {formatLifespan(page.days_remaining)}
                  </div>
                </div>
              </div>

              {/* Content */}
              <div className="p-4">
                <h3 className="font-semibold text-slate-900 line-clamp-2 mb-1 group-hover:text-indigo-600 transition-colors">
                  {page.title || 'Untitled Page'}
                </h3>
                <div className="text-xs text-slate-500 truncate mb-3">
                  {page.domain || new URL(page.url).hostname}
                </div>

                {/* Event link if available */}
                {page.event_name && (
                  <div className="text-xs text-indigo-600 truncate mb-3 flex items-center gap-1">
                    <span>📌</span>
                    <span>{page.event_name}</span>
                  </div>
                )}

                {/* Lifecycle footer */}
                <div className="flex items-center justify-between pt-3 border-t border-slate-100">
                  <div className="flex items-center gap-3">
                    {/* Backer count */}
                    <div className="flex items-center gap-1 text-xs text-slate-500">
                      <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0z" />
                      </svg>
                      <span>{page.backer_count}</span>
                    </div>
                    {/* Total credits */}
                    <div className="flex items-center gap-1 text-xs text-amber-600 font-medium">
                      <span>{page.total_credits.toLocaleString()}</span>
                      <span className="text-amber-500">cr</span>
                    </div>
                  </div>
                  {/* + days button */}
                  <button
                    onClick={(e) => handleDonate(e, page.id)}
                    className="px-2.5 py-1 rounded-lg bg-gradient-to-r from-amber-50 to-orange-50 hover:from-amber-100 hover:to-orange-100 border border-amber-200 text-amber-700 text-xs font-semibold transition-colors flex items-center gap-1"
                    title="1 credit = 1 extra day of preservation"
                  >
                    <span>+</span>
                    <span>days</span>
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Stats Footer */}
      {!loading && filteredPages.length > 0 && (
        <div className="mt-8 text-center text-sm text-slate-500">
          Showing {filteredPages.length} of {pages.length} archived pages
        </div>
      )}
    </div>
  )
}

export default ArchivePage
