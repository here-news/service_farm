import React, { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import Header from './components/layout/Header'
import PendingSubmission, { EventSubmission } from './components/home/PendingSubmission'

function ArchivePage() {
  const navigate = useNavigate()
  const [submissions, setSubmissions] = useState<EventSubmission[]>([])
  const [loading, setLoading] = useState(true)
  const [filter, setFilter] = useState<'all' | 'pending' | 'extracting' | 'completed' | 'failed'>('all')
  const [searchQuery, setSearchQuery] = useState('')

  useEffect(() => {
    loadSubmissions()
    // Poll for updates every 2 seconds
    const interval = setInterval(loadSubmissions, 2000)
    return () => clearInterval(interval)
  }, [])

  const loadSubmissions = async () => {
    try {
      const response = await fetch('/api/events/mine', {
        credentials: 'include'
      })
      if (response.ok) {
        const data = await response.json()
        setSubmissions(data)
      }
      setLoading(false)
    } catch (err) {
      console.error('Failed to load submissions:', err)
      setLoading(false)
    }
  }

  // Filter submissions
  const filteredSubmissions = submissions.filter(sub => {
    // Filter by status
    if (filter !== 'all' && sub.status !== filter) {
      return false
    }

    // Filter by search query
    if (searchQuery) {
      const query = searchQuery.toLowerCase()
      return (
        sub.urls?.toLowerCase().includes(query) ||
        sub.content?.toLowerCase().includes(query)
      )
    }

    return true
  })

  // Get status counts
  const statusCounts = {
    all: submissions.length,
    pending: submissions.filter(s => s.status === 'pending').length,
    extracting: submissions.filter(s => s.status === 'extracting').length,
    completed: submissions.filter(s => s.status === 'completed').length,
    failed: submissions.filter(s => s.status === 'failed').length
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-50 to-white">
      <Header />

      <div className="max-w-4xl mx-auto px-4 py-8">
        {/* Page Header */}
        <div className="mb-8">
          <button
            onClick={() => navigate('/')}
            className="text-indigo-600 hover:text-indigo-800 mb-4 flex items-center gap-2"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
            </svg>
            Back to Feed
          </button>

          <h1 className="text-3xl font-bold text-slate-900 mb-2">My Archive</h1>
          <p className="text-slate-600">View and manage your submitted URLs</p>
        </div>

        {/* Search and Filters */}
        <div className="bg-white rounded-lg shadow-sm p-4 mb-6">
          {/* Search */}
          <div className="mb-4">
            <input
              type="text"
              placeholder="Search by URL or content..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full px-4 py-2 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500"
            />
          </div>

          {/* Status Filter Tabs */}
          <div className="flex gap-2 flex-wrap">
            {(['all', 'pending', 'extracting', 'completed', 'failed'] as const).map((status) => (
              <button
                key={status}
                onClick={() => setFilter(status)}
                className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                  filter === status
                    ? 'bg-indigo-600 text-white'
                    : 'bg-slate-100 text-slate-700 hover:bg-slate-200'
                }`}
              >
                {status.charAt(0).toUpperCase() + status.slice(1)}
                <span className="ml-2 text-sm opacity-75">({statusCounts[status]})</span>
              </button>
            ))}
          </div>
        </div>

        {/* Submissions List */}
        {loading ? (
          <div className="text-center py-16">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600 mx-auto"></div>
            <p className="mt-4 text-slate-600">Loading your archive...</p>
          </div>
        ) : filteredSubmissions.length === 0 ? (
          <div className="text-center py-16 bg-white rounded-lg shadow-sm">
            <svg className="w-16 h-16 mx-auto text-slate-300 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
            <h3 className="text-lg font-semibold text-slate-900 mb-2">No submissions found</h3>
            <p className="text-slate-600 mb-4">
              {searchQuery || filter !== 'all'
                ? 'Try adjusting your filters or search query'
                : 'Start by submitting your first URL from the home page'}
            </p>
            {(searchQuery || filter !== 'all') && (
              <button
                onClick={() => {
                  setSearchQuery('')
                  setFilter('all')
                }}
                className="text-indigo-600 hover:text-indigo-800 font-medium"
              >
                Clear filters
              </button>
            )}
          </div>
        ) : (
          <div className="space-y-4">
            {filteredSubmissions.map((submission) => (
              <PendingSubmission key={submission.id} submission={submission} />
            ))}
          </div>
        )}

        {/* Stats Footer */}
        {!loading && submissions.length > 0 && (
          <div className="mt-8 text-center text-sm text-slate-500">
            Showing {filteredSubmissions.length} of {submissions.length} submission{submissions.length !== 1 ? 's' : ''}
          </div>
        )}
      </div>
    </div>
  )
}

export default ArchivePage
