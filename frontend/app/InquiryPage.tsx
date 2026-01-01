import React, { useState, useEffect } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { InquiryCarousel, InquirySummary } from './components/inquiry'
import InquiryCard from './components/inquiry/InquiryCard'
import { SIMULATED_INQUIRIES } from './data/simulatedInquiries'

function InquiryPage() {
  const navigate = useNavigate()
  const [searchParams] = useSearchParams()
  const searchQuery = searchParams.get('q') || ''
  const [inquiries, setInquiries] = useState<InquirySummary[]>([])
  const [loading, setLoading] = useState(true)
  const [sortBy, setSortBy] = useState<'stake' | 'entropy' | 'contributions' | 'created'>('stake')

  useEffect(() => {
    loadInquiries()
  }, [])

  const loadInquiries = async () => {
    try {
      const res = await fetch('/api/inquiry')
      if (res.ok) {
        const data = await res.json()
        // Merge real and simulated
        setInquiries([...data, ...SIMULATED_INQUIRIES])
      } else {
        setInquiries(SIMULATED_INQUIRIES)
      }
    } catch (e) {
      setInquiries(SIMULATED_INQUIRIES)
    } finally {
      setLoading(false)
    }
  }

  // Categorized inquiries
  const resolvedInquiries = inquiries.filter(i => i.status === 'resolved')
  const topBountiedInquiries = inquiries
    .filter(i => i.status === 'open' && i.stake > 100)
    .sort((a, b) => b.stake - a.stake)
  const contestedInquiries = inquiries
    .filter(i => i.status === 'open' && i.entropy_bits > 3)
    .sort((a, b) => b.entropy_bits - a.entropy_bits)
  const openInquiries = inquiries.filter(i => i.status === 'open')

  // Filtered by search
  const filteredInquiries = searchQuery
    ? inquiries.filter(i =>
        i.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
        (i.scope_entities || []).some(e => e.toLowerCase().includes(searchQuery.toLowerCase()))
      )
    : []

  // Sort open inquiries
  const sortedOpenInquiries = [...openInquiries].sort((a, b) => {
    switch (sortBy) {
      case 'stake': return b.stake - a.stake
      case 'entropy': return b.entropy_bits - a.entropy_bits
      case 'contributions': return b.contributions - a.contributions
      default: return 0
    }
  })

  const handleSelect = (id: string) => {
    navigate(`/inquiry/${id}`)
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[50vh]">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600 mx-auto mb-4" />
          <p className="text-slate-500">Loading inquiries...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="max-w-6xl mx-auto px-4 py-8">
      {/* Header */}
      <div className="text-center mb-8">
        <h1 className="text-3xl font-bold text-slate-800 mb-2">What's True?</h1>
        <p className="text-slate-500">Get answers to contested questions. Earn rewards for truth.</p>
        {/* Quick stats */}
        <div className="flex justify-center gap-6 mt-4 text-sm">
          <div className="flex items-center gap-1.5 text-amber-600">
            <span>💰</span>
            <span className="font-semibold">${topBountiedInquiries.reduce((sum, i) => sum + i.stake, 0).toLocaleString()}</span>
            <span className="text-slate-400">in bounties</span>
          </div>
          <div className="flex items-center gap-1.5 text-green-600">
            <span>✅</span>
            <span className="font-semibold">{resolvedInquiries.length}</span>
            <span className="text-slate-400">resolved</span>
          </div>
          <div className="flex items-center gap-1.5 text-indigo-600">
            <span>🔍</span>
            <span className="font-semibold">{openInquiries.length}</span>
            <span className="text-slate-400">open</span>
          </div>
        </div>
      </div>


      {/* Search Results */}
      {searchQuery && filteredInquiries.length > 0 && (
        <section className="mb-10">
          <div className="flex items-center gap-2 mb-4">
            <span className="text-xl">🔎</span>
            <h2 className="text-lg font-semibold text-slate-700">Search Results</h2>
            <span className="text-xs bg-slate-100 text-slate-600 px-2 py-0.5 rounded-full">
              {filteredInquiries.length} found
            </span>
          </div>
          <div className="space-y-3">
            {filteredInquiries.slice(0, 5).map((inq) => (
              <InquiryCard
                key={inq.id}
                inquiry={inq}
                onClick={() => handleSelect(inq.id)}
              />
            ))}
          </div>
        </section>
      )}

      {/* No Results */}
      {searchQuery && filteredInquiries.length === 0 && (
        <div className="text-center py-8 mb-8 bg-white rounded-xl border border-slate-200">
          <div className="text-4xl mb-2">🤷</div>
          <p className="text-slate-600 mb-3">No matching questions found for "{searchQuery}"</p>
          <div className="flex justify-center gap-4">
            <button
              onClick={() => navigate('/inquiry')}
              className="text-slate-500 hover:text-slate-700"
            >
              Clear search
            </button>
            <button
              onClick={() => navigate('/inquiry/new')}
              className="text-indigo-500 hover:text-indigo-700 font-medium"
            >
              Create this question →
            </button>
          </div>
        </div>
      )}

      {/* Carousels (hidden during search) */}
      {!searchQuery && (
        <>
          {/* Recently Resolved */}
          <InquiryCarousel
            title="Recently Resolved"
            icon="✅"
            badge={{ text: 'Truth found', color: 'bg-green-100 text-green-700' }}
            inquiries={resolvedInquiries}
            variant="resolved"
            onSelect={handleSelect}
          />

          {/* Top Bounties */}
          <InquiryCarousel
            title="Top Bounties"
            icon="💰"
            badge={{ text: 'Earn rewards', color: 'bg-amber-100 text-amber-700' }}
            inquiries={topBountiedInquiries}
            variant="bounty"
            onSelect={handleSelect}
          />

          {/* Highly Contested */}
          <InquiryCarousel
            title="Highly Contested"
            icon="⚔️"
            badge={{ text: 'Conflicting evidence', color: 'bg-red-100 text-red-600' }}
            inquiries={contestedInquiries}
            variant="contested"
            onSelect={handleSelect}
          />

          {/* All Open Questions */}
          <section className="mb-10">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2">
                <span className="text-xl">🔍</span>
                <h2 className="text-lg font-semibold text-slate-700">All Open Questions</h2>
              </div>
              <select
                value={sortBy}
                onChange={(e) => setSortBy(e.target.value as any)}
                className="text-sm border border-slate-200 rounded-lg px-3 py-1.5 bg-white"
              >
                <option value="stake">By Bounty</option>
                <option value="entropy">By Disagreement</option>
                <option value="contributions">By Activity</option>
              </select>
            </div>
            <div className="space-y-3">
              {sortedOpenInquiries.map((inq) => (
                <InquiryCard
                  key={inq.id}
                  inquiry={inq}
                  onClick={() => handleSelect(inq.id)}
                />
              ))}
            </div>
          </section>
        </>
      )}

      {/* FAB for new inquiry */}
      <button
        onClick={() => navigate('/inquiry/new')}
        className="fixed bottom-6 right-6 w-14 h-14 bg-gradient-to-r from-indigo-600 to-purple-600 text-white rounded-full shadow-lg hover:shadow-xl flex items-center justify-center transition-all hover:scale-105"
        title="Ask a new question"
      >
        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
        </svg>
      </button>
    </div>
  )
}

export default InquiryPage
