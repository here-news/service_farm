import React, { useState, useEffect } from 'react'
import { useNavigate, useSearchParams, Link } from 'react-router-dom'
import { InquiryCarousel, InquirySummary } from './components/inquiry'
import InquiryCard from './components/inquiry/InquiryCard'
import { SIMULATED_INQUIRIES } from './data/simulatedInquiries'

// =============================================================================
// Trending Entities Data
// =============================================================================

interface TrendingEntity {
  id: string
  slug: string
  name: string
  type: 'PERSON' | 'ORG' | 'LOCATION' | 'EVENT'
  image?: string
  open_inquiries: number
  total_bounty: number
  hot_inquiry?: string  // Most active question
  trend: 'up' | 'stable' | 'new'
}

const TRENDING_ENTITIES: TrendingEntity[] = [
  {
    id: 'elon-musk',
    slug: 'elon-musk',
    name: 'Elon Musk',
    type: 'PERSON',
    image: 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/34/Elon_Musk_Royal_Society_%28crop2%29.jpg/220px-Elon_Musk_Royal_Society_%28crop2%29.jpg',
    open_inquiries: 3,
    total_bounty: 250,
    hot_inquiry: 'How many children does Elon Musk have?',
    trend: 'up'
  },
  {
    id: 'japan',
    slug: 'japan',
    name: 'Japan',
    type: 'LOCATION',
    open_inquiries: 3,
    total_bounty: 105,
    hot_inquiry: 'Is there a tsunami warning today?',
    trend: 'new'
  },
  {
    id: 'tesla',
    slug: 'tesla',
    name: 'Tesla',
    type: 'ORG',
    open_inquiries: 5,
    total_bounty: 450,
    hot_inquiry: 'How many cars did Tesla deliver in Q4 2024?',
    trend: 'up'
  },
  {
    id: 'openai',
    slug: 'openai',
    name: 'OpenAI',
    type: 'ORG',
    open_inquiries: 4,
    total_bounty: 380,
    hot_inquiry: 'Will GPT-5 be released before July 2025?',
    trend: 'stable'
  },
  {
    id: 'china',
    slug: 'china',
    name: 'China',
    type: 'LOCATION',
    open_inquiries: 6,
    total_bounty: 520,
    hot_inquiry: 'What is the actual unemployment rate?',
    trend: 'up'
  },
  {
    id: 'gaza',
    slug: 'gaza',
    name: 'Gaza',
    type: 'LOCATION',
    open_inquiries: 8,
    total_bounty: 1200,
    hot_inquiry: 'How many people were killed in the hospital explosion?',
    trend: 'up'
  }
]

// Knowledge gap categories
interface KnowledgeGap {
  category: string
  icon: string
  description: string
  example_questions: string[]
  total_bounty: number
  urgency: 'high' | 'medium' | 'low'
}

const KNOWLEDGE_GAPS: KnowledgeGap[] = [
  {
    category: 'Casualty Counts',
    icon: '📊',
    description: 'Verify death tolls and injury figures from conflicts and disasters',
    example_questions: ['Gaza hospital explosion', 'Ukraine war casualties', 'Hong Kong fire deaths'],
    total_bounty: 3200,
    urgency: 'high'
  },
  {
    category: 'Corporate Claims',
    icon: '🏢',
    description: 'Fact-check company announcements, financials, and projections',
    example_questions: ['Tesla deliveries', 'OpenAI valuation', 'Layoff numbers'],
    total_bounty: 1800,
    urgency: 'medium'
  },
  {
    category: 'Government Data',
    icon: '🏛️',
    description: 'Verify official statistics and policy claims',
    example_questions: ['China unemployment', 'US inflation', 'Election results'],
    total_bounty: 2100,
    urgency: 'medium'
  },
  {
    category: 'Scientific Claims',
    icon: '🔬',
    description: 'Validate research findings and health claims',
    example_questions: ['COVID origins', 'Climate data', 'Drug efficacy'],
    total_bounty: 950,
    urgency: 'low'
  }
]

// =============================================================================
// Live Pulse Data - The Epistemic Loop in Action
// =============================================================================

type PulseEventType =
  | 'evidence_added'    // New claim/source surfaced
  | 'posterior_update'  // Belief changed
  | 'task_completed'    // Human work done
  | 'inquiry_resolved'  // Contract fulfilled
  | 'bounty_staked'     // Attention committed
  | 'conflict_detected' // CONFLICTS relation found
  | 'proto_inquiry'     // REEE emitted new question

interface PulseEvent {
  id: string
  type: PulseEventType
  timestamp: Date
  inquiry_title?: string
  inquiry_id?: string
  entity?: string
  actor?: string  // username or 'REEE'
  detail: string
  delta?: string  // e.g., "+$50", "72% → 85%", "+1 source"
}

const PULSE_EVENT_CONFIG: Record<PulseEventType, { icon: string; color: string; label: string }> = {
  evidence_added: { icon: '📄', color: 'text-blue-600 bg-blue-50', label: 'Evidence' },
  posterior_update: { icon: '📊', color: 'text-purple-600 bg-purple-50', label: 'Belief Update' },
  task_completed: { icon: '✓', color: 'text-green-600 bg-green-50', label: 'Task Done' },
  inquiry_resolved: { icon: '✅', color: 'text-emerald-600 bg-emerald-50', label: 'Resolved' },
  bounty_staked: { icon: '💰', color: 'text-amber-600 bg-amber-50', label: 'Bounty' },
  conflict_detected: { icon: '!', color: 'text-red-600 bg-red-50', label: 'Conflict' },
  proto_inquiry: { icon: '?', color: 'text-indigo-600 bg-indigo-50', label: 'Proto-Inquiry' }
}

// Simulated recent activity (in production, this comes from WebSocket/SSE)
const INITIAL_PULSE_EVENTS: PulseEvent[] = [
  {
    id: 'p1',
    type: 'evidence_added',
    timestamp: new Date(Date.now() - 45000),
    inquiry_title: 'Wang Fuk Court Fire death toll',
    inquiry_id: 'wang-fuk-deaths',
    actor: '@hkwatcher',
    detail: 'Added RTHK report confirming 46 bodies recovered',
    delta: '+1 source'
  },
  {
    id: 'p2',
    type: 'posterior_update',
    timestamp: new Date(Date.now() - 120000),
    inquiry_title: 'Gaza hospital explosion casualties',
    inquiry_id: 'sim_gaza_hospital',
    actor: 'REEE',
    detail: 'Updated posterior after Al Jazeera retraction',
    delta: '500 → 300-350'
  },
  {
    id: 'p3',
    type: 'task_completed',
    timestamp: new Date(Date.now() - 180000),
    inquiry_title: 'Elon Musk children count',
    inquiry_id: 'elon-musk-children',
    actor: '@factchecker',
    detail: 'Verified Shivon Zilis twins via court records',
    delta: 'Rigor A'
  },
  {
    id: 'p4',
    type: 'bounty_staked',
    timestamp: new Date(Date.now() - 240000),
    inquiry_title: 'Ukraine casualties Dec 2024',
    inquiry_id: 'sim_ukraine_deaths',
    actor: '@osint_collective',
    detail: 'Increased bounty for primary source verification',
    delta: '+$500'
  },
  {
    id: 'p5',
    type: 'conflict_detected',
    timestamp: new Date(Date.now() - 300000),
    inquiry_title: 'Tesla Q4 deliveries',
    inquiry_id: 'tesla-q4',
    actor: 'REEE',
    detail: 'Reuters vs Bloomberg figures conflict',
    delta: '! CONFLICTS'
  },
  {
    id: 'p6',
    type: 'proto_inquiry',
    timestamp: new Date(Date.now() - 360000),
    entity: 'Hong Kong',
    actor: 'REEE',
    detail: 'High uncertainty detected: "Were fire safety inspections conducted?"',
    delta: '4.2 bits'
  },
  {
    id: 'p7',
    type: 'inquiry_resolved',
    timestamp: new Date(Date.now() - 420000),
    inquiry_title: 'Did Elon Musk acquire Twitter in 2022?',
    inquiry_id: 'sim_twitter_acquisition',
    actor: 'REEE',
    detail: 'Resolved YES with 99% confidence',
    delta: '✓ Resolved'
  }
]

// =============================================================================
// Sub-components
// =============================================================================

function TrendingEntityCard({ entity, onClick }: { entity: TrendingEntity; onClick: () => void }) {
  const typeColors = {
    PERSON: 'bg-purple-100 text-purple-700',
    ORG: 'bg-blue-100 text-blue-700',
    LOCATION: 'bg-green-100 text-green-700',
    EVENT: 'bg-red-100 text-red-700'
  }

  const trendIcons = {
    up: '📈',
    stable: '➡️',
    new: '🆕'
  }

  return (
    <div
      onClick={onClick}
      className="bg-white rounded-xl border border-slate-200 p-4 cursor-pointer hover:shadow-md hover:-translate-y-0.5 transition-all min-w-[200px]"
    >
      <div className="flex items-start gap-3">
        {/* Avatar/Image */}
        <div className="w-12 h-12 rounded-lg bg-gradient-to-br from-slate-100 to-slate-200 flex items-center justify-center overflow-hidden flex-shrink-0">
          {entity.image ? (
            <img src={entity.image} alt={entity.name} className="w-full h-full object-cover" />
          ) : (
            <span className="text-xl">
              {entity.type === 'PERSON' ? '👤' : entity.type === 'ORG' ? '🏢' : entity.type === 'LOCATION' ? '📍' : '📅'}
            </span>
          )}
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-1.5 mb-1">
            <span className={`text-[10px] px-1.5 py-0.5 rounded ${typeColors[entity.type]}`}>
              {entity.type}
            </span>
            <span className="text-xs" title={entity.trend === 'up' ? 'Trending up' : entity.trend === 'new' ? 'New' : 'Stable'}>
              {trendIcons[entity.trend]}
            </span>
          </div>
          <h3 className="font-semibold text-slate-800 text-sm truncate">{entity.name}</h3>
        </div>
      </div>

      {/* Hot inquiry */}
      {entity.hot_inquiry && (
        <p className="text-xs text-slate-500 mt-2 line-clamp-2 italic">
          "{entity.hot_inquiry}"
        </p>
      )}

      {/* Stats */}
      <div className="flex items-center gap-3 mt-3 pt-2 border-t border-slate-100 text-xs">
        <span className="text-amber-600 font-medium">${entity.total_bounty}</span>
        <span className="text-slate-400">{entity.open_inquiries} questions</span>
      </div>
    </div>
  )
}

function KnowledgeGapCard({ gap, onClick }: { gap: KnowledgeGap; onClick: () => void }) {
  const urgencyColors = {
    high: 'border-l-red-500 bg-red-50',
    medium: 'border-l-amber-500 bg-amber-50',
    low: 'border-l-slate-400 bg-slate-50'
  }

  return (
    <div
      onClick={onClick}
      className={`rounded-lg border border-slate-200 border-l-4 ${urgencyColors[gap.urgency]} p-4 cursor-pointer hover:shadow-md transition-all`}
    >
      <div className="flex items-start gap-3">
        <span className="text-2xl">{gap.icon}</span>
        <div className="flex-1">
          <h3 className="font-semibold text-slate-800">{gap.category}</h3>
          <p className="text-xs text-slate-500 mt-1">{gap.description}</p>
          <div className="flex flex-wrap gap-1 mt-2">
            {gap.example_questions.slice(0, 2).map(q => (
              <span key={q} className="text-xs px-2 py-0.5 bg-white/70 rounded text-slate-600">
                {q}
              </span>
            ))}
          </div>
          <div className="text-xs text-amber-600 font-medium mt-2">
            ${gap.total_bounty.toLocaleString()} in bounties
          </div>
        </div>
      </div>
    </div>
  )
}

function LiveEventCard({
  title,
  type,
  location,
  date,
  bounty,
  inquiries,
  claims,
  status,
  onClick
}: {
  title: string
  type: string
  location: string
  date: string
  bounty: number
  inquiries: number
  claims: number
  status: 'live' | 'developing' | 'resolved'
  onClick: () => void
}) {
  return (
    <div
      onClick={onClick}
      className="bg-gradient-to-r from-red-50 to-orange-50 border border-red-200 rounded-xl p-4 cursor-pointer hover:shadow-md hover:-translate-y-0.5 transition-all"
    >
      <div className="flex items-center gap-2 mb-2">
        <span className="px-2 py-0.5 bg-red-100 text-red-700 text-xs rounded-full uppercase font-medium">{type}</span>
        <span className="text-xs text-slate-400">{date} · {location}</span>
        {status === 'developing' && (
          <span className="px-2 py-0.5 bg-amber-100 text-amber-700 text-xs rounded-full flex items-center gap-1 ml-auto">
            <span className="w-1.5 h-1.5 bg-amber-500 rounded-full animate-pulse"></span>
            Developing
          </span>
        )}
        {status === 'live' && (
          <span className="px-2 py-0.5 bg-red-100 text-red-700 text-xs rounded-full flex items-center gap-1 ml-auto">
            <span className="w-1.5 h-1.5 bg-red-500 rounded-full animate-pulse"></span>
            Live
          </span>
        )}
      </div>
      <h3 className="font-bold text-slate-800 mb-2">{title}</h3>
      <div className="flex items-center gap-4 text-sm">
        <span className="text-amber-600 font-semibold">${bounty.toLocaleString()}</span>
        <span className="text-slate-500">{inquiries} inquiries</span>
        <span className="text-slate-400">{claims} claims</span>
        <span className="ml-auto text-indigo-600 font-medium text-xs">Investigate →</span>
      </div>
    </div>
  )
}

function LivePulse({ events, onEventClick }: { events: PulseEvent[]; onEventClick?: (e: PulseEvent) => void }) {
  const [visibleEvents, setVisibleEvents] = useState(events.slice(0, 5))
  const [isLive, setIsLive] = useState(true)

  // Simulate new events arriving (in production: WebSocket/SSE)
  useEffect(() => {
    if (!isLive) return

    const interval = setInterval(() => {
      // Rotate events to simulate activity
      setVisibleEvents(prev => {
        const rotated = [...prev.slice(1), prev[0]]
        return rotated
      })
    }, 8000)

    return () => clearInterval(interval)
  }, [isLive])

  const formatTime = (date: Date) => {
    const seconds = Math.floor((Date.now() - date.getTime()) / 1000)
    if (seconds < 60) return `${seconds}s ago`
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`
    return `${Math.floor(seconds / 3600)}h ago`
  }

  return (
    <div className="bg-gradient-to-br from-slate-900 to-slate-800 rounded-xl p-4 text-white">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <div className={`w-2 h-2 rounded-full ${isLive ? 'bg-green-400 animate-pulse' : 'bg-slate-500'}`} />
          <h3 className="font-semibold text-sm">Live Pulse</h3>
          <span className="text-xs text-slate-400">The epistemic loop</span>
        </div>
        <button
          onClick={() => setIsLive(!isLive)}
          className={`text-xs px-2 py-0.5 rounded ${isLive ? 'bg-green-500/20 text-green-400' : 'bg-slate-600 text-slate-400'}`}
        >
          {isLive ? 'LIVE' : 'PAUSED'}
        </button>
      </div>

      {/* Activity Feed */}
      <div className="space-y-2">
        {visibleEvents.map((event, idx) => {
          const config = PULSE_EVENT_CONFIG[event.type]
          return (
            <div
              key={event.id}
              onClick={() => onEventClick?.(event)}
              className={`flex items-start gap-2 p-2 rounded-lg bg-white/5 hover:bg-white/10 cursor-pointer transition-all ${idx === 0 ? 'animate-pulse' : ''}`}
              style={{ opacity: 1 - idx * 0.15 }}
            >
              <span className={`w-6 h-6 rounded flex items-center justify-center text-xs font-bold ${config.color}`}>
                {config.icon}
              </span>
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <span className={`text-[10px] px-1.5 py-0.5 rounded ${config.color}`}>
                    {config.label}
                  </span>
                  <span className="text-xs text-slate-500">{formatTime(event.timestamp)}</span>
                  {event.actor && (
                    <span className={`text-xs ${event.actor === 'REEE' ? 'text-indigo-400' : 'text-slate-400'}`}>
                      {event.actor}
                    </span>
                  )}
                </div>
                <p className="text-xs text-slate-300 truncate mt-0.5">{event.detail}</p>
                {event.inquiry_title && (
                  <p className="text-[10px] text-slate-500 truncate">→ {event.inquiry_title}</p>
                )}
              </div>
              {event.delta && (
                <span className="text-xs font-mono text-emerald-400 whitespace-nowrap">{event.delta}</span>
              )}
            </div>
          )
        })}
      </div>

      {/* Loop Status */}
      <div className="mt-3 pt-3 border-t border-white/10 flex items-center justify-between text-xs">
        <div className="flex items-center gap-4">
          <span className="text-slate-400">
            <span className="text-blue-400 font-medium">12</span> evidence/hr
          </span>
          <span className="text-slate-400">
            <span className="text-green-400 font-medium">3</span> tasks/hr
          </span>
          <span className="text-slate-400">
            <span className="text-purple-400 font-medium">2</span> resolutions/day
          </span>
        </div>
        <span className="text-slate-500">System healthy</span>
      </div>
    </div>
  )
}

// =============================================================================
// Main Component
// =============================================================================

function InquiryPage() {
  const navigate = useNavigate()
  const [searchParams] = useSearchParams()
  const searchQuery = searchParams.get('q') || ''
  const [inquiries, setInquiries] = useState<InquirySummary[]>([])
  const [loading, setLoading] = useState(true)
  const [sortBy, setSortBy] = useState<'stake' | 'entropy' | 'contributions' | 'created'>('stake')
  const [activeTab, setActiveTab] = useState<'discover' | 'questions'>('discover')

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

  // Calculate total bounty across all sources
  const totalBountyAvailable = topBountiedInquiries.reduce((sum, i) => sum + i.stake, 0) +
    TRENDING_ENTITIES.reduce((sum, e) => sum + e.total_bounty, 0)

  // LIVE WORLD EVENTS - What's actually happening right now
  const liveWorldEvents = [
    {
      id: 'wang-fuk',
      type: 'FIRE',
      location: 'Hong Kong',
      headline: 'Wang Fuk Court Fire',
      subhead: 'Death toll rises to 46, investigation ongoing',
      timestamp: '2h ago',
      status: 'developing' as const,
      inquiries: 4,
      hotQuestion: 'What is the confirmed death toll?',
      confidence: 72,
      bounty: 2500
    },
    {
      id: 'syria',
      type: 'CONFLICT',
      location: 'Syria',
      headline: 'Assad Regime Collapses',
      subhead: 'Rebel forces enter Damascus',
      timestamp: '6h ago',
      status: 'breaking' as const,
      inquiries: 12,
      hotQuestion: 'Where is Assad now?',
      confidence: 45,
      bounty: 8500
    },
    {
      id: 'korea',
      type: 'POLITICS',
      location: 'South Korea',
      headline: 'Martial Law Declared',
      subhead: 'President faces impeachment',
      timestamp: '1d ago',
      status: 'developing' as const,
      inquiries: 8,
      hotQuestion: 'Was the declaration constitutional?',
      confidence: 62,
      bounty: 3200
    },
    {
      id: 'openai',
      type: 'TECH',
      location: 'Global',
      headline: 'OpenAI Restructuring',
      subhead: 'Board drama continues',
      timestamp: '3d ago',
      status: 'active' as const,
      inquiries: 5,
      hotQuestion: 'Will Altman retain control?',
      confidence: 78,
      bounty: 1500
    }
  ]

  return (
    <div className="min-h-screen bg-slate-50">
      {/* BIG STATS - System Pulse */}
      <div className="bg-white border-b border-slate-200">
        <div className="max-w-6xl mx-auto px-4 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
              <span className="text-sm text-slate-500">Live</span>
            </div>
            <div className="flex items-center gap-8">
              <div className="text-right">
                <div className="text-3xl font-bold text-amber-600">${totalBountyAvailable.toLocaleString()}</div>
                <div className="text-xs text-slate-400">staked on truth</div>
              </div>
              <div className="text-right">
                <div className="text-3xl font-bold text-slate-800">{openInquiries.length}</div>
                <div className="text-xs text-slate-400">open questions</div>
              </div>
              <div className="text-right">
                <div className="text-3xl font-bold text-green-600">{resolvedInquiries.length}</div>
                <div className="text-xs text-slate-400">verified</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-6xl mx-auto px-4 py-6">
        {/* EVENTS FIRST - What's happening now */}
        <section className="mb-8">
          <div className="space-y-3">
            {/* Breaking */}
            <div
              onClick={() => navigate('/event-inquiry/syria')}
              className="bg-white rounded-xl border border-slate-200 p-4 cursor-pointer hover:shadow-md transition-all flex gap-4"
            >
              <div className="w-1 bg-red-500 rounded-full flex-shrink-0" />
              <div className="flex-1">
                <div className="flex items-center gap-2 mb-2">
                  <span className="flex items-center gap-1 text-[10px] font-bold uppercase tracking-wider text-red-600 bg-red-50 px-2 py-0.5 rounded-full">
                    <span className="w-1.5 h-1.5 bg-red-500 rounded-full animate-pulse" />
                    Breaking
                  </span>
                  <span className="text-xs text-slate-400">Syria · 6h ago</span>
                </div>
                <h2 className="text-lg font-semibold text-slate-800 mb-1">Assad Regime Collapses</h2>
                <p className="text-sm text-slate-500 mb-2">Rebel forces enter Damascus after rapid advance</p>
                {/* Key Question - THE HOOK */}
                <div className="bg-gradient-to-r from-indigo-50 to-purple-50 border border-indigo-100 rounded-lg p-3 flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="text-indigo-600 text-lg">?</span>
                    <span className="font-medium text-slate-800">Where is Assad now?</span>
                  </div>
                  <div className="flex items-center gap-3">
                    <span className="text-xs px-2 py-0.5 bg-red-100 text-red-600 rounded-full">45% certain</span>
                    <span className="text-amber-600 font-bold">$8,500</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Developing */}
            <div
              onClick={() => navigate('/event-inquiry/wang-fuk-court-fire')}
              className="bg-white rounded-xl border border-slate-200 p-4 cursor-pointer hover:shadow-md transition-all flex gap-4"
            >
              <div className="w-1 bg-amber-500 rounded-full flex-shrink-0" />
              <div className="flex-1">
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-[10px] font-bold uppercase tracking-wider text-amber-600 bg-amber-50 px-2 py-0.5 rounded-full">
                    Developing
                  </span>
                  <span className="text-xs text-slate-400">Hong Kong · 2h ago</span>
                </div>
                <h2 className="text-lg font-semibold text-slate-800 mb-1">Wang Fuk Court Fire</h2>
                <p className="text-sm text-slate-500 mb-2">Death toll rises to 46, investigation ongoing</p>
                {/* Key Question - THE HOOK */}
                <div className="bg-gradient-to-r from-indigo-50 to-purple-50 border border-indigo-100 rounded-lg p-3 flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="text-indigo-600 text-lg">?</span>
                    <span className="font-medium text-slate-800">What is the confirmed death toll?</span>
                  </div>
                  <div className="flex items-center gap-3">
                    <span className="text-xs px-2 py-0.5 bg-amber-100 text-amber-600 rounded-full">72% certain</span>
                    <span className="text-amber-600 font-bold">$2,500</span>
                  </div>
                </div>
              </div>
            </div>

            {/* More events row */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              <div
                onClick={() => navigate('/event-inquiry/korea')}
                className="bg-white rounded-lg border border-slate-200 p-3 cursor-pointer hover:shadow-md transition-all flex gap-3"
              >
                <div className="w-1 bg-slate-300 rounded-full flex-shrink-0" />
                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-[10px] text-slate-400">South Korea · 1d ago</span>
                    <span className="text-[10px] bg-slate-100 text-slate-500 px-1.5 py-0.5 rounded">Politics</span>
                  </div>
                  <h3 className="font-medium text-slate-800 text-sm mb-1">Martial Law Declared</h3>
                  <div className="flex items-center gap-2 text-xs">
                    <span className="text-amber-500">62%</span>
                    <span className="text-amber-600">$3,200</span>
                  </div>
                </div>
              </div>

              <div
                onClick={() => navigate('/event-inquiry/openai')}
                className="bg-white rounded-lg border border-slate-200 p-3 cursor-pointer hover:shadow-md transition-all flex gap-3"
              >
                <div className="w-1 bg-slate-300 rounded-full flex-shrink-0" />
                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-[10px] text-slate-400">Global · 3d ago</span>
                    <span className="text-[10px] bg-slate-100 text-slate-500 px-1.5 py-0.5 rounded">Tech</span>
                  </div>
                  <h3 className="font-medium text-slate-800 text-sm mb-1">OpenAI Restructuring</h3>
                  <div className="flex items-center gap-2 text-xs">
                    <span className="text-green-500">78%</span>
                    <span className="text-amber-600">$1,500</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* ENTITIES - Ongoing topics */}
        <section>
          <h3 className="text-sm font-medium text-slate-500 uppercase tracking-wide mb-4">Trending Topics</h3>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
            {TRENDING_ENTITIES.map(entity => (
              <div
                key={entity.id}
                onClick={() => navigate(`/entity-inquiry/${entity.slug}`)}
                className="bg-white rounded-lg border border-slate-200 p-3 cursor-pointer hover:shadow-md hover:-translate-y-0.5 transition-all"
              >
                <div className="flex items-center gap-2 mb-2">
                  {entity.image ? (
                    <img src={entity.image} alt={entity.name} className="w-8 h-8 rounded-full object-cover" />
                  ) : (
                    <span className="w-8 h-8 rounded-full bg-slate-100 flex items-center justify-center">
                      {entity.type === 'PERSON' ? '👤' : entity.type === 'ORG' ? '🏢' : '📍'}
                    </span>
                  )}
                  <div className="min-w-0">
                    <h4 className="font-medium text-slate-800 text-sm truncate">{entity.name}</h4>
                    <div className="text-[10px] text-slate-400">{entity.open_inquiries}q · ${entity.total_bounty}</div>
                  </div>
                </div>
                {entity.hot_inquiry && (
                  <div className="text-[11px] text-slate-500 line-clamp-2">{entity.hot_inquiry}</div>
                )}
              </div>
            ))}
          </div>
        </section>

        {/* FEATURED INQUIRIES - High bounty questions */}
        <section className="mt-8">
          <h3 className="text-sm font-medium text-slate-500 uppercase tracking-wide mb-4">Featured Questions</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
            {topBountiedInquiries.slice(0, 6).map(inq => (
              <div
                key={inq.id}
                onClick={() => handleSelect(inq.id)}
                className="bg-white rounded-lg border border-slate-200 p-4 cursor-pointer hover:shadow-md transition-all"
              >
                <div className="flex items-start justify-between gap-2 mb-2">
                  <h4 className="font-medium text-slate-800 text-sm line-clamp-2">{inq.title}</h4>
                  <span className="text-amber-600 font-bold text-sm whitespace-nowrap">${inq.stake}</span>
                </div>
                <div className="flex items-center gap-2 text-xs text-slate-400">
                  <span className={`px-1.5 py-0.5 rounded ${
                    inq.entropy_bits > 3.5 ? 'bg-red-100 text-red-600' :
                    inq.entropy_bits > 2 ? 'bg-amber-100 text-amber-600' :
                    'bg-green-100 text-green-600'
                  }`}>
                    {inq.entropy_bits > 3.5 ? 'Contested' : inq.entropy_bits > 2 ? 'Uncertain' : 'Confident'}
                  </span>
                  <span>{inq.contributions} contributions</span>
                </div>
                {inq.scope_entities && inq.scope_entities.length > 0 && (
                  <div className="flex flex-wrap gap-1 mt-2">
                    {inq.scope_entities.slice(0, 2).map(e => (
                      <span key={e} className="text-[10px] px-1.5 py-0.5 bg-slate-100 rounded text-slate-500">{e}</span>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        </section>

        {/* Activity pulse - minimal */}
        <div className="mt-8 pt-4 border-t border-slate-200 flex items-center gap-3 overflow-x-auto text-xs text-slate-400">
          <span className="flex items-center gap-1 flex-shrink-0">
            <span className="w-1.5 h-1.5 bg-green-500 rounded-full" />
            now
          </span>
          {INITIAL_PULSE_EVENTS.slice(0, 4).map(event => (
            <span key={event.id} className="flex-shrink-0">
              {event.detail.slice(0, 35)}...
            </span>
          ))}
        </div>
      </div>

      {/* Questions Tab - Full list view */}
      {activeTab === 'questions' && (
        <div className="max-w-7xl mx-auto px-6 pb-24">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-lg font-medium text-white">All Questions</h2>
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value as any)}
              className="text-sm bg-slate-800 border border-slate-700 rounded-lg px-3 py-1.5 text-white"
            >
              <option value="stake">By Bounty</option>
              <option value="entropy">By Uncertainty</option>
              <option value="contributions">By Activity</option>
            </select>
          </div>
          <div className="space-y-2">
            {sortedOpenInquiries.map((inq) => (
              <div
                key={inq.id}
                onClick={() => handleSelect(inq.id)}
                className="flex items-center justify-between p-4 bg-slate-900 hover:bg-slate-800 rounded-lg border border-slate-800 cursor-pointer transition-colors"
              >
                <div className="flex-1 min-w-0">
                  <div className="text-sm text-white truncate">{inq.title}</div>
                  <div className="flex items-center gap-2 mt-1">
                    {inq.scope_entities?.slice(0, 2).map(e => (
                      <span key={e} className="text-xs text-slate-500">{e}</span>
                    ))}
                  </div>
                </div>
                <div className="flex items-center gap-4 text-xs">
                  <span className="text-amber-400">${inq.stake}</span>
                  <span className={inq.entropy_bits > 3 ? 'text-red-400' : 'text-slate-500'}>
                    {inq.entropy_bits.toFixed(1)} bits
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

export default InquiryPage
