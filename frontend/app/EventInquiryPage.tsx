import React, { useState, useEffect } from 'react'
import { useParams, Link, useNavigate } from 'react-router-dom'
import { MapContainer, TileLayer, Marker, CircleMarker, useMap } from 'react-leaflet'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'

// =============================================================================
// Types
// =============================================================================

interface InquiryRef {
  id: string
  question: string
  type: 'auto' | 'user'
  best_estimate: string | number
  confidence: number
  bounty: number
  status: 'open' | 'resolved'
}

interface Entity {
  id: string
  name: string
  type: 'PERSON' | 'ORG' | 'LOCATION' | 'EVENT'
  wikidata_qid?: string
  description?: string
}

interface TimelineEvent {
  id: string
  time: string
  text: string
  source?: string
}

interface EventData {
  id: string
  slug: string
  title: string
  event_type: string
  location: string
  coordinates: { lat: number; lon: number }
  date_start: string
  date_end?: string
  status: 'live' | 'developing' | 'resolved'
  total_bounty: number
  claim_count: number
  source_count: number
  // Rich content
  lead_paragraph: string
  body_sections: Array<{
    id: string
    title: string
    paragraphs: string[]  // With inline entity/inquiry markers
    inquiries: InquiryRef[]
  }>
  key_figures: Array<{ label: string; value: string; inquiry_id?: string }>
  entities: Entity[]
  timeline: TimelineEvent[]
  sources: Array<{ name: string; url?: string; type: string }>
  related_events?: Array<{ id: string; title: string; relation: string }>
}

// =============================================================================
// Simulated Data - Wang Fuk Court Fire (Wikipedia-style)
// =============================================================================

const WANG_FUK_EVENT: EventData = {
  id: 'evt_wang_fuk_fire',
  slug: 'wang-fuk-court-fire',
  title: 'Wang Fuk Court Fire',
  event_type: 'Fire',
  location: 'Tai Po, Hong Kong',
  coordinates: { lat: 22.4513, lon: 114.1694 },
  date_start: '2025-11-26',
  status: 'developing',
  total_bounty: 2500,
  claim_count: 156,
  source_count: 24,
  lead_paragraph: `The **Wang Fuk Court fire** was a devastating blaze that broke out at a public housing estate in [[Tai Po]], [[Hong Kong]] on November 26, 2025. The fire, which started around 11:00 PM local time, rapidly spread through scaffolding and protective netting that had been installed as part of renovation work, ultimately affecting multiple residential towers. It is considered one of the deadliest fires in Hong Kong's recent history, with [[inq:death_toll|contested casualty figures]] ranging from initial reports to significantly higher numbers after DNA identification.`,

  body_sections: [
    {
      id: 'background',
      title: 'Background',
      paragraphs: [
        `Wang Fuk Court is a public housing estate managed by the [[Hong Kong Housing Authority]], located in the Tai Po district of the New Territories. The estate, built in the 1980s, had been undergoing exterior renovation work since September 2025. The renovation project involved the installation of scaffolding and protective netting around several residential blocks.`,
        `Residents had previously raised concerns about the materials being used, particularly [[styrofoam]] insulation panels which they believed posed a fire risk. These concerns were reportedly dismissed by contractors.`
      ],
      inquiries: []
    },
    {
      id: 'fire',
      title: 'The Fire',
      paragraphs: [
        `The fire broke out at approximately 11:00 PM on November 26, 2025. According to witnesses, flames were first observed on scaffolding near the lower floors of Block A before rapidly spreading upward. The protective netting and scaffolding materials, combined with strong winds, allowed the fire to engulf the building facade within minutes.`,
        `Many residents were trapped in their apartments as the fire blocked stairwells and elevators. Some attempted to escape via windows, while others waited for rescue on balconies. The [[Hong Kong Fire Services Department]] deployed over [[inq:fire_trucks|100 firefighting vehicles]] to the scene.`,
        `The fire burned for approximately 12 hours before being brought under control. Several adjacent buildings in the estate also sustained damage.`
      ],
      inquiries: []
    },
    {
      id: 'casualties',
      title: 'Casualties',
      paragraphs: [
        `The fire resulted in significant loss of life, though [[inq:death_toll|the exact death toll remains contested]]. Initial reports from local media suggested around 36 fatalities, but this figure was revised upward multiple times as rescue operations continued and DNA testing identified additional victims.`,
        `As of December 2025, authorities have confirmed that the death toll exceeds 100, with some international media outlets reporting figures as high as 160. Nine [[Indonesia]]n domestic workers were among the confirmed victims.`,
        `Over 200 people were injured, with dozens requiring hospitalization for smoke inhalation and burns.`
      ],
      inquiries: [
        {
          id: 'inq_death_toll',
          question: 'What is the confirmed death toll?',
          type: 'auto',
          best_estimate: 160,
          confidence: 0.62,
          bounty: 800,
          status: 'open'
        }
      ]
    },
    {
      id: 'response',
      title: 'Government Response',
      paragraphs: [
        `[[Chief Executive]] [[John Lee]] visited the scene on November 27 and announced the establishment of an independent investigation committee, to be led by Judge [[David Lok]]. The committee has been given nine months to submit its findings.`,
        `The government also announced a HK$300 million relief fund for affected residents, though [[inq:relief_fund|disbursement has been slow]]. Temporary housing has been provided to displaced families.`
      ],
      inquiries: [
        {
          id: 'inq_relief_fund',
          question: 'Has the relief fund been disbursed?',
          type: 'user',
          best_estimate: 'partially',
          confidence: 0.35,
          bounty: 500,
          status: 'open'
        }
      ]
    },
    {
      id: 'investigation',
      title: 'Investigation and Arrests',
      paragraphs: [
        `The Hong Kong Police Force has made [[inq:arrests|multiple arrests]] in connection with the fire. Those arrested include contractors, site supervisors, and company directors associated with the renovation project.`,
        `The [[ICAC|Independent Commission Against Corruption]] has also opened an investigation into potential corruption related to the awarding of the renovation contract and building inspection procedures.`,
        `Preliminary findings suggest that safety protocols were not followed, and that the scaffolding materials did not meet fire safety standards.`
      ],
      inquiries: [
        {
          id: 'inq_arrests',
          question: 'How many arrests have been made?',
          type: 'auto',
          best_estimate: 20,
          confidence: 0.78,
          bounty: 200,
          status: 'open'
        },
        {
          id: 'inq_safety_inspection',
          question: 'Were safety inspections conducted?',
          type: 'user',
          best_estimate: 'unknown',
          confidence: 0.35,
          bounty: 500,
          status: 'open'
        }
      ]
    }
  ],

  key_figures: [
    { label: 'Deaths', value: '~160', inquiry_id: 'inq_death_toll' },
    { label: 'Injured', value: '200+' },
    { label: 'Arrests', value: '20', inquiry_id: 'inq_arrests' },
    { label: 'Relief Fund', value: 'HK$300M' }
  ],

  entities: [
    { id: 'e1', name: 'Tai Po', type: 'LOCATION', wikidata_qid: 'Q860562', description: 'District in Hong Kong' },
    { id: 'e2', name: 'Hong Kong', type: 'LOCATION', wikidata_qid: 'Q8646', description: 'Special administrative region of China' },
    { id: 'e3', name: 'John Lee', type: 'PERSON', wikidata_qid: 'Q15981939', description: 'Chief Executive of Hong Kong' },
    { id: 'e4', name: 'Hong Kong Housing Authority', type: 'ORG', description: 'Public housing agency' },
    { id: 'e5', name: 'Hong Kong Fire Services', type: 'ORG', description: 'Fire and rescue service' },
    { id: 'e6', name: 'ICAC', type: 'ORG', description: 'Anti-corruption agency' }
  ],

  timeline: [
    { id: 't1', time: '2025-11-26T23:00', text: 'Fire breaks out on scaffolding', source: 'RTHK' },
    { id: 't2', time: '2025-11-26T23:15', text: 'First fire trucks arrive on scene', source: 'SCMP' },
    { id: 't3', time: '2025-11-26T23:45', text: 'Fire spreads to adjacent blocks', source: 'Reuters' },
    { id: 't4', time: '2025-11-27T03:00', text: 'Chief Executive visits scene', source: 'gov.hk' },
    { id: 't5', time: '2025-11-27T11:00', text: 'Fire brought under control', source: 'AP' },
    { id: 't6', time: '2025-11-28T10:00', text: 'Investigation committee announced', source: 'SCMP' },
    { id: 't7', time: '2025-12-02T00:00', text: 'Death toll revised after DNA tests', source: 'HKFP' }
  ],

  sources: [
    { name: 'Reuters', type: 'Wire' },
    { name: 'Associated Press', type: 'Wire' },
    { name: 'BBC', type: 'International' },
    { name: 'The Guardian', type: 'International' },
    { name: 'South China Morning Post', type: 'Local' },
    { name: 'Hong Kong Free Press', type: 'Local' },
    { name: 'RTHK', type: 'Local' }
  ],

  related_events: [
    { id: 'evt_hk_building_safety', title: 'Hong Kong Building Safety Review 2025', relation: 'caused' }
  ]
}

// =============================================================================
// Helper Components
// =============================================================================

// Map marker icon
const createEpicenterIcon = () => L.divIcon({
  className: 'epicenter-marker',
  html: `<div style="
    background: radial-gradient(circle, #ef4444 0%, #dc2626 100%);
    width: 12px; height: 12px; border-radius: 50%;
    border: 2px solid #fff;
    box-shadow: 0 0 8px rgba(239, 68, 68, 0.6);
  "></div>`,
  iconSize: [12, 12],
  iconAnchor: [6, 6],
})

const SetView: React.FC<{ center: [number, number]; zoom: number }> = ({ center, zoom }) => {
  const map = useMap()
  useEffect(() => { map.setView(center, zoom) }, [center, zoom, map])
  return null
}

// Inline Inquiry Marker - minimal footnote-style, non-intrusive
function InlineInquiryMarker({ inquiry, onClick }: { inquiry: InquiryRef; onClick: () => void }) {
  const [showTooltip, setShowTooltip] = useState(false)

  // Color based on confidence (subtle)
  const dotColor = inquiry.confidence >= 0.7 ? 'bg-green-400' : inquiry.confidence >= 0.4 ? 'bg-amber-400' : 'bg-red-400'

  return (
    <span className="relative inline-block">
      <sup
        onClick={(e) => { e.stopPropagation(); onClick() }}
        onMouseEnter={() => setShowTooltip(true)}
        onMouseLeave={() => setShowTooltip(false)}
        className="cursor-pointer text-indigo-500 hover:text-indigo-700 font-medium ml-0.5 select-none"
        style={{ fontSize: '0.65em', verticalAlign: 'super' }}
      >
        [?]
      </sup>

      {/* Hover tooltip with inquiry details */}
      {showTooltip && (
        <div className="absolute bottom-full left-0 mb-1 z-50 w-64 p-3 bg-slate-800 text-white rounded-lg shadow-xl text-xs pointer-events-none">
          <div className="font-medium mb-1.5">{inquiry.question}</div>
          <div className="flex items-center gap-3 text-slate-300">
            <span>Best: <span className="text-white font-medium">{inquiry.best_estimate}</span></span>
            <span className="flex items-center gap-1">
              <span className={`w-1.5 h-1.5 rounded-full ${dotColor}`}></span>
              {Math.round(inquiry.confidence * 100)}%
            </span>
            {inquiry.bounty > 0 && (
              <span className="text-amber-400">${inquiry.bounty}</span>
            )}
          </div>
          <div className="text-indigo-300 mt-1.5 text-[10px]">Click to investigate →</div>
          {/* Arrow */}
          <div className="absolute top-full left-4 -mt-1 border-4 border-transparent border-t-slate-800"></div>
        </div>
      )}
    </span>
  )
}

// Entity link with hover card
function EntityLink({ children, entity }: { children: string; entity?: Entity }) {
  const [showCard, setShowCard] = useState(false)

  if (!entity) {
    return <span className="text-indigo-600 hover:underline cursor-pointer">{children}</span>
  }

  return (
    <span className="relative inline-block">
      <span
        className="text-indigo-600 hover:underline cursor-pointer"
        onMouseEnter={() => setShowCard(true)}
        onMouseLeave={() => setShowCard(false)}
      >
        {children}
      </span>
      {showCard && (
        <div className="absolute bottom-full left-0 mb-2 z-50 w-64 p-3 bg-white rounded-lg shadow-lg border border-slate-200">
          <div className="font-medium text-slate-800">{entity.name}</div>
          <div className="text-xs text-slate-500 mt-1">{entity.description}</div>
          {entity.wikidata_qid && (
            <div className="text-xs text-indigo-500 mt-2">
              Wikidata: {entity.wikidata_qid}
            </div>
          )}
        </div>
      )}
    </span>
  )
}

// Parse Wikipedia-style markup in text
function RichText({
  text,
  entities,
  inquiries,
  onInquiryClick
}: {
  text: string
  entities: Entity[]
  inquiries: InquiryRef[]
  onInquiryClick: (id: string) => void
}) {
  // Parse [[entity]] and [[inq:id|text]] patterns
  const parts: React.ReactNode[] = []
  let lastIndex = 0

  // Pattern for [[text]] or [[inq:id|text]] or **bold**
  const pattern = /\[\[([^\]]+)\]\]|\*\*([^*]+)\*\*/g
  let match

  while ((match = pattern.exec(text)) !== null) {
    // Add text before match
    if (match.index > lastIndex) {
      parts.push(text.slice(lastIndex, match.index))
    }

    if (match[1]) {
      // Wiki-style link
      const content = match[1]
      if (content.startsWith('inq:')) {
        // Inquiry reference
        const [, rest] = content.split('inq:')
        const [inquiryId, displayText] = rest.includes('|') ? rest.split('|') : [rest, null]
        const inquiry = inquiries.find(i => i.id === inquiryId)

        if (inquiry) {
          parts.push(
            <span key={match.index}>
              {displayText && <span>{displayText}</span>}
              <InlineInquiryMarker
                inquiry={inquiry}
                onClick={() => onInquiryClick(inquiry.id)}
              />
            </span>
          )
        } else {
          parts.push(displayText || inquiryId)
        }
      } else {
        // Entity link
        const entity = entities.find(e =>
          e.name.toLowerCase() === content.toLowerCase()
        )
        parts.push(
          <EntityLink key={match.index} entity={entity}>
            {content}
          </EntityLink>
        )
      }
    } else if (match[2]) {
      // Bold text
      parts.push(<strong key={match.index}>{match[2]}</strong>)
    }

    lastIndex = match.index + match[0].length
  }

  // Add remaining text
  if (lastIndex < text.length) {
    parts.push(text.slice(lastIndex))
  }

  return <>{parts}</>
}

// Infobox (Wikipedia-style sidebar)
function Infobox({ event, onInquiryClick }: { event: EventData; onInquiryClick: (id: string) => void }) {
  return (
    <div className="bg-slate-50 border border-slate-200 rounded-lg overflow-hidden w-72 float-right ml-6 mb-4">
      {/* Map */}
      <div className="h-40">
        <MapContainer
          center={[event.coordinates.lat, event.coordinates.lon]}
          zoom={11}
          style={{ height: '100%', width: '100%' }}
          zoomControl={false}
          attributionControl={false}
          dragging={false}
          scrollWheelZoom={false}
        >
          <TileLayer url="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png" />
          <Marker position={[event.coordinates.lat, event.coordinates.lon]} icon={createEpicenterIcon()} />
          <CircleMarker
            center={[event.coordinates.lat, event.coordinates.lon]}
            radius={15}
            pathOptions={{ color: '#ef4444', fillColor: '#ef4444', fillOpacity: 0.1, weight: 1 }}
          />
          <SetView center={[event.coordinates.lat, event.coordinates.lon]} zoom={11} />
        </MapContainer>
      </div>

      {/* Title */}
      <div className="px-3 py-2 bg-slate-200 text-center">
        <div className="font-bold text-slate-800">{event.title}</div>
      </div>

      {/* Details */}
      <div className="p-3 text-sm">
        <table className="w-full">
          <tbody>
            <tr className="border-b border-slate-200">
              <td className="py-1.5 text-slate-500 pr-2">Date</td>
              <td className="py-1.5 text-slate-800">{new Date(event.date_start).toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric' })}</td>
            </tr>
            <tr className="border-b border-slate-200">
              <td className="py-1.5 text-slate-500 pr-2">Location</td>
              <td className="py-1.5 text-slate-800">{event.location}</td>
            </tr>
            <tr className="border-b border-slate-200">
              <td className="py-1.5 text-slate-500 pr-2">Type</td>
              <td className="py-1.5 text-slate-800">{event.event_type}</td>
            </tr>
            {event.key_figures.map((fig, i) => (
              <tr key={i} className={i < event.key_figures.length - 1 ? 'border-b border-slate-200' : ''}>
                <td className="py-1.5 text-slate-500 pr-2">{fig.label}</td>
                <td className="py-1.5 text-slate-800 flex items-center gap-1">
                  {fig.value}
                  {fig.inquiry_id && (
                    <button
                      onClick={() => onInquiryClick(fig.inquiry_id!)}
                      className="text-[10px] text-amber-600 bg-amber-50 px-1 rounded hover:bg-amber-100"
                      title="Contested value - click to investigate"
                    >
                      ?
                    </button>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Bounty badge */}
      {event.total_bounty > 0 && (
        <div className="px-3 py-2 bg-amber-50 border-t border-amber-200 text-center">
          <div className="text-xs text-amber-600">Open Bounties</div>
          <div className="text-lg font-bold text-amber-700">${event.total_bounty.toLocaleString()}</div>
        </div>
      )}
    </div>
  )
}

// Timeline sidebar
function TimelineSidebar({ events }: { events: TimelineEvent[] }) {
  const [expanded, setExpanded] = useState(false)
  const displayEvents = expanded ? events : events.slice(0, 4)

  return (
    <div className="bg-white border border-slate-200 rounded-lg overflow-hidden">
      <div className="px-3 py-2 bg-slate-50 border-b border-slate-200 flex items-center justify-between">
        <span className="font-medium text-slate-700 text-sm">Timeline</span>
        <span className="text-xs text-slate-400">{events.length} events</span>
      </div>
      <div className="p-3">
        <div className="relative pl-4">
          <div className="absolute left-[5px] top-2 bottom-2 w-0.5 bg-gradient-to-b from-indigo-400 to-slate-200" />
          <div className="space-y-3">
            {displayEvents.map((evt) => (
              <div key={evt.id} className="relative">
                <div className="absolute -left-[9px] top-1 w-2 h-2 rounded-full bg-indigo-500 border-2 border-white" />
                <div className="text-xs">
                  <div className="text-slate-400">
                    {new Date(evt.time).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}
                  </div>
                  <div className="text-slate-700">{evt.text}</div>
                  {evt.source && <div className="text-slate-400 italic">{evt.source}</div>}
                </div>
              </div>
            ))}
          </div>
        </div>
        {events.length > 4 && (
          <button
            onClick={() => setExpanded(!expanded)}
            className="mt-3 text-xs text-indigo-600 hover:underline w-full text-center"
          >
            {expanded ? 'Show less' : `Show ${events.length - 4} more`}
          </button>
        )}
      </div>
    </div>
  )
}

// Open inquiries sidebar panel
function OpenInquiriesPanel({
  sections,
  onInquiryClick
}: {
  sections: EventData['body_sections']
  onInquiryClick: (id: string) => void
}) {
  const allInquiries = sections.flatMap(s => s.inquiries)
  if (allInquiries.length === 0) return null

  return (
    <div className="bg-white border border-slate-200 rounded-lg overflow-hidden">
      <div className="px-3 py-2 bg-amber-50 border-b border-amber-200 flex items-center justify-between">
        <span className="font-medium text-amber-800 text-sm">Open Questions</span>
        <span className="text-xs bg-amber-200 text-amber-800 px-1.5 py-0.5 rounded-full">
          {allInquiries.length}
        </span>
      </div>
      <div className="p-2 space-y-2">
        {allInquiries.map(inq => (
          <div
            key={inq.id}
            onClick={() => onInquiryClick(inq.id)}
            className="p-2 rounded-lg bg-slate-50 hover:bg-slate-100 cursor-pointer transition text-xs"
          >
            <div className="flex items-start justify-between gap-2">
              <span className="text-slate-700 line-clamp-2">{inq.question}</span>
              <span className={`px-1.5 py-0.5 rounded text-[10px] font-medium shrink-0 ${
                inq.type === 'auto' ? 'bg-violet-100 text-violet-700' : 'bg-cyan-100 text-cyan-700'
              }`}>
                {inq.type === 'auto' ? 'AUTO' : 'USER'}
              </span>
            </div>
            <div className="flex items-center gap-2 mt-1.5 text-slate-500">
              <span>{Math.round(inq.confidence * 100)}% conf</span>
              {inq.bounty > 0 && <span className="text-amber-600 font-medium">${inq.bounty}</span>}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

// =============================================================================
// Main Component
// =============================================================================

export default function EventInquiryPage() {
  const { eventSlug } = useParams<{ eventSlug: string }>()
  const navigate = useNavigate()
  const [event, setEvent] = useState<EventData | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    if (eventSlug === 'wang-fuk-court-fire') {
      setEvent(WANG_FUK_EVENT)
    }
    setLoading(false)
  }, [eventSlug])

  const handleInquiryClick = (inquiryId: string) => {
    navigate(`/inquiry/${inquiryId}`)
  }

  // Collect all inquiries for inline reference
  const allInquiries = event?.body_sections.flatMap(s => s.inquiries) || []

  if (loading) {
    return (
      <div className="flex items-center justify-center py-20">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600" />
      </div>
    )
  }

  if (!event) {
    return (
      <div className="max-w-4xl mx-auto px-4 py-12 text-center">
        <h1 className="text-2xl font-bold text-slate-800 mb-4">Event Not Found</h1>
        <Link to="/inquiry" className="text-indigo-600 hover:underline">← Back to Inquiries</Link>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-white">
      {/* Minimal header */}
      <header className="bg-white border-b border-slate-200 sticky top-0 z-50">
        <div className="max-w-6xl mx-auto px-4 py-2 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Link to="/inquiry" className="text-lg font-bold text-indigo-600">φ HERE</Link>
            <span className="text-slate-300">|</span>
            <span className="text-sm text-slate-500">Event</span>
          </div>
          {event.status !== 'resolved' && (
            <span className={`px-2 py-0.5 rounded text-xs flex items-center gap-1 ${
              event.status === 'live' ? 'bg-red-100 text-red-700' : 'bg-amber-100 text-amber-700'
            }`}>
              <span className={`w-1.5 h-1.5 rounded-full animate-pulse ${
                event.status === 'live' ? 'bg-red-500' : 'bg-amber-500'
              }`} />
              {event.status === 'live' ? 'Live' : 'Developing'}
            </span>
          )}
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-4 py-6">
        <div className="flex gap-6">
          {/* Main article content */}
          <article className="flex-1 min-w-0">
            {/* Title */}
            <h1 className="text-3xl font-serif font-bold text-slate-900 mb-2">
              {event.title}
            </h1>

            {/* Article meta */}
            <div className="text-sm text-slate-500 mb-4 pb-4 border-b border-slate-200">
              From HERE.news, the collaborative encyclopedia
              <span className="mx-2">·</span>
              <span>{event.source_count} sources</span>
              <span className="mx-2">·</span>
              <span>{event.claim_count} claims verified</span>
            </div>

            {/* Infobox (floats right) */}
            <Infobox event={event} onInquiryClick={handleInquiryClick} />

            {/* Lead paragraph */}
            <p className="text-lg text-slate-700 leading-relaxed mb-6">
              <RichText
                text={event.lead_paragraph}
                entities={event.entities}
                inquiries={allInquiries}
                onInquiryClick={handleInquiryClick}
              />
            </p>

            {/* Body sections */}
            {event.body_sections.map(section => (
              <section key={section.id} className="mb-8">
                <h2 className="text-xl font-serif font-bold text-slate-800 mb-3 pb-1 border-b border-slate-200">
                  {section.title}
                </h2>
                {section.paragraphs.map((para, i) => (
                  <p key={i} className="text-slate-700 leading-relaxed mb-4">
                    <RichText
                      text={para}
                      entities={event.entities}
                      inquiries={allInquiries}
                      onInquiryClick={handleInquiryClick}
                    />
                  </p>
                ))}
              </section>
            ))}

            {/* Clear float */}
            <div className="clear-both" />

            {/* Sources section */}
            <section className="mt-8 pt-6 border-t border-slate-200">
              <h2 className="text-lg font-serif font-bold text-slate-800 mb-3">Sources</h2>
              <div className="flex flex-wrap gap-2">
                {event.sources.map((src, i) => (
                  <span
                    key={i}
                    className={`px-2 py-1 rounded text-xs ${
                      src.type === 'Wire' ? 'bg-blue-50 text-blue-700' :
                      src.type === 'International' ? 'bg-green-50 text-green-700' :
                      'bg-purple-50 text-purple-700'
                    }`}
                  >
                    {src.name}
                  </span>
                ))}
              </div>
            </section>

            {/* Related events */}
            {event.related_events && event.related_events.length > 0 && (
              <section className="mt-6">
                <h3 className="text-sm font-medium text-slate-500 mb-2">Related Events</h3>
                <div className="flex flex-wrap gap-2">
                  {event.related_events.map(rel => (
                    <Link
                      key={rel.id}
                      to={`/event-inquiry/${rel.id}`}
                      className="px-3 py-1.5 bg-slate-50 border border-slate-200 rounded text-sm text-slate-700 hover:bg-slate-100"
                    >
                      {rel.relation}: {rel.title}
                    </Link>
                  ))}
                </div>
              </section>
            )}
          </article>

          {/* Right sidebar */}
          <aside className="w-64 flex-shrink-0 space-y-4">
            <TimelineSidebar events={event.timeline} />
            <OpenInquiriesPanel
              sections={event.body_sections}
              onInquiryClick={handleInquiryClick}
            />
          </aside>
        </div>
      </main>
    </div>
  )
}
