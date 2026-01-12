import React, { useState, useEffect } from 'react'
import { useParams, Link, useNavigate } from 'react-router-dom'

// =============================================================================
// Types
// =============================================================================

interface EntityInquiry {
  id: string
  question: string
  schema_type: 'count' | 'boolean' | 'categorical' | 'date' | 'text'
  timestamp?: string  // When the question applies (e.g., "as of Dec 2025")
  context?: string    // Additional context
  best_estimate: string | number | boolean
  confidence: number
  bounty: number
  contributors: number
  status: 'open' | 'resolved'
  created_by: string
  created_at: string
  related_entities: string[]
}

interface EntityData {
  id: string
  name: string
  type: 'PERSON' | 'ORG' | 'LOCATION' | 'EVENT' | 'CONCEPT'
  wikidata_qid?: string
  description: string
  image_url?: string
  properties: Array<{ label: string; value: string; inquiry_id?: string }>
  inquiries: EntityInquiry[]
  related_entities: Array<{ id: string; name: string; type: string; relation: string }>
}

// =============================================================================
// Simulated Data
// =============================================================================

const ENTITIES: Record<string, EntityData> = {
  'elon-musk': {
    id: 'elon-musk',
    name: 'Elon Musk',
    type: 'PERSON',
    wikidata_qid: 'Q317521',
    description: 'Business magnate, CEO of Tesla and SpaceX, owner of X (formerly Twitter)',
    image_url: 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/34/Elon_Musk_Royal_Society_%28crop2%29.jpg/220px-Elon_Musk_Royal_Society_%28crop2%29.jpg',
    properties: [
      { label: 'Born', value: 'June 28, 1971' },
      { label: 'Nationality', value: 'South African, Canadian, American' },
      { label: 'Net Worth', value: '~$250B', inquiry_id: 'inq_musk_networth' },
      { label: 'Children', value: '12', inquiry_id: 'inq_musk_children' },
      { label: 'Companies', value: 'Tesla, SpaceX, X, Neuralink, The Boring Company' },
    ],
    inquiries: [
      {
        id: 'inq_musk_children',
        question: 'How many children does Elon Musk have?',
        schema_type: 'count',
        timestamp: 'as of Dec 2025',
        best_estimate: 12,
        confidence: 0.85,
        bounty: 50,
        contributors: 8,
        status: 'open',
        created_by: '@factchecker',
        created_at: '2025-11-15',
        related_entities: ['Grimes', 'Justine Musk', 'Shivon Zilis']
      },
      {
        id: 'inq_musk_networth',
        question: 'What is Elon Musk\'s current net worth?',
        schema_type: 'count',
        timestamp: 'as of Dec 2025',
        context: 'Based on Forbes/Bloomberg estimates',
        best_estimate: 250000000000,
        confidence: 0.65,
        bounty: 100,
        contributors: 12,
        status: 'open',
        created_by: '@wealthtracker',
        created_at: '2025-12-01',
        related_entities: ['Tesla', 'SpaceX']
      },
      {
        id: 'inq_musk_twitter_purchase',
        question: 'How much did Elon Musk pay to acquire Twitter?',
        schema_type: 'count',
        timestamp: 'Oct 2022',
        best_estimate: 44000000000,
        confidence: 0.99,
        bounty: 0,
        contributors: 15,
        status: 'resolved',
        created_by: '@dealwatcher',
        created_at: '2022-10-28',
        related_entities: ['Twitter', 'X Corp']
      }
    ],
    related_entities: [
      { id: 'tesla', name: 'Tesla', type: 'ORG', relation: 'CEO' },
      { id: 'spacex', name: 'SpaceX', type: 'ORG', relation: 'CEO' },
      { id: 'x-corp', name: 'X Corp', type: 'ORG', relation: 'Owner' },
    ]
  },
  'japan': {
    id: 'japan',
    name: 'Japan',
    type: 'LOCATION',
    wikidata_qid: 'Q17',
    description: 'Island country in East Asia, known for earthquakes and tsunamis',
    image_url: 'https://upload.wikimedia.org/wikipedia/commons/thumb/9/9e/Flag_of_Japan.svg/220px-Flag_of_Japan.svg.png',
    properties: [
      { label: 'Capital', value: 'Tokyo' },
      { label: 'Population', value: '~125 million', inquiry_id: 'inq_japan_pop' },
      { label: 'Area', value: '377,975 km²' },
      { label: 'Currency', value: 'Japanese Yen (¥)' },
    ],
    inquiries: [
      {
        id: 'inq_japan_tsunami_today',
        question: 'Is there a tsunami warning in Japan today?',
        schema_type: 'boolean',
        timestamp: '2025-12-28',
        context: 'Japan Meteorological Agency alerts',
        best_estimate: false,
        confidence: 0.95,
        bounty: 25,
        contributors: 3,
        status: 'open',
        created_by: '@japanwatch',
        created_at: '2025-12-28',
        related_entities: ['Japan Meteorological Agency']
      },
      {
        id: 'inq_japan_earthquake',
        question: 'Was there a major earthquake (M6+) in Japan this week?',
        schema_type: 'boolean',
        timestamp: 'Week of Dec 23-29, 2025',
        best_estimate: true,
        confidence: 0.88,
        bounty: 50,
        contributors: 6,
        status: 'open',
        created_by: '@seismictracker',
        created_at: '2025-12-25',
        related_entities: ['USGS', 'Japan Meteorological Agency']
      },
      {
        id: 'inq_japan_pop',
        question: 'What is Japan\'s current population?',
        schema_type: 'count',
        timestamp: 'as of 2025',
        best_estimate: 124500000,
        confidence: 0.92,
        bounty: 30,
        contributors: 5,
        status: 'open',
        created_by: '@demographics',
        created_at: '2025-01-10',
        related_entities: ['Statistics Bureau of Japan']
      }
    ],
    related_entities: [
      { id: 'tokyo', name: 'Tokyo', type: 'LOCATION', relation: 'Capital' },
      { id: 'jma', name: 'Japan Meteorological Agency', type: 'ORG', relation: 'Weather authority' },
    ]
  }
}

// =============================================================================
// Inquiry Wizard Component
// =============================================================================

interface WizardStep {
  id: string
  title: string
  description: string
}

const WIZARD_STEPS: WizardStep[] = [
  { id: 'question', title: 'Your Question', description: 'What do you want to know?' },
  { id: 'type', title: 'Answer Type', description: 'What kind of answer?' },
  { id: 'context', title: 'Context', description: 'When and where does this apply?' },
  { id: 'bounty', title: 'Bounty', description: 'Fund the investigation' },
  { id: 'review', title: 'Review', description: 'Confirm and submit' },
]

function InquiryWizard({
  entity,
  onClose,
  onSubmit
}: {
  entity: EntityData
  onClose: () => void
  onSubmit: (inquiry: Partial<EntityInquiry>) => void
}) {
  const [step, setStep] = useState(0)
  const [formData, setFormData] = useState({
    question: '',
    schema_type: 'count' as EntityInquiry['schema_type'],
    timestamp: '',
    context: '',
    bounty: 10,
    related_entities: [] as string[]
  })
  const [similarInquiries, setSimilarInquiries] = useState<EntityInquiry[]>([])

  // Find similar inquiries as user types
  useEffect(() => {
    if (formData.question.length > 10) {
      const words = formData.question.toLowerCase().split(' ')
      const similar = entity.inquiries.filter(inq => {
        const inqWords = inq.question.toLowerCase()
        return words.some(w => w.length > 3 && inqWords.includes(w))
      })
      setSimilarInquiries(similar)
    } else {
      setSimilarInquiries([])
    }
  }, [formData.question, entity.inquiries])

  const currentStep = WIZARD_STEPS[step]

  const renderStepContent = () => {
    switch (currentStep.id) {
      case 'question':
        return (
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">
                What do you want to know about {entity.name}?
              </label>
              <textarea
                value={formData.question}
                onChange={(e) => setFormData({ ...formData, question: e.target.value })}
                placeholder={`e.g., "How many ${entity.type === 'PERSON' ? 'companies does' : 'people live in'} ${entity.name}..."`}
                className="w-full border border-slate-300 rounded-lg px-4 py-3 text-sm focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 resize-none"
                rows={3}
              />
            </div>

            {/* Similar inquiries warning */}
            {similarInquiries.length > 0 && (
              <div className="bg-amber-50 border border-amber-200 rounded-lg p-4">
                <div className="flex items-center gap-2 text-amber-700 font-medium text-sm mb-2">
                  <span>⚠️</span>
                  <span>Similar questions already exist</span>
                </div>
                <p className="text-xs text-amber-600 mb-3">
                  Consider contributing to an existing inquiry instead of creating a duplicate.
                </p>
                <div className="space-y-2">
                  {similarInquiries.slice(0, 3).map(inq => (
                    <div
                      key={inq.id}
                      className="bg-white rounded-lg p-3 border border-amber-200 cursor-pointer hover:border-indigo-300"
                      onClick={() => {/* Navigate to existing inquiry */}}
                    >
                      <div className="text-sm text-slate-700">{inq.question}</div>
                      <div className="flex items-center gap-3 mt-1 text-xs text-slate-500">
                        <span className="text-amber-600 font-medium">${inq.bounty} bounty</span>
                        <span>{inq.contributors} contributors</span>
                        <span>{Math.round(inq.confidence * 100)}% confident</span>
                      </div>
                      <div className="text-xs text-indigo-600 mt-1">
                        → Add to this bounty instead
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )

      case 'type':
        return (
          <div className="space-y-3">
            <label className="block text-sm font-medium text-slate-700 mb-2">
              What type of answer are you looking for?
            </label>
            {[
              { id: 'count', label: 'Number', desc: 'A numeric value (e.g., 12, $50M, 1.5kg)', icon: '#' },
              { id: 'boolean', label: 'Yes/No', desc: 'A true or false answer', icon: '✓' },
              { id: 'categorical', label: 'Choice', desc: 'One of several options', icon: '◉' },
              { id: 'date', label: 'Date', desc: 'A specific date or time', icon: '📅' },
              { id: 'text', label: 'Text', desc: 'A descriptive answer', icon: '📝' },
            ].map(type => (
              <div
                key={type.id}
                onClick={() => setFormData({ ...formData, schema_type: type.id as any })}
                className={`
                  p-4 rounded-lg border-2 cursor-pointer transition
                  ${formData.schema_type === type.id
                    ? 'border-indigo-500 bg-indigo-50'
                    : 'border-slate-200 hover:border-slate-300'
                  }
                `}
              >
                <div className="flex items-center gap-3">
                  <span className="text-xl w-8 text-center">{type.icon}</span>
                  <div>
                    <div className="font-medium text-slate-800">{type.label}</div>
                    <div className="text-xs text-slate-500">{type.desc}</div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )

      case 'context':
        return (
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">
                Time reference <span className="text-slate-400">(when does this apply?)</span>
              </label>
              <div className="grid grid-cols-2 gap-2 mb-2">
                {['Today', 'This week', 'As of Dec 2025', 'Historical'].map(preset => (
                  <button
                    key={preset}
                    onClick={() => setFormData({ ...formData, timestamp: preset })}
                    className={`px-3 py-2 text-sm rounded-lg border transition ${
                      formData.timestamp === preset
                        ? 'border-indigo-500 bg-indigo-50 text-indigo-700'
                        : 'border-slate-200 hover:border-slate-300'
                    }`}
                  >
                    {preset}
                  </button>
                ))}
              </div>
              <input
                type="text"
                value={formData.timestamp}
                onChange={(e) => setFormData({ ...formData, timestamp: e.target.value })}
                placeholder="Or enter custom timeframe..."
                className="w-full border border-slate-300 rounded-lg px-3 py-2 text-sm"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">
                Additional context <span className="text-slate-400">(optional)</span>
              </label>
              <textarea
                value={formData.context}
                onChange={(e) => setFormData({ ...formData, context: e.target.value })}
                placeholder="Any specific sources to check, methodology notes, etc."
                className="w-full border border-slate-300 rounded-lg px-3 py-2 text-sm resize-none"
                rows={2}
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">
                Related entities <span className="text-slate-400">(who/what is involved?)</span>
              </label>
              <div className="flex flex-wrap gap-2">
                {entity.related_entities.map(rel => (
                  <button
                    key={rel.id}
                    onClick={() => {
                      const current = formData.related_entities
                      setFormData({
                        ...formData,
                        related_entities: current.includes(rel.name)
                          ? current.filter(e => e !== rel.name)
                          : [...current, rel.name]
                      })
                    }}
                    className={`px-3 py-1.5 text-sm rounded-full border transition ${
                      formData.related_entities.includes(rel.name)
                        ? 'border-indigo-500 bg-indigo-50 text-indigo-700'
                        : 'border-slate-200 hover:border-slate-300'
                    }`}
                  >
                    {rel.name}
                  </button>
                ))}
              </div>
            </div>
          </div>
        )

      case 'bounty':
        return (
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">
                Set your bounty
              </label>
              <p className="text-xs text-slate-500 mb-4">
                Higher bounties attract more researchers and get answered faster.
                You only pay when the question is resolved with sufficient confidence.
              </p>

              <div className="grid grid-cols-4 gap-2 mb-4">
                {[10, 25, 50, 100].map(amount => (
                  <button
                    key={amount}
                    onClick={() => setFormData({ ...formData, bounty: amount })}
                    className={`py-3 rounded-lg border-2 font-medium transition ${
                      formData.bounty === amount
                        ? 'border-amber-500 bg-amber-50 text-amber-700'
                        : 'border-slate-200 hover:border-slate-300'
                    }`}
                  >
                    ${amount}
                  </button>
                ))}
              </div>

              <div className="flex items-center gap-3">
                <span className="text-slate-500">$</span>
                <input
                  type="number"
                  value={formData.bounty}
                  onChange={(e) => setFormData({ ...formData, bounty: parseInt(e.target.value) || 0 })}
                  min={5}
                  className="w-24 border border-slate-300 rounded-lg px-3 py-2 text-sm"
                />
                <span className="text-xs text-slate-400">Minimum $5</span>
              </div>
            </div>

            {/* Bounty impact estimate */}
            <div className="bg-slate-50 rounded-lg p-4">
              <div className="text-sm font-medium text-slate-700 mb-2">Estimated impact</div>
              <div className="grid grid-cols-3 gap-4 text-center">
                <div>
                  <div className="text-lg font-bold text-slate-800">
                    {formData.bounty < 25 ? '~3 days' : formData.bounty < 50 ? '~1 day' : '~12 hours'}
                  </div>
                  <div className="text-xs text-slate-500">Time to answer</div>
                </div>
                <div>
                  <div className="text-lg font-bold text-slate-800">
                    {formData.bounty < 25 ? '2-5' : formData.bounty < 50 ? '5-10' : '10+'}
                  </div>
                  <div className="text-xs text-slate-500">Contributors</div>
                </div>
                <div>
                  <div className="text-lg font-bold text-slate-800">
                    {formData.bounty < 25 ? '70%' : formData.bounty < 50 ? '85%' : '95%'}
                  </div>
                  <div className="text-xs text-slate-500">Resolution rate</div>
                </div>
              </div>
            </div>
          </div>
        )

      case 'review':
        return (
          <div className="space-y-4">
            <div className="bg-slate-50 rounded-lg p-4 space-y-3">
              <div>
                <div className="text-xs text-slate-500 uppercase tracking-wide">Question</div>
                <div className="font-medium text-slate-800">{formData.question}</div>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <div className="text-xs text-slate-500 uppercase tracking-wide">Answer type</div>
                  <div className="text-slate-800 capitalize">{formData.schema_type}</div>
                </div>
                <div>
                  <div className="text-xs text-slate-500 uppercase tracking-wide">Timeframe</div>
                  <div className="text-slate-800">{formData.timestamp || 'Not specified'}</div>
                </div>
              </div>
              {formData.context && (
                <div>
                  <div className="text-xs text-slate-500 uppercase tracking-wide">Context</div>
                  <div className="text-slate-800 text-sm">{formData.context}</div>
                </div>
              )}
              {formData.related_entities.length > 0 && (
                <div>
                  <div className="text-xs text-slate-500 uppercase tracking-wide">Related</div>
                  <div className="flex gap-1 flex-wrap">
                    {formData.related_entities.map(e => (
                      <span key={e} className="px-2 py-0.5 bg-indigo-100 text-indigo-700 rounded text-sm">{e}</span>
                    ))}
                  </div>
                </div>
              )}
            </div>

            <div className="bg-amber-50 border border-amber-200 rounded-lg p-4">
              <div className="flex items-center justify-between">
                <div>
                  <div className="font-medium text-slate-800">Your bounty</div>
                  <div className="text-xs text-slate-500">Paid when resolved</div>
                </div>
                <div className="text-2xl font-bold text-amber-600">${formData.bounty}</div>
              </div>
            </div>

            <p className="text-xs text-slate-500">
              By submitting, you agree to fund this inquiry. Your bounty will be distributed
              to contributors who help resolve the question with verified evidence.
            </p>
          </div>
        )
    }
  }

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-xl w-full max-w-lg max-h-[90vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="px-5 py-4 border-b border-slate-200 flex items-center justify-between">
          <div>
            <h3 className="font-semibold text-slate-800">Create Inquiry</h3>
            <p className="text-xs text-slate-500">About {entity.name}</p>
          </div>
          <button onClick={onClose} className="text-slate-400 hover:text-slate-600 text-xl">×</button>
        </div>

        {/* Progress */}
        <div className="px-5 py-3 bg-slate-50 border-b border-slate-200">
          <div className="flex items-center gap-2">
            {WIZARD_STEPS.map((s, i) => (
              <React.Fragment key={s.id}>
                <div
                  className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-medium ${
                    i < step ? 'bg-green-500 text-white' :
                    i === step ? 'bg-indigo-500 text-white' :
                    'bg-slate-200 text-slate-500'
                  }`}
                >
                  {i < step ? '✓' : i + 1}
                </div>
                {i < WIZARD_STEPS.length - 1 && (
                  <div className={`flex-1 h-0.5 ${i < step ? 'bg-green-500' : 'bg-slate-200'}`} />
                )}
              </React.Fragment>
            ))}
          </div>
          <div className="mt-2">
            <div className="text-sm font-medium text-slate-800">{currentStep.title}</div>
            <div className="text-xs text-slate-500">{currentStep.description}</div>
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-5">
          {renderStepContent()}
        </div>

        {/* Footer */}
        <div className="px-5 py-4 border-t border-slate-200 flex justify-between">
          <button
            onClick={() => step > 0 ? setStep(step - 1) : onClose()}
            className="px-4 py-2 text-sm text-slate-600 hover:bg-slate-100 rounded-lg"
          >
            {step > 0 ? '← Back' : 'Cancel'}
          </button>
          <button
            onClick={() => {
              if (step < WIZARD_STEPS.length - 1) {
                setStep(step + 1)
              } else {
                onSubmit(formData)
                onClose()
              }
            }}
            disabled={step === 0 && formData.question.length < 10}
            className="px-4 py-2 text-sm bg-indigo-600 hover:bg-indigo-700 disabled:bg-slate-300 text-white rounded-lg"
          >
            {step < WIZARD_STEPS.length - 1 ? 'Continue →' : 'Submit Inquiry'}
          </button>
        </div>
      </div>
    </div>
  )
}

// =============================================================================
// Entity Inquiry Card
// =============================================================================

function EntityInquiryCard({
  inquiry,
  onClick
}: {
  inquiry: EntityInquiry
  onClick: () => void
}) {
  const isResolved = inquiry.status === 'resolved'

  return (
    <div
      onClick={onClick}
      className={`
        rounded-xl p-4 cursor-pointer transition-all hover:shadow-md hover:-translate-y-0.5
        ${isResolved
          ? 'bg-green-50 border border-green-200'
          : 'bg-white border border-slate-200'
        }
      `}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            {inquiry.timestamp && (
              <span className="text-xs px-2 py-0.5 bg-slate-100 text-slate-600 rounded">
                {inquiry.timestamp}
              </span>
            )}
            {isResolved && (
              <span className="text-xs px-2 py-0.5 bg-green-100 text-green-700 rounded font-medium">
                ✓ Resolved
              </span>
            )}
          </div>
          <h3 className="font-medium text-slate-800 mb-2">{inquiry.question}</h3>

          {/* Stats row */}
          <div className="flex items-center gap-4 text-sm">
            <div className="flex items-center gap-1.5">
              <span className="text-slate-500">Best:</span>
              <span className="font-semibold text-slate-800">
                {typeof inquiry.best_estimate === 'boolean'
                  ? (inquiry.best_estimate ? 'Yes' : 'No')
                  : typeof inquiry.best_estimate === 'number' && inquiry.best_estimate > 1000000
                    ? `$${(inquiry.best_estimate / 1000000000).toFixed(0)}B`
                    : String(inquiry.best_estimate)
                }
              </span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className={`w-2 h-2 rounded-full ${
                inquiry.confidence >= 0.8 ? 'bg-green-400' :
                inquiry.confidence >= 0.5 ? 'bg-amber-400' : 'bg-red-400'
              }`}></span>
              <span className="text-slate-600">{Math.round(inquiry.confidence * 100)}%</span>
            </div>
            <div className="text-slate-500">
              {inquiry.contributors} contributors
            </div>
          </div>
        </div>

        {/* Bounty */}
        {inquiry.bounty > 0 && (
          <div className="text-right flex-shrink-0">
            <div className="text-lg font-bold text-amber-600">${inquiry.bounty}</div>
            <div className="text-xs text-slate-400">bounty</div>
          </div>
        )}
      </div>

      {/* Related entities */}
      {inquiry.related_entities.length > 0 && (
        <div className="flex gap-1 mt-3 flex-wrap">
          {inquiry.related_entities.slice(0, 3).map(e => (
            <span key={e} className="text-xs px-2 py-0.5 bg-indigo-50 text-indigo-600 rounded">
              {e}
            </span>
          ))}
        </div>
      )}

      {/* Footer */}
      <div className="flex items-center justify-between mt-3 pt-3 border-t border-slate-100 text-xs text-slate-400">
        <span>by {inquiry.created_by}</span>
        <span>{inquiry.created_at}</span>
      </div>
    </div>
  )
}

// =============================================================================
// Main Page Component
// =============================================================================

export default function EntityInquiryPage() {
  const { entityId } = useParams<{ entityId: string }>()
  const navigate = useNavigate()
  const [entity, setEntity] = useState<EntityData | null>(null)
  const [loading, setLoading] = useState(true)
  const [showWizard, setShowWizard] = useState(false)
  const [filter, setFilter] = useState<'all' | 'open' | 'resolved'>('all')

  useEffect(() => {
    const loadEntity = async () => {
      setLoading(true)

      // If it's an en_ ID, fetch from API
      if (entityId?.startsWith('en_')) {
        try {
          const response = await fetch(`/api/entities/${entityId}`)
          if (response.ok) {
            const data = await response.json()

            // Build properties from API data
            const properties: Array<{ label: string; value: string }> = []
            if (data.wikidata_description) {
              properties.push({ label: 'Description', value: data.wikidata_description })
            }
            if (data.aliases && data.aliases.length > 0) {
              properties.push({ label: 'Also known as', value: data.aliases.join(', ') })
            }
            if (data.claim_count) {
              properties.push({ label: 'Claims', value: String(data.claim_count) })
            }
            if (data.source_count) {
              properties.push({ label: 'Sources', value: String(data.source_count) })
            }
            if (data.last_active) {
              properties.push({ label: 'Last active', value: new Date(data.last_active).toLocaleDateString() })
            }

            // Transform API response to EntityData format
            const transformed: EntityData = {
              id: data.id,
              name: data.canonical_name || 'Unknown Entity',
              type: data.entity_type || 'PERSON',
              wikidata_qid: data.wikidata_qid,
              description: data.profile_summary || data.narrative || '',
              image_url: data.image_url,
              properties: properties,
              inquiries: [],
              related_entities: (data.related_events || []).map((evId: string) => ({
                id: evId,
                name: evId,
                type: 'EVENT',
                relation: 'involved in'
              }))
            }
            setEntity(transformed)
          }
        } catch (err) {
          console.error('Failed to load entity:', err)
        }
      } else if (entityId && ENTITIES[entityId]) {
        // Demo data for slug-based access
        setEntity(ENTITIES[entityId])
      }

      setLoading(false)
    }

    loadEntity()
  }, [entityId])

  if (loading) {
    return (
      <div className="flex items-center justify-center py-20">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600"></div>
      </div>
    )
  }

  if (!entity) {
    return (
      <div className="max-w-4xl mx-auto px-4 py-12 text-center">
        <h1 className="text-2xl font-bold text-slate-800 mb-4">Entity Not Found</h1>
        <Link to="/inquiry" className="text-indigo-600 hover:underline">← Back to Inquiries</Link>
      </div>
    )
  }

  const filteredInquiries = entity.inquiries.filter(inq => {
    if (filter === 'open') return inq.status === 'open'
    if (filter === 'resolved') return inq.status === 'resolved'
    return true
  })

  const openCount = entity.inquiries.filter(i => i.status === 'open').length
  const totalBounty = entity.inquiries.reduce((sum, i) => sum + i.bounty, 0)

  return (
    <div className="min-h-screen bg-slate-50">
      <main className="max-w-5xl mx-auto px-4 py-6">
        <div className="flex gap-6">
          {/* Main content */}
          <div className="flex-1">
            {/* Entity header */}
            <div className="bg-white rounded-xl border border-slate-200 p-6 mb-6">
              <div className="flex items-start gap-4">
                {entity.image_url && (
                  <img
                    src={entity.image_url}
                    alt={entity.name}
                    className="w-24 h-24 rounded-lg object-cover"
                  />
                )}
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="text-xs px-2 py-0.5 bg-slate-100 text-slate-600 rounded uppercase">
                      {entity.type}
                    </span>
                    {entity.wikidata_qid && (
                      <a
                        href={`https://www.wikidata.org/wiki/${entity.wikidata_qid}`}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-xs text-indigo-500 hover:underline"
                      >
                        Wikidata ↗
                      </a>
                    )}
                  </div>
                  <h1 className="text-2xl font-bold text-slate-800 mb-2">{entity.name}</h1>
                  <p className="text-slate-600">{entity.description}</p>
                </div>
              </div>

              {/* Properties */}
              <div className="mt-4 pt-4 border-t border-slate-100">
                <div className="grid grid-cols-2 md:grid-cols-3 gap-4 text-sm">
                  {entity.properties.map(prop => (
                    <div key={prop.label}>
                      <div className="text-slate-500">{prop.label}</div>
                      <div className="font-medium text-slate-800 flex items-center gap-1">
                        {prop.value}
                        {prop.inquiry_id && (
                          <sup className="text-indigo-500 cursor-pointer">[?]</sup>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Inquiries section */}
            <div className="bg-white rounded-xl border border-slate-200 p-6">
              <div className="flex items-center justify-between mb-4">
                <div>
                  <h2 className="text-lg font-semibold text-slate-800">Questions about {entity.name}</h2>
                  <p className="text-sm text-slate-500 mt-1">
                    {openCount} open · ${totalBounty} in bounties
                  </p>
                </div>
                <button
                  onClick={() => setShowWizard(true)}
                  className="px-4 py-2 bg-indigo-600 hover:bg-indigo-700 text-white text-sm rounded-lg font-medium"
                >
                  + Ask Question
                </button>
              </div>

              {/* Filter tabs */}
              <div className="flex gap-2 mb-4">
                {(['all', 'open', 'resolved'] as const).map(f => (
                  <button
                    key={f}
                    onClick={() => setFilter(f)}
                    className={`px-3 py-1.5 text-sm rounded-lg transition ${
                      filter === f
                        ? 'bg-indigo-100 text-indigo-700'
                        : 'text-slate-600 hover:bg-slate-100'
                    }`}
                  >
                    {f.charAt(0).toUpperCase() + f.slice(1)}
                  </button>
                ))}
              </div>

              {/* Inquiry list */}
              <div className="space-y-3">
                {filteredInquiries.map(inquiry => (
                  <EntityInquiryCard
                    key={inquiry.id}
                    inquiry={inquiry}
                    onClick={() => navigate(`/inquiry/${inquiry.id}`)}
                  />
                ))}
              </div>

              {filteredInquiries.length === 0 && (
                <div className="text-center py-8">
                  <div className="text-3xl mb-2">🤔</div>
                  <p className="text-slate-500">No {filter !== 'all' ? filter : ''} questions yet</p>
                  <button
                    onClick={() => setShowWizard(true)}
                    className="mt-3 text-indigo-600 hover:underline text-sm"
                  >
                    Be the first to ask →
                  </button>
                </div>
              )}
            </div>
          </div>

          {/* Sidebar */}
          <div className="w-72 flex-shrink-0 space-y-4">
            {/* Related entities */}
            <div className="bg-white rounded-xl border border-slate-200 p-4">
              <h3 className="font-medium text-slate-700 mb-3">Related</h3>
              <div className="space-y-2">
                {entity.related_entities.map(rel => (
                  <Link
                    key={rel.id}
                    to={rel.id.startsWith('ev_') ? `/event/${rel.id}` : `/entity/${rel.id}`}
                    className="flex items-center justify-between p-2 rounded-lg hover:bg-slate-50 transition"
                  >
                    <div>
                      <div className="text-sm font-medium text-slate-800">{rel.name}</div>
                      <div className="text-xs text-slate-500">{rel.relation}</div>
                    </div>
                    <span className="text-xs text-slate-400">→</span>
                  </Link>
                ))}
              </div>
            </div>

            {/* Quick stats */}
            <div className="bg-gradient-to-br from-amber-50 to-orange-50 border border-amber-200 rounded-xl p-4">
              <h3 className="font-medium text-slate-700 mb-3">Bounty Pool</h3>
              <div className="text-2xl font-bold text-amber-600 mb-1">${totalBounty}</div>
              <p className="text-xs text-slate-500 mb-3">
                across {openCount} open questions
              </p>
              <button
                onClick={() => setShowWizard(true)}
                className="w-full px-3 py-2 bg-amber-500 hover:bg-amber-600 text-white text-sm rounded-lg font-medium"
              >
                Add Your Question
              </button>
            </div>
          </div>
        </div>
      </main>

      {/* Wizard modal */}
      {showWizard && (
        <InquiryWizard
          entity={entity}
          onClose={() => setShowWizard(false)}
          onSubmit={(data) => {
            console.log('New inquiry:', data)
            // Would submit to API
          }}
        />
      )}
    </div>
  )
}
