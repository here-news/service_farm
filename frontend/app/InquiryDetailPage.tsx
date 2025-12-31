import React, { useState, useEffect, useRef } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import {
  InquiryDetail,
  InquiryTrace,
  Contribution,
  ContributionInput,
  ObservationKind,
  Task
} from './types/inquiry'
import { SIMULATED_INQUIRIES, generateSimulatedTrace } from './data/simulatedInquiries'

// =============================================================================
// SimpleShareBox - Natural text input with upload (integrated into Community)
// =============================================================================
const MAX_CHARS = 500

function SimpleShareBox({
  onSubmit,
  isSubmitting
}: {
  onSubmit: (input: ContributionInput) => Promise<void>
  isSubmitting: boolean
}) {
  const [text, setText] = useState('')
  const [files, setFiles] = useState<File[]>([])
  const fileInputRef = useRef<HTMLInputElement>(null)

  const charsLeft = MAX_CHARS - text.length
  const isOverLimit = charsLeft < 0

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!text.trim() || isOverLimit) return

    await onSubmit({
      type: 'evidence', // AI will classify the type from natural text
      text: text.trim(),
    })

    setText('')
    setFiles([])
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      setFiles(Array.from(e.target.files))
    }
  }

  const removeFile = (index: number) => {
    setFiles(files.filter((_, i) => i !== index))
  }

  return (
    <form onSubmit={handleSubmit}>
      {/* Textarea */}
      <div className="relative">
        <textarea
          value={text}
          onChange={e => setText(e.target.value)}
          rows={3}
          placeholder="Share what you know... paste a quote, link, or describe evidence you've found"
          className="w-full border border-slate-200 rounded-xl px-4 py-3 text-sm resize-none focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
        />
      </div>

      {/* Attached files */}
      {files.length > 0 && (
        <div className="flex flex-wrap gap-2 mt-2">
          {files.map((file, i) => (
            <div key={i} className="flex items-center gap-1 bg-slate-100 rounded-lg px-2 py-1 text-xs">
              <span className="truncate max-w-[120px]">{file.name}</span>
              <button
                type="button"
                onClick={() => removeFile(i)}
                className="text-slate-400 hover:text-red-500"
              >
                <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
          ))}
        </div>
      )}

      {/* Footer */}
      <div className="flex items-center justify-between mt-3">
        <div className="flex items-center gap-3">
          {/* File upload */}
          <input
            type="file"
            ref={fileInputRef}
            accept="image/*,.pdf,.doc,.docx"
            multiple
            className="hidden"
            onChange={handleFileChange}
          />
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            className="flex items-center gap-1 text-xs text-slate-500 hover:text-indigo-600 transition"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.172 7l-6.586 6.586a2 2 0 102.828 2.828l6.414-6.586a4 4 0 00-5.656-5.656l-6.415 6.585a6 6 0 108.486 8.486L20.5 13" />
            </svg>
            Attach
          </button>

          {/* Character count */}
          <span className={`text-xs ${isOverLimit ? 'text-red-500' : charsLeft < 50 ? 'text-amber-500' : 'text-slate-400'}`}>
            {charsLeft}
          </span>
        </div>

        <button
          type="submit"
          disabled={isSubmitting || !text.trim() || isOverLimit}
          className="px-4 py-2 bg-indigo-500 hover:bg-indigo-600 text-white rounded-lg text-sm font-medium transition disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isSubmitting ? 'Posting...' : 'Post'}
        </button>
      </div>
    </form>
  )
}

// =============================================================================
// Conclusion Panel - Big answer display with confidence
// =============================================================================
function ConclusionPanel({
  map,
  probability,
  schemaType,
  status,
  credibleInterval
}: {
  map: any
  probability: number
  schemaType?: string
  status?: string
  credibleInterval?: [number, number] | [string, string]
}) {
  const confidenceColor = probability > 0.8 ? 'text-green-600' : probability > 0.5 ? 'text-amber-500' : 'text-slate-400'
  const barColor = probability > 0.8 ? 'bg-green-500' : probability > 0.5 ? 'bg-amber-500' : 'bg-slate-400'
  const isResolved = status === 'resolved'

  return (
    <div className={`rounded-2xl p-6 ${isResolved ? 'bg-gradient-to-br from-green-50 to-white border-green-200' : 'bg-slate-50'} border`}>
      <div className="flex items-center gap-6">
        {/* Big Answer */}
        <div className="text-center min-w-[100px]">
          <div className={`text-4xl font-bold ${isResolved ? 'text-green-700' : 'text-slate-800'}`}>
            {map ?? '?'}
          </div>
          <div className="text-xs text-slate-500 mt-1">
            {isResolved ? 'Resolved' : 'Current Best'}
          </div>
        </div>

        {/* Confidence bar */}
        <div className="flex-1">
          <div className="flex justify-between text-sm mb-2">
            <span className="text-slate-600">Confidence</span>
            <span className={`font-bold ${confidenceColor}`}>
              {Math.round(probability * 100)}%
            </span>
          </div>
          <div className="h-3 bg-slate-200 rounded-full overflow-hidden">
            <div
              className={`h-full rounded-full transition-all duration-500 ${barColor}`}
              style={{ width: `${probability * 100}%` }}
            />
          </div>
          {credibleInterval && (
            <div className="flex justify-between text-xs text-slate-400 mt-2">
              <span>
                95% CI: [{credibleInterval[0]}, {credibleInterval[1]}]
              </span>
              <span>
                {schemaType === 'monotone_count' ? 'Count' :
                 schemaType === 'boolean' ? 'Yes/No' :
                 schemaType === 'forecast' ? 'Forecast' : 'Category'}
              </span>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

// =============================================================================
// Distribution Chart - Posterior probability bars
// =============================================================================
function DistributionChart({
  posteriorTop10,
  map
}: {
  posteriorTop10: Array<{ value: any; probability: number }>
  map?: any
}) {
  if (!posteriorTop10 || posteriorTop10.length === 0) return null

  const maxProb = Math.max(...posteriorTop10.map(p => p.probability))

  return (
    <div className="bg-white rounded-2xl p-5 shadow-sm border border-slate-100">
      <h3 className="font-medium text-slate-700 mb-4 text-sm">Probability Distribution</h3>
      <div className="flex items-end gap-1 h-24">
        {posteriorTop10.slice(0, 10).map((item, i) => (
          <div key={i} className="flex-1 flex flex-col items-center">
            <div className="text-xs text-slate-500 mb-1">
              {Math.round(item.probability * 100)}%
            </div>
            <div
              className={`w-full rounded-t transition-all duration-300 ${
                item.value === map ? 'bg-indigo-500' : 'bg-indigo-200'
              }`}
              style={{ height: `${(item.probability / maxProb) * 60}px` }}
            />
            <div className="text-xs font-mono text-slate-600 mt-1 truncate w-full text-center">
              {String(item.value).length > 5 ? String(item.value).slice(0, 5) + '...' : item.value}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

// =============================================================================
// Surfaces Panel - Claim clusters with identity relations
// =============================================================================
function SurfacesPanel({
  surfaces
}: {
  surfaces: Array<{
    id: string
    name: string
    claim_count: number
    sources?: string[]
    in_scope: boolean
    relations?: Array<{ type: string; target: string }>
  }>
}) {
  if (!surfaces || surfaces.length === 0) return null

  return (
    <div className="bg-white rounded-2xl p-5 shadow-sm border border-slate-100">
      <h3 className="font-medium text-slate-700 mb-4 text-sm flex items-center gap-2">
        <span>Claim Clusters</span>
        <span className="text-xs bg-slate-100 text-slate-500 px-2 py-0.5 rounded-full">
          {surfaces.length} surfaces
        </span>
      </h3>
      <div className="space-y-2">
        {surfaces.map(surface => (
          <div
            key={surface.id}
            className={`rounded-lg p-3 text-sm border ${
              surface.in_scope
                ? 'border-green-200 bg-green-50'
                : 'border-red-200 bg-red-50'
            }`}
          >
            <div className="flex items-center justify-between mb-1">
              <span className="font-medium text-slate-700">{surface.name}</span>
              <span className={`text-xs ${surface.in_scope ? 'text-green-600' : 'text-red-600'}`}>
                {surface.in_scope ? 'In scope' : 'Flagged'}
              </span>
            </div>
            <div className="text-xs text-slate-500">
              {surface.claim_count} claims
              {surface.sources && surface.sources.length > 0 && (
                <span> from {surface.sources.slice(0, 2).join(', ')}</span>
              )}
            </div>
            {surface.relations && surface.relations.length > 0 && (
              <div className="flex flex-wrap gap-1 mt-2">
                {surface.relations.map((rel, i) => (
                  <span
                    key={i}
                    className={`px-1.5 py-0.5 rounded text-xs ${
                      rel.type === 'CONFIRMS' ? 'bg-green-100 text-green-700' :
                      rel.type === 'SUPERSEDES' ? 'bg-blue-100 text-blue-700' :
                      'bg-red-100 text-red-700'
                    }`}
                  >
                    {rel.type} {rel.target}
                  </span>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}

// =============================================================================
// Bounty Panel - Stake display and add bounty
// =============================================================================
interface RecentReward {
  user: string
  amount: number
  reason: string
  time: string
}

function BountyBlock({
  totalBounty,
  distributedBounty = 0,
  recentRewards = [],
  onAddBounty,
  userCredits,
  isAuthenticated
}: {
  totalBounty: number
  distributedBounty?: number
  recentRewards?: RecentReward[]
  onAddBounty: (amount: number) => void
  userCredits: number
  isAuthenticated: boolean
}) {
  const [amount, setAmount] = useState('')
  const [showRewards, setShowRewards] = useState(false)

  const handleAdd = () => {
    const num = parseFloat(amount)
    if (num > 0) {
      onAddBounty(num)
      setAmount('')
    }
  }

  // Mock recent rewards for display
  const mockRewards: RecentReward[] = recentRewards.length > 0 ? recentRewards : [
    { user: 'Sarah Chen', amount: 12.50, reason: 'Primary source verification', time: '2h ago' },
    { user: 'Mike Torres', amount: 8.00, reason: 'Conflicting evidence resolution', time: '5h ago' },
    { user: 'Anonymous', amount: 5.25, reason: 'Attribution chain', time: '1d ago' },
  ]

  return (
    <div className="bg-gradient-to-br from-amber-50 to-white rounded-2xl p-5 border border-amber-200">
      {/* Header with totals */}
      <div className="flex items-center justify-between mb-3">
        <div>
          <h3 className="font-medium text-slate-700 text-sm">Bounty Pool</h3>
          <div className="text-2xl font-bold text-amber-600">
            ${totalBounty?.toFixed(2) || '0.00'}
          </div>
        </div>
        <div className="text-right">
          <div className="text-xs text-slate-500">Distributed</div>
          <div className="text-lg font-semibold text-green-600">
            ${distributedBounty?.toFixed(2) || '0.00'}
          </div>
        </div>
      </div>

      {/* Add bounty input */}
      <div className="flex gap-2 mb-3">
        <div className="flex-1 relative">
          <span className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400">$</span>
          <input
            type="number"
            value={amount}
            onChange={e => setAmount(e.target.value)}
            placeholder="Add to pool"
            min="0.01"
            step="0.01"
            className="w-full pl-7 pr-3 py-2 border border-slate-200 rounded-lg text-sm bg-white"
          />
        </div>
        <button
          onClick={handleAdd}
          disabled={!amount || parseFloat(amount) <= 0}
          className="px-4 py-2 bg-amber-500 hover:bg-amber-600 text-white rounded-lg text-sm font-medium transition disabled:opacity-50"
        >
          Add
        </button>
      </div>

      {/* Recent Rewards Toggle */}
      <button
        onClick={() => setShowRewards(!showRewards)}
        className="w-full flex items-center justify-between text-xs text-slate-500 hover:text-slate-700 py-2 border-t border-amber-100"
      >
        <span>Recent rewards</span>
        <svg
          className={`w-4 h-4 transition-transform ${showRewards ? 'rotate-180' : ''}`}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {/* Recent Rewards List */}
      {showRewards && (
        <div className="space-y-2 pt-2">
          {mockRewards.map((reward, i) => (
            <div key={i} className="flex items-center gap-2 text-xs">
              <div className="w-6 h-6 rounded-full bg-green-100 flex items-center justify-center text-green-600 font-medium">
                {reward.user.charAt(0)}
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-1">
                  <span className="font-medium text-slate-700 truncate">{reward.user}</span>
                  <span className="text-green-600 font-medium">+${reward.amount.toFixed(2)}</span>
                </div>
                <div className="text-slate-400 truncate">{reward.reason}</div>
              </div>
              <div className="text-slate-400 whitespace-nowrap">{reward.time}</div>
            </div>
          ))}
        </div>
      )}

      {isAuthenticated && userCredits > 0 && (
        <div className="text-xs text-slate-400 mt-2 pt-2 border-t border-amber-100">
          Your credits: ${userCredits.toFixed(2)}
        </div>
      )}
    </div>
  )
}

// =============================================================================
// Tasks Panel - Open tasks with claim buttons
// =============================================================================
function TasksPanel({
  tasks,
  onClaimTask
}: {
  tasks: Task[]
  onClaimTask: (task: Task) => void
}) {
  if (!tasks || tasks.length === 0) {
    return (
      <div className="bg-white rounded-2xl p-5 shadow-sm border border-slate-100">
        <h3 className="font-medium text-slate-700 text-sm mb-3">Tasks</h3>
        <div className="text-sm text-slate-400 text-center py-4">
          No open tasks
        </div>
      </div>
    )
  }

  const openTasks = tasks.filter(t => !t.completed)
  const completedTasks = tasks.filter(t => t.completed)

  return (
    <div className="bg-white rounded-2xl p-5 shadow-sm border border-slate-100">
      <h3 className="font-medium text-slate-700 text-sm mb-3 flex items-center gap-2">
        <span>Tasks</span>
        {openTasks.length > 0 && (
          <span className="text-xs bg-amber-100 text-amber-700 px-2 py-0.5 rounded-full">
            {openTasks.length} open
          </span>
        )}
      </h3>
      <div className="space-y-2">
        {openTasks.map(task => (
          <div
            key={task.id}
            className="rounded-lg p-3 bg-amber-50 border border-amber-200"
          >
            <div className="flex items-start justify-between">
              <div>
                <div className="flex items-center gap-2 text-xs">
                  <span className="text-amber-600 font-medium">{task.type}</span>
                </div>
                <p className="text-xs text-slate-600 mt-1">{task.description}</p>
              </div>
              <span className="text-amber-700 font-medium text-sm whitespace-nowrap">
                ${task.bounty?.toFixed(2) || '0.00'}
              </span>
            </div>
            <button
              onClick={() => onClaimTask(task)}
              className="mt-2 w-full bg-amber-100 hover:bg-amber-200 text-amber-700 text-xs font-medium py-1.5 rounded transition"
            >
              Claim Task
            </button>
          </div>
        ))}
        {completedTasks.map(task => (
          <div
            key={task.id}
            className="rounded-lg p-3 bg-green-50 border border-green-200 opacity-60"
          >
            <div className="flex items-center gap-2 text-xs text-green-700">
              <span>Completed: {task.type}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

// =============================================================================
// Avatar Component - Generate from name/initials
// =============================================================================
function Avatar({ name, size = 'md' }: { name: string; size?: 'sm' | 'md' | 'lg' }) {
  const initials = name
    .split(' ')
    .map(n => n[0])
    .join('')
    .slice(0, 2)
    .toUpperCase()

  // Generate consistent color from name
  const colors = [
    'bg-blue-500', 'bg-green-500', 'bg-purple-500', 'bg-pink-500',
    'bg-indigo-500', 'bg-teal-500', 'bg-orange-500', 'bg-cyan-500'
  ]
  const colorIndex = name.split('').reduce((acc, c) => acc + c.charCodeAt(0), 0) % colors.length

  const sizeClasses = {
    sm: 'w-6 h-6 text-xs',
    md: 'w-9 h-9 text-sm',
    lg: 'w-12 h-12 text-base'
  }

  return (
    <div className={`${sizeClasses[size]} ${colors[colorIndex]} rounded-full flex items-center justify-center text-white font-medium`}>
      {initials || '?'}
    </div>
  )
}

// =============================================================================
// Relative Time Helper
// =============================================================================
function timeAgo(dateStr: string): string {
  const date = new Date(dateStr)
  const now = new Date()
  const seconds = Math.floor((now.getTime() - date.getTime()) / 1000)

  if (seconds < 60) return 'just now'
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m`
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h`
  if (seconds < 604800) return `${Math.floor(seconds / 86400)}d`
  return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
}

// =============================================================================
// Contribution Card - Social media style
// =============================================================================
function ContributionCard({ contrib }: { contrib: Contribution }) {
  const userName = contrib.user_name || contrib.source || 'Anonymous'
  const commentCount = Math.floor(Math.random() * 5) // Placeholder - would come from API

  const typeConfig = {
    evidence: { bg: 'bg-indigo-50', border: 'border-indigo-200', badge: 'bg-indigo-100 text-indigo-700', icon: '📊' },
    attribution: { bg: 'bg-purple-50', border: 'border-purple-200', badge: 'bg-purple-100 text-purple-700', icon: '🔗' },
    scope_correction: { bg: 'bg-red-50', border: 'border-red-200', badge: 'bg-red-100 text-red-700', icon: '⚠️' },
    refutation: { bg: 'bg-amber-50', border: 'border-amber-200', badge: 'bg-amber-100 text-amber-700', icon: '❌' },
    disambiguation: { bg: 'bg-cyan-50', border: 'border-cyan-200', badge: 'bg-cyan-100 text-cyan-700', icon: '🔍' },
  }
  const config = typeConfig[contrib.type as keyof typeof typeConfig] || typeConfig.evidence

  return (
    <div className={`rounded-xl p-4 ${config.bg} border ${config.border} transition hover:shadow-md`}>
      {/* Header: Avatar + Name + Time */}
      <div className="flex items-start gap-3">
        <Avatar name={userName} size="md" />
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="font-medium text-slate-800 text-sm">{userName}</span>
            <span className={`text-xs px-2 py-0.5 rounded-full ${config.badge}`}>
              {config.icon} {contrib.type?.replace('_', ' ')}
            </span>
            {contrib.impact && contrib.impact > 0.02 && (
              <span className="text-xs font-medium text-green-600 bg-green-100 px-2 py-0.5 rounded-full">
                +{Math.round(contrib.impact * 100)}% impact
              </span>
            )}
          </div>
          <div className="text-xs text-slate-400 mt-0.5">
            {contrib.created_at ? timeAgo(contrib.created_at) : 'recently'}
            {contrib.source_name && <span> · via {contrib.source_name}</span>}
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="mt-3 text-sm text-slate-700 leading-relaxed">
        {contrib.text}
      </div>

      {/* Extracted Value Badge */}
      {contrib.extracted_value !== undefined && (
        <div className="mt-3 inline-flex items-center gap-2 bg-white/70 rounded-lg px-3 py-1.5 border border-slate-200">
          <span className="text-xs text-slate-500">Extracted:</span>
          <span className="font-mono font-medium text-slate-800">{contrib.extracted_value}</span>
          {contrib.observation_kind && (
            <span className="text-xs text-slate-400">
              ({contrib.observation_kind === 'point' ? '=' : contrib.observation_kind === 'lower_bound' ? '≥' : '~'})
            </span>
          )}
        </div>
      )}

      {/* Source Link */}
      {contrib.source_url && (
        <a
          href={contrib.source_url}
          target="_blank"
          rel="noopener noreferrer"
          className="mt-3 block text-xs text-indigo-600 hover:text-indigo-800 truncate"
        >
          {contrib.source_url}
        </a>
      )}

      {/* Footer: Actions */}
      <div className="mt-3 pt-3 border-t border-slate-200/50 flex items-center gap-4">
        <button className="flex items-center gap-1.5 text-xs text-slate-500 hover:text-indigo-600 transition">
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
          </svg>
          <span>{commentCount}</span>
        </button>
        <button className="flex items-center gap-1.5 text-xs text-slate-500 hover:text-green-600 transition">
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M5 15l7-7 7 7" />
          </svg>
          <span>Verify</span>
        </button>
        <button className="flex items-center gap-1.5 text-xs text-slate-500 hover:text-amber-600 transition">
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <span>Flag</span>
        </button>
      </div>
    </div>
  )
}

// =============================================================================
// Community Section - ShareBox + Contributions Feed
// =============================================================================
function CommunitySection({
  contributions,
  onSubmit,
  isSubmitting
}: {
  contributions: Contribution[]
  onSubmit: (input: ContributionInput) => Promise<void>
  isSubmitting: boolean
}) {
  const [sortBy, setSortBy] = useState<'recent' | 'impact'>('recent')

  const sortedContributions = [...contributions].sort((a, b) => {
    if (sortBy === 'impact') {
      return (b.impact || 0) - (a.impact || 0)
    }
    return new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
  })

  return (
    <div className="bg-white rounded-2xl p-5 shadow-sm border border-slate-100">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <h3 className="font-medium text-slate-700 flex items-center gap-2">
          <span>Community</span>
          <span className="text-xs bg-slate-100 text-slate-500 px-2 py-0.5 rounded-full">
            {contributions.length}
          </span>
        </h3>
        <div className="flex gap-1 bg-slate-100 rounded-lg p-0.5">
          <button
            onClick={() => setSortBy('recent')}
            className={`px-3 py-1 rounded text-xs font-medium transition ${
              sortBy === 'recent' ? 'bg-white shadow text-slate-800' : 'text-slate-500 hover:text-slate-700'
            }`}
          >
            Recent
          </button>
          <button
            onClick={() => setSortBy('impact')}
            className={`px-3 py-1 rounded text-xs font-medium transition ${
              sortBy === 'impact' ? 'bg-white shadow text-slate-800' : 'text-slate-500 hover:text-slate-700'
            }`}
          >
            Impact
          </button>
        </div>
      </div>

      {/* Share Box */}
      <div className="mb-4 pb-4 border-b border-slate-100">
        <SimpleShareBox onSubmit={onSubmit} isSubmitting={isSubmitting} />
      </div>

      {/* Contributions Feed */}
      {contributions.length === 0 ? (
        <div className="text-center py-6">
          <div className="text-3xl mb-2">💬</div>
          <p className="text-slate-500 text-sm">No contributions yet</p>
          <p className="text-slate-400 text-xs mt-1">Be the first to share evidence!</p>
        </div>
      ) : (
        <div className="space-y-3">
          {sortedContributions.map((contrib, i) => (
            <ContributionCard key={contrib.id || i} contrib={contrib} />
          ))}
        </div>
      )}
    </div>
  )
}

// =============================================================================
// Main InquiryDetailPage Component
// =============================================================================
function InquiryDetailPage() {
  const { inquiryId } = useParams<{ inquiryId: string }>()
  const navigate = useNavigate()

  // Data state
  const [inquiry, setInquiry] = useState<InquiryDetail | null>(null)
  const [trace, setTrace] = useState<InquiryTrace | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // UI state
  const [toast, setToast] = useState<string | null>(null)
  const [isSubmitting, setIsSubmitting] = useState(false)

  // User state
  const [user, setUser] = useState<{ credits: number } | null>(null)
  const [isAuthenticated, setIsAuthenticated] = useState(false)

  // Load user auth status
  useEffect(() => {
    fetch('/api/auth/status')
      .then(res => res.json())
      .then(data => {
        setIsAuthenticated(data.authenticated)
        if (data.user) setUser({ credits: data.user.credits || 0 })
      })
      .catch(() => {})
  }, [])

  // Load inquiry data
  useEffect(() => {
    if (inquiryId) {
      loadInquiry(inquiryId)
    }
  }, [inquiryId])

  const loadInquiry = async (id: string) => {
    setLoading(true)
    setError(null)

    try {
      // Try real API first
      const [inqRes, traceRes] = await Promise.all([
        fetch(`/api/inquiry/${id}`),
        fetch(`/api/inquiry/${id}/trace`)
      ])

      if (inqRes.ok && traceRes.ok) {
        const inqData = await inqRes.json()
        const traceData = await traceRes.json()
        setInquiry(inqData)
        setTrace(traceData)
      } else {
        // Fall back to simulated data
        loadSimulatedData(id)
      }
    } catch (e) {
      // Fall back to simulated data
      loadSimulatedData(id)
    } finally {
      setLoading(false)
    }
  }

  const loadSimulatedData = (id: string) => {
    const found = SIMULATED_INQUIRIES.find(i => i.id === id)
    if (!found) {
      setError('Inquiry not found')
      return
    }

    const simulatedTrace = generateSimulatedTrace(found) as InquiryTrace
    setInquiry(found as InquiryDetail)
    setTrace(simulatedTrace)
  }

  const showToast = (message: string) => {
    setToast(message)
    setTimeout(() => setToast(null), 3000)
  }

  const handleAddBounty = async (amount: number) => {
    if (!inquiryId) return

    try {
      const res = await fetch(`/api/inquiry/${inquiryId}/stake`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ amount })
      })

      if (res.ok) {
        showToast(`Added $${amount.toFixed(2)} bounty!`)
        loadInquiry(inquiryId)
      } else {
        showToast('Demo mode - bounty logged')
      }
    } catch (e) {
      showToast('Demo mode - bounty logged')
    }
  }

  const handleSubmitContribution = async (input: ContributionInput) => {
    if (!inquiryId) return
    setIsSubmitting(true)

    try {
      const res = await fetch(`/api/inquiry/${inquiryId}/contribute`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(input)
      })

      if (res.ok) {
        const result = await res.json()
        const impact = Math.round((result.contribution?.posterior_impact || 0) * 100)
        showToast(impact > 5 ? `Evidence added! +${impact}% impact` : 'Evidence added!')
        loadInquiry(inquiryId)
      } else {
        showToast('Demo mode - contribution logged')
      }
    } catch (e) {
      showToast('Demo mode - contribution logged')
    } finally {
      setIsSubmitting(false)
    }
  }

  const handleClaimTask = (task: Task) => {
    showToast(`Task claimed: ${task.type}`)
  }

  // Loading state
  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[50vh]">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600 mx-auto mb-4" />
          <p className="text-slate-500">Loading inquiry...</p>
        </div>
      </div>
    )
  }

  // Error state
  if (error || !inquiry || !trace) {
    return (
      <div className="text-center py-16">
        <div className="text-6xl mb-4">?</div>
        <h2 className="text-xl font-semibold text-slate-700 mb-2">
          {error || 'Inquiry not found'}
        </h2>
        <button
          onClick={() => navigate('/inquiry')}
          className="text-indigo-500 hover:text-indigo-700"
        >
          Back to inquiries
        </button>
      </div>
    )
  }

  const beliefState = trace.belief_state

  return (
    <div className="max-w-6xl mx-auto px-4 py-6">
      {/* Header */}
      <div className="mb-6">
        <button
          onClick={() => navigate('/inquiry')}
          className="flex items-center gap-2 text-slate-500 hover:text-slate-700 mb-4 transition text-sm"
        >
          Back to questions
        </button>

        <div className="flex items-start justify-between">
          <div className="flex-1">
            <div className="flex items-center gap-2 mb-2">
              <span
                className={`px-2 py-0.5 rounded-full text-xs font-medium ${
                  inquiry.status === 'resolved'
                    ? 'bg-green-100 text-green-700'
                    : inquiry.status === 'open'
                    ? 'bg-indigo-100 text-indigo-700'
                    : 'bg-slate-100 text-slate-600'
                }`}
              >
                {inquiry.status?.toUpperCase()}
              </span>
              <span className="text-xs text-slate-400">Rigor {inquiry.rigor}</span>
              {inquiry.schema_type === 'forecast' && (
                <span className="text-xs bg-purple-100 text-purple-700 px-2 py-0.5 rounded-full">
                  Forecast
                </span>
              )}
            </div>
            <h1 className="text-xl font-semibold text-slate-800 mb-2">
              {inquiry.title}
            </h1>
            {inquiry.scope_entities && inquiry.scope_entities.length > 0 && (
              <div className="flex flex-wrap gap-1">
                {inquiry.scope_entities.map((entity, i) => (
                  <span
                    key={i}
                    className="text-xs bg-purple-100 text-purple-700 px-2 py-0.5 rounded-full"
                  >
                    {entity}
                  </span>
                ))}
              </div>
            )}
          </div>

          {/* Resolution policy */}
          <div className="bg-slate-50 rounded-lg p-3 text-xs text-slate-600 max-w-xs ml-4">
            <strong>Resolution:</strong> P(MAP) ≥ 95% for 24h + no blocking tasks
          </div>
        </div>
      </div>

      {/* Two-Pane Layout - 60/40 split */}
      <div className="grid lg:grid-cols-5 gap-6">
        {/* Left Pane: Analysis (60%) */}
        <div className="lg:col-span-3 space-y-4">
          {/* Context - Why this question matters */}
          <div className="bg-gradient-to-r from-slate-50 to-white rounded-2xl p-5 border border-slate-200">
            <p className="text-sm text-slate-600 leading-relaxed">
              {inquiry.description || `This question tracks ${inquiry.schema_type === 'monotone_count' ? 'a count that can only increase over time' : inquiry.schema_type === 'boolean' ? 'a yes/no outcome' : 'evolving information'}. Numbers from different sources often conflict—our system weighs evidence to find the most likely truth.`}
            </p>
          </div>

          {/* Current Best Answer Card */}
          <div className={`rounded-2xl p-6 ${inquiry.status === 'resolved' ? 'bg-gradient-to-br from-green-50 to-white border-green-200' : 'bg-white border-slate-200'} border shadow-sm`}>
            <div className="flex items-start justify-between mb-4">
              <div>
                <div className="text-xs text-slate-500 mb-1">
                  {inquiry.status === 'resolved' ? 'Resolved Answer' : 'Current Best Estimate'}
                </div>
                <div className={`text-4xl font-bold ${inquiry.status === 'resolved' ? 'text-green-700' : 'text-slate-800'}`}>
                  {beliefState?.map ?? '?'}
                </div>
              </div>
              <div className="flex gap-2">
                <button className="p-2 text-slate-400 hover:text-indigo-600 hover:bg-indigo-50 rounded-lg transition" title="Share">
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8.684 13.342C8.886 12.938 9 12.482 9 12c0-.482-.114-.938-.316-1.342m0 2.684a3 3 0 110-2.684m0 2.684l6.632 3.316m-6.632-6l6.632-3.316m0 0a3 3 0 105.367-2.684 3 3 0 00-5.367 2.684zm0 9.316a3 3 0 105.368 2.684 3 3 0 00-5.368-2.684z" />
                  </svg>
                </button>
                <button className="p-2 text-slate-400 hover:text-indigo-600 hover:bg-indigo-50 rounded-lg transition" title="Embed">
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
                  </svg>
                </button>
              </div>
            </div>

            {/* Confidence bar */}
            <div className="mb-4">
              <div className="flex justify-between text-sm mb-1">
                <span className="text-slate-500">Confidence</span>
                <span className={`font-semibold ${(beliefState?.map_probability || 0) > 0.8 ? 'text-green-600' : (beliefState?.map_probability || 0) > 0.5 ? 'text-amber-500' : 'text-slate-400'}`}>
                  {Math.round((beliefState?.map_probability || 0) * 100)}%
                </span>
              </div>
              <div className="h-2 bg-slate-100 rounded-full overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all duration-500 ${(beliefState?.map_probability || 0) > 0.8 ? 'bg-green-500' : (beliefState?.map_probability || 0) > 0.5 ? 'bg-amber-500' : 'bg-slate-300'}`}
                  style={{ width: `${(beliefState?.map_probability || 0) * 100}%` }}
                />
              </div>
            </div>

            {/* Stats row */}
            <div className="grid grid-cols-3 gap-4 text-center pt-4 border-t border-slate-100">
              <div>
                <div className="text-lg font-semibold text-slate-700">{beliefState?.observation_count || 0}</div>
                <div className="text-xs text-slate-400">Sources</div>
              </div>
              <div>
                <div className="text-lg font-semibold text-slate-700">{(beliefState?.entropy_bits || 0).toFixed(1)}</div>
                <div className="text-xs text-slate-400">Entropy</div>
              </div>
              <div>
                <div className="text-lg font-semibold text-slate-700">{inquiry.credible_interval ? `${inquiry.credible_interval[0]}-${inquiry.credible_interval[1]}` : '—'}</div>
                <div className="text-xs text-slate-400">95% CI</div>
              </div>
            </div>
          </div>

          {/* Resolution Status */}
          <div className="bg-white rounded-2xl p-4 shadow-sm border border-slate-100">
            <div className="flex items-center gap-3">
              <div className={`w-3 h-3 rounded-full ${inquiry.status === 'resolved' ? 'bg-green-500' : (beliefState?.map_probability || 0) >= 0.95 ? 'bg-amber-500 animate-pulse' : 'bg-slate-300'}`} />
              <div className="flex-1">
                <div className="text-sm font-medium text-slate-700">
                  {inquiry.status === 'resolved'
                    ? 'Resolved'
                    : (beliefState?.map_probability || 0) >= 0.95
                      ? 'Approaching Resolution'
                      : 'Gathering Evidence'}
                </div>
                <div className="text-xs text-slate-400">
                  {inquiry.status === 'resolved'
                    ? 'Confidence held ≥95% for 24 hours'
                    : (beliefState?.map_probability || 0) >= 0.95
                      ? `${Math.round((beliefState?.map_probability || 0) * 100)}% confidence - needs 24h stability`
                      : `Need ${Math.round((0.95 - (beliefState?.map_probability || 0)) * 100)}% more confidence for resolution`}
                </div>
              </div>
            </div>
          </div>

          {/* Distribution */}
          <DistributionChart
            posteriorTop10={trace.posterior_top_10 || []}
            map={beliefState?.map}
          />

          {/* Gaps (formerly Tasks) - System generated */}
          <div className="bg-white rounded-2xl p-5 shadow-sm border border-slate-100">
            <h3 className="font-medium text-slate-700 text-sm mb-3 flex items-center gap-2">
              <span>Evidence Gaps</span>
              {(trace.tasks || []).filter(t => !t.completed).length > 0 && (
                <span className="text-xs bg-amber-100 text-amber-700 px-2 py-0.5 rounded-full">
                  {(trace.tasks || []).filter(t => !t.completed).length} open
                </span>
              )}
            </h3>
            {(trace.tasks || []).length === 0 ? (
              <p className="text-sm text-slate-400">No gaps identified yet</p>
            ) : (
              <div className="space-y-2">
                {(trace.tasks || []).filter(t => !t.completed).map(task => (
                  <div key={task.id} className="flex items-center gap-3 p-3 bg-amber-50 rounded-lg border border-amber-200">
                    <div className="w-2 h-2 rounded-full bg-amber-400" />
                    <div className="flex-1">
                      <div className="text-sm text-slate-700">{task.description}</div>
                      <div className="text-xs text-slate-400">{task.type?.replace(/_/g, ' ')}</div>
                    </div>
                    <div className="text-sm font-medium text-amber-600">${task.bounty?.toFixed(0) || 0}</div>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Claims Topology - Simplified */}
          <SurfacesPanel surfaces={trace.surfaces || []} />
        </div>

        {/* Right Pane: Community (40%) */}
        <div className="lg:col-span-2 space-y-4">
          {/* Bounty Block */}
          <BountyBlock
            totalBounty={inquiry.stake || 0}
            distributedBounty={25.75}
            onAddBounty={handleAddBounty}
            userCredits={user?.credits || 0}
            isAuthenticated={isAuthenticated}
          />

          {/* Community Section (includes ShareBox + Feed) */}
          <CommunitySection
            contributions={trace.contributions || []}
            onSubmit={handleSubmitContribution}
            isSubmitting={isSubmitting}
          />
        </div>
      </div>

      {/* Toast */}
      {toast && (
        <div className="fixed bottom-6 left-1/2 transform -translate-x-1/2 bg-slate-800 text-white px-6 py-3 rounded-full shadow-lg z-50">
          {toast}
        </div>
      )}
    </div>
  )
}

export default InquiryDetailPage
