import React, { useState } from 'react'
import { InquiryTask } from '../../types/inquiry'

interface BountyPanelProps {
  totalBounty: number
  tasks: InquiryTask[]
  onAddBounty: (amount: number) => Promise<void>
  onClaimTask?: (task: InquiryTask) => void
  userCredits?: number
  isAuthenticated?: boolean
}

const BOUNTY_AMOUNTS = [1, 5, 10]

function BountyPanel({
  totalBounty,
  tasks,
  onAddBounty,
  onClaimTask,
  userCredits = 0,
  isAuthenticated = false
}: BountyPanelProps) {
  const [adding, setAdding] = useState(false)
  const [selectedAmount, setSelectedAmount] = useState<number | null>(null)

  const handleAddBounty = async (amount: number) => {
    if (!isAuthenticated) {
      window.location.href = '/api/auth/login'
      return
    }

    setSelectedAmount(amount)
    setAdding(true)
    try {
      await onAddBounty(amount)
    } finally {
      setAdding(false)
      setSelectedAmount(null)
    }
  }

  const openTasks = tasks.filter(t => !t.completed)
  const completedTasks = tasks.filter(t => t.completed)

  return (
    <div className="space-y-4">
      {/* Bounty Total & Add */}
      <div className="bg-gradient-to-br from-amber-50 to-yellow-50 rounded-2xl p-5 border border-amber-200">
        <div className="flex items-center justify-between mb-4">
          <div>
            <div className="text-xs text-amber-600 uppercase tracking-wide">Total Bounty</div>
            <div className="text-3xl font-bold text-amber-700">${totalBounty.toFixed(2)}</div>
          </div>
          <div className="text-4xl">💰</div>
        </div>

        {/* Add Bounty Buttons */}
        <div className="flex gap-2">
          {BOUNTY_AMOUNTS.map(amount => (
            <button
              key={amount}
              onClick={() => handleAddBounty(amount)}
              disabled={adding || (isAuthenticated && userCredits < amount)}
              className={`
                flex-1 py-2 px-3 rounded-lg font-medium text-sm transition
                ${adding && selectedAmount === amount
                  ? 'bg-amber-300 text-amber-800'
                  : 'bg-amber-100 hover:bg-amber-200 text-amber-700'
                }
                disabled:opacity-50 disabled:cursor-not-allowed
              `}
            >
              {adding && selectedAmount === amount ? (
                <span className="animate-pulse">Adding...</span>
              ) : (
                `+$${amount}`
              )}
            </button>
          ))}
        </div>

        {/* User credits or login prompt */}
        {isAuthenticated ? (
          <div className="mt-2 text-xs text-amber-600 text-center">
            Your balance: ${userCredits.toFixed(2)}
          </div>
        ) : (
          <div className="mt-2 text-xs text-amber-600 text-center">
            <a href="/api/auth/login" className="underline hover:no-underline">
              Sign in to add bounty
            </a>
          </div>
        )}
      </div>

      {/* Open Tasks */}
      {openTasks.length > 0 && (
        <div className="bg-white rounded-2xl p-5 shadow-sm border border-slate-100">
          <h3 className="font-semibold text-slate-800 mb-3 flex items-center gap-2">
            <span>⚡</span> Earn Bounty
            <span className="text-xs bg-amber-100 text-amber-600 px-2 py-0.5 rounded-full">
              {openTasks.length} tasks
            </span>
          </h3>
          <div className="space-y-2">
            {openTasks.map(task => (
              <TaskCard
                key={task.id}
                task={task}
                onClaim={() => onClaimTask?.(task)}
                isAuthenticated={isAuthenticated}
              />
            ))}
          </div>
        </div>
      )}

      {/* Completed Tasks */}
      {completedTasks.length > 0 && (
        <div className="bg-white rounded-2xl p-5 shadow-sm border border-slate-100">
          <h3 className="font-semibold text-slate-800 mb-3 flex items-center gap-2">
            <span>✅</span> Completed Tasks
          </h3>
          <div className="space-y-2 opacity-60">
            {completedTasks.map(task => (
              <TaskCard key={task.id} task={task} completed />
            ))}
          </div>
        </div>
      )}

      {/* No tasks */}
      {tasks.length === 0 && (
        <div className="bg-white rounded-2xl p-5 shadow-sm border border-slate-100">
          <h3 className="font-semibold text-slate-800 mb-3 flex items-center gap-2">
            <span>⚡</span> Tasks
          </h3>
          <div className="text-center py-4 text-slate-400">
            <div className="text-2xl mb-2">✨</div>
            <p className="text-sm">No tasks needed right now</p>
            <p className="text-xs mt-1">Tasks appear when REEE detects issues</p>
          </div>
        </div>
      )}
    </div>
  )
}

interface TaskCardProps {
  task: InquiryTask
  onClaim?: () => void
  isAuthenticated?: boolean
  completed?: boolean
}

function TaskCard({ task, onClaim, isAuthenticated, completed }: TaskCardProps) {
  const getTaskIcon = (type: string) => {
    switch (type) {
      case 'need_primary_source': return '📋'
      case 'unresolved_conflict': return '⚔️'
      case 'single_source_only': return '1️⃣'
      case 'high_entropy': return '🎲'
      case 'stale': return '⏰'
      default: return '📌'
    }
  }

  const getTaskLabel = (type: string) => {
    switch (type) {
      case 'need_primary_source': return 'Primary Source Needed'
      case 'unresolved_conflict': return 'Conflict Resolution'
      case 'single_source_only': return 'Corroboration Needed'
      case 'high_entropy': return 'Reduce Uncertainty'
      case 'stale': return 'Update Needed'
      default: return type
    }
  }

  return (
    <div className={`
      rounded-lg p-3 border transition
      ${completed
        ? 'bg-green-50 border-green-200'
        : 'bg-amber-50 border-amber-200 hover:border-amber-300'
      }
    `}>
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <div className="flex items-center gap-2 mb-1">
            <span>{getTaskIcon(task.type)}</span>
            <span className="text-xs font-medium text-slate-700">
              {getTaskLabel(task.type)}
            </span>
          </div>
          <p className="text-xs text-slate-600">{task.description}</p>
        </div>
        <div className="text-right">
          <div className="font-bold text-amber-600">${task.bounty.toFixed(2)}</div>
          {completed && (
            <div className="text-xs text-green-600">Completed</div>
          )}
        </div>
      </div>

      {!completed && onClaim && (
        <button
          onClick={onClaim}
          disabled={!isAuthenticated}
          className={`
            mt-2 w-full text-xs font-medium py-1.5 rounded transition
            ${isAuthenticated
              ? 'bg-amber-100 hover:bg-amber-200 text-amber-700'
              : 'bg-slate-100 text-slate-400 cursor-not-allowed'
            }
          `}
        >
          {isAuthenticated ? 'Claim Task' : 'Sign in to claim'}
        </button>
      )}
    </div>
  )
}

export default BountyPanel
