import React, { useState } from 'react'

interface ChatUnlockModalProps {
  storyTitle: string
  userCredits: number
  onUnlock: () => Promise<void>
  onCancel: () => void
}

const UNLOCK_COST = 10

function ChatUnlockModal({ storyTitle, userCredits, onUnlock, onCancel }: ChatUnlockModalProps) {
  const [isUnlocking, setIsUnlocking] = useState(false)
  const canAfford = userCredits >= UNLOCK_COST

  const handleUnlock = async () => {
    if (!canAfford || isUnlocking) return

    setIsUnlocking(true)
    try {
      await onUnlock()
    } catch (error) {
      console.error('Unlock failed:', error)
      setIsUnlocking(false)
    }
  }

  return (
    <>
      {/* Overlay */}
      <div className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4">
        {/* Modal */}
        <div className="bg-white rounded-2xl shadow-2xl max-w-md w-full p-8 relative">
          {/* Close button */}
          <button
            onClick={onCancel}
            className="absolute top-4 right-4 text-slate-400 hover:text-slate-600 transition"
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>

          {/* Icon */}
          <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-gradient-to-br from-indigo-500 to-purple-500 flex items-center justify-center">
            <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
            </svg>
          </div>

          {/* Title */}
          <h2 className="text-2xl font-bold text-center text-slate-900 mb-2">
            Unlock AI Chat
          </h2>

          {/* Description */}
          <p className="text-center text-slate-600 mb-6">
            Get AI-powered insights and answers about:
          </p>

          {/* Story title */}
          <div className="bg-slate-50 rounded-lg p-4 mb-6">
            <p className="text-sm font-medium text-slate-900 line-clamp-2">
              {storyTitle}
            </p>
          </div>

          {/* Features */}
          <div className="space-y-3 mb-6">
            <div className="flex items-center gap-3 text-sm text-slate-700">
              <svg className="w-5 h-5 text-green-500 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
              </svg>
              <span>Up to 100 messages per story</span>
            </div>
            <div className="flex items-center gap-3 text-sm text-slate-700">
              <svg className="w-5 h-5 text-green-500 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
              </svg>
              <span>Access to verified claims & entities</span>
            </div>
            <div className="flex items-center gap-3 text-sm text-slate-700">
              <svg className="w-5 h-5 text-green-500 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
              </svg>
              <span>Powered by GPT-4</span>
            </div>
          </div>

          {/* Price and credits */}
          <div className="flex items-center justify-between bg-gradient-to-r from-amber-50 to-yellow-50 border border-amber-200 rounded-lg p-4 mb-6">
            <div>
              <div className="text-sm text-slate-600">Unlock Cost</div>
              <div className="text-2xl font-bold text-amber-700 flex items-center gap-1">
                <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 20 20">
                  <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
                </svg>
                {UNLOCK_COST}
              </div>
            </div>
            <div>
              <div className="text-sm text-slate-600 text-right">Your Balance</div>
              <div className={`text-2xl font-bold text-right ${canAfford ? 'text-green-600' : 'text-red-600'}`}>
                {userCredits}★
              </div>
            </div>
          </div>

          {/* Insufficient credits warning */}
          {!canAfford && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-3 mb-6">
              <p className="text-sm text-red-800 text-center">
                Insufficient credits. You need {UNLOCK_COST - userCredits} more credits.
              </p>
            </div>
          )}

          {/* Action buttons */}
          <div className="flex gap-3">
            <button
              onClick={onCancel}
              className="flex-1 px-6 py-3 border border-slate-300 rounded-lg font-medium text-slate-700 hover:bg-slate-50 transition"
            >
              Cancel
            </button>
            <button
              onClick={handleUnlock}
              disabled={!canAfford || isUnlocking}
              className="flex-1 px-6 py-3 bg-gradient-to-r from-indigo-600 to-purple-600 text-white rounded-lg font-medium hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed transition"
            >
              {isUnlocking ? 'Unlocking...' : `Unlock for ${UNLOCK_COST}★`}
            </button>
          </div>
        </div>
      </div>
    </>
  )
}

export default ChatUnlockModal
