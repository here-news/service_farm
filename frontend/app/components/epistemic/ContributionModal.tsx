import React, { useState } from 'react';
import { EpistemicGap } from './EpistemicStateCard';

interface ContributionModalProps {
  isOpen: boolean;
  onClose: () => void;
  eventId: string;
  eventName: string;
  selectedQuest?: EpistemicGap | null;
  onSubmit?: (contribution: ContributionSubmission) => Promise<void>;
}

export interface ContributionSubmission {
  url: string;
  eventId: string;
  questType?: string;
  note?: string;
}

const ContributionModal: React.FC<ContributionModalProps> = ({
  isOpen,
  onClose,
  eventId,
  eventName,
  selectedQuest,
  onSubmit
}) => {
  const [url, setUrl] = useState('');
  const [note, setNote] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);

  if (!isOpen) return null;

  const validateUrl = (url: string): boolean => {
    try {
      new URL(url);
      return true;
    } catch {
      return false;
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);

    // Validate URL
    if (!url.trim()) {
      setError('Please enter a URL');
      return;
    }

    if (!validateUrl(url.trim())) {
      setError('Please enter a valid URL (e.g., https://example.com/article)');
      return;
    }

    setIsSubmitting(true);

    try {
      const contribution: ContributionSubmission = {
        url: url.trim(),
        eventId,
        questType: selectedQuest?.type,
        note: note.trim() || undefined
      };

      if (onSubmit) {
        await onSubmit(contribution);
      } else {
        // Default API call
        const response = await fetch('/api/contributions/submit', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(contribution)
        });

        if (!response.ok) {
          const data = await response.json();
          throw new Error(data.detail || 'Failed to submit contribution');
        }
      }

      setSuccess(true);
      setTimeout(() => {
        onClose();
        setUrl('');
        setNote('');
        setSuccess(false);
      }, 2000);
    } catch (err: any) {
      setError(err.message || 'Failed to submit. Please try again.');
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleClose = () => {
    if (!isSubmitting) {
      onClose();
      setUrl('');
      setNote('');
      setError(null);
      setSuccess(false);
    }
  };

  return (
    <div
      className="fixed inset-0 bg-black/50 flex items-center justify-center z-[9999] p-4"
      onClick={handleClose}
    >
      <div
        className="bg-white rounded-xl shadow-2xl w-full max-w-lg overflow-hidden"
        onClick={e => e.stopPropagation()}
      >
        {/* Header */}
        <div className="px-6 py-4 bg-gradient-to-r from-indigo-50 to-purple-50 border-b border-indigo-100">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-lg font-bold text-slate-900 flex items-center gap-2">
                <span>📎</span>
                Submit a Source
              </h2>
              <p className="text-sm text-slate-500 mt-0.5">
                for: {eventName}
              </p>
            </div>
            <button
              onClick={handleClose}
              className="text-slate-400 hover:text-slate-600 transition-colors"
              disabled={isSubmitting}
            >
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        </div>

        {/* Success State */}
        {success ? (
          <div className="p-8 text-center">
            <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-4">
              <svg className="w-8 h-8 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
              </svg>
            </div>
            <h3 className="text-xl font-bold text-slate-900 mb-2">Submitted!</h3>
            <p className="text-slate-500">
              Your source is being processed. You'll be credited once verified.
            </p>
            {selectedQuest?.bounty && (
              <p className="text-amber-600 font-medium mt-2">
                Potential reward: +{selectedQuest.bounty} credits
              </p>
            )}
          </div>
        ) : (
          <form onSubmit={handleSubmit}>
            <div className="p-6 space-y-4">
              {/* Quest context */}
              {selectedQuest && (
                <div className="bg-amber-50 border border-amber-200 rounded-lg p-3">
                  <div className="flex items-center gap-2 text-sm">
                    <span className="text-amber-600 font-medium">Quest:</span>
                    <span className="text-slate-700">{selectedQuest.description}</span>
                  </div>
                  {selectedQuest.bounty && (
                    <div className="text-xs text-amber-600 mt-1">
                      Reward: +{selectedQuest.bounty} credits
                    </div>
                  )}
                </div>
              )}

              {/* URL input */}
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-1.5">
                  Source URL <span className="text-red-500">*</span>
                </label>
                <input
                  type="url"
                  value={url}
                  onChange={(e) => setUrl(e.target.value)}
                  placeholder="https://example.com/article"
                  className="w-full px-4 py-2.5 border border-slate-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 transition-colors"
                  disabled={isSubmitting}
                  autoFocus
                />
                <p className="text-xs text-slate-400 mt-1">
                  Paste a URL to a news article, official statement, or other source
                </p>
              </div>

              {/* Note input */}
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-1.5">
                  Note <span className="text-slate-400">(optional)</span>
                </label>
                <textarea
                  value={note}
                  onChange={(e) => setNote(e.target.value)}
                  placeholder="Why is this source relevant? What does it add?"
                  rows={2}
                  className="w-full px-4 py-2.5 border border-slate-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 transition-colors resize-none"
                  disabled={isSubmitting}
                />
              </div>

              {/* Error message */}
              {error && (
                <div className="bg-red-50 border border-red-200 rounded-lg p-3 text-sm text-red-700">
                  {error}
                </div>
              )}

              {/* Guidelines */}
              <div className="bg-slate-50 rounded-lg p-3">
                <h4 className="text-xs font-semibold text-slate-600 uppercase tracking-wide mb-2">
                  Submission Guidelines
                </h4>
                <ul className="text-xs text-slate-500 space-y-1">
                  <li className="flex items-start gap-1.5">
                    <span className="text-green-500 mt-0.5">✓</span>
                    <span>News articles from reputable sources</span>
                  </li>
                  <li className="flex items-start gap-1.5">
                    <span className="text-green-500 mt-0.5">✓</span>
                    <span>Official government statements</span>
                  </li>
                  <li className="flex items-start gap-1.5">
                    <span className="text-green-500 mt-0.5">✓</span>
                    <span>Reports from NGOs or research institutions</span>
                  </li>
                  <li className="flex items-start gap-1.5">
                    <span className="text-red-500 mt-0.5">✗</span>
                    <span>Social media posts or unverified sources</span>
                  </li>
                </ul>
              </div>
            </div>

            {/* Footer */}
            <div className="px-6 py-4 bg-slate-50 border-t border-slate-100 flex gap-3">
              <button
                type="button"
                onClick={handleClose}
                className="flex-1 py-2.5 border border-slate-300 text-slate-700 rounded-lg font-medium hover:bg-slate-100 transition-colors"
                disabled={isSubmitting}
              >
                Cancel
              </button>
              <button
                type="submit"
                disabled={isSubmitting || !url.trim()}
                className="flex-1 py-2.5 bg-gradient-to-r from-indigo-500 to-purple-500 hover:from-indigo-600 hover:to-purple-600 disabled:from-slate-300 disabled:to-slate-400 text-white rounded-lg font-medium transition-all flex items-center justify-center gap-2"
              >
                {isSubmitting ? (
                  <>
                    <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                    </svg>
                    Submitting...
                  </>
                ) : (
                  <>
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                    </svg>
                    Submit Source
                  </>
                )}
              </button>
            </div>
          </form>
        )}
      </div>
    </div>
  );
};

export default ContributionModal;
