import React, { useState } from 'react';

interface Revision {
  id: string;
  timestamp: string;
  type: 'claim_added' | 'entity_linked' | 'narrative_updated' | 'source_added';
  summary: string;
  author?: string;
}

interface EventSidebarProps {
  eventId: string;
  eventSlug: string;
  currentFund: number;
  distributedPercent?: number;
  contributorCount: number;
  claimCount: number;
  sourceCount: number;
  onNavigateToDebate?: () => void;
}

// Mock data - will be replaced with real API calls
const mockRevisions: Revision[] = [
  { id: 'r1', timestamp: new Date(Date.now() - 3600000).toISOString(), type: 'claim_added', summary: '3 new claims verified', author: 'System' },
  { id: 'r2', timestamp: new Date(Date.now() - 7200000).toISOString(), type: 'source_added', summary: 'Reuters article linked', author: 'user123' },
  { id: 'r3', timestamp: new Date(Date.now() - 86400000).toISOString(), type: 'narrative_updated', summary: 'Summary regenerated', author: 'System' },
  { id: 'r4', timestamp: new Date(Date.now() - 172800000).toISOString(), type: 'entity_linked', summary: '5 entities disambiguated', author: 'System' },
];

const EventSidebar: React.FC<EventSidebarProps> = ({
  eventId: _eventId,
  eventSlug: _eventSlug,
  currentFund,
  distributedPercent = 34,
  contributorCount,
  claimCount,
  sourceCount,
  onNavigateToDebate,
}) => {
  const [showFundModal, setShowFundModal] = useState(false);
  const [fundAmount, setFundAmount] = useState(5);
  const [displayFund, setDisplayFund] = useState(currentFund);
  const [quickFundAnimation, setQuickFundAnimation] = useState(false);
  const [hasQuickFunded, setHasQuickFunded] = useState(false);

  // Format large numbers with commas
  const formatNumber = (num: number) => {
    return num.toLocaleString('en-US');
  };

  const handleFundMore = () => {
    setShowFundModal(true);
  };

  const handleQuickFund = () => {
    // Quick +1 credit with animation
    setDisplayFund(prev => prev + 1);
    setQuickFundAnimation(true);
    setHasQuickFunded(true);
    setTimeout(() => setQuickFundAnimation(false), 300);
    // TODO: Implement actual API call for +1 credit
  };

  const handleSubmitFund = () => {
    // TODO: Implement actual funding API call
    setDisplayFund(prev => prev + fundAmount);
    setShowFundModal(false);
  };

  const handleViewDebate = () => {
    if (onNavigateToDebate) {
      onNavigateToDebate();
    }
  };

  const formatTimeAgo = (timestamp: string) => {
    const diff = Date.now() - new Date(timestamp).getTime();
    const hours = Math.floor(diff / 3600000);
    const days = Math.floor(diff / 86400000);

    if (days > 0) return `${days}d ago`;
    if (hours > 0) return `${hours}h ago`;
    return 'Just now';
  };

  const getRevisionIcon = (type: Revision['type']) => {
    switch (type) {
      case 'claim_added': return '📋';
      case 'entity_linked': return '🔗';
      case 'narrative_updated': return '📝';
      case 'source_added': return '📰';
      default: return '•';
    }
  };

  return (
    <div className="w-72 flex-shrink-0 space-y-4">
      {/* Fund Card */}
      <div className="bg-white rounded-lg border border-slate-200 shadow-sm overflow-hidden">
        <div className="px-4 py-3 bg-gradient-to-r from-amber-50 to-yellow-50 border-b border-amber-100">
          <div className="flex items-center justify-between">
            <h3 className="font-semibold text-slate-800 flex items-center gap-2">
              <svg className="w-4 h-4 text-amber-600" fill="currentColor" viewBox="0 0 20 20">
                <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
              </svg>
              Event Fund
            </h3>
          </div>
        </div>

        <div className="p-4">
          {/* Fund amount - absolute number with quick fund button */}
          <div className="mb-3">
            <div className="flex items-center gap-3">
              <div className="flex items-baseline gap-2 flex-1">
                <span className={`text-3xl font-bold text-amber-600 transition-transform ${quickFundAnimation ? 'scale-110' : ''}`}>
                  {formatNumber(displayFund)}
                </span>
                <span className="text-slate-400">credits</span>
              </div>
              <button
                onClick={handleQuickFund}
                className={`w-10 h-10 flex items-center justify-center rounded-full transition-all hover:scale-110 active:scale-95 ${
                  hasQuickFunded
                    ? 'bg-amber-100 hover:bg-amber-200'
                    : 'bg-slate-100 hover:bg-slate-200'
                }`}
                title="Quick fund +1 credit"
              >
                <span className={`text-xl transition-all ${hasQuickFunded ? '' : 'grayscale opacity-50'}`}>👍</span>
              </button>
            </div>
          </div>

          {/* Distribution bar */}
          <div className="mb-3">
            <div className="flex items-center justify-between text-xs text-slate-500 mb-1">
              <span>Distribution</span>
              <span className="font-medium text-green-600">{distributedPercent}% distributed</span>
            </div>
            <div className="h-2 bg-slate-100 rounded-full overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-green-400 to-emerald-500 transition-all duration-500"
                style={{ width: `${distributedPercent}%` }}
              />
            </div>
          </div>

          {/* Fund button */}
          <button
            onClick={handleFundMore}
            className="w-full py-2.5 bg-gradient-to-r from-amber-500 to-yellow-500 hover:from-amber-600 hover:to-yellow-600 text-white rounded-lg font-medium transition-all shadow-sm hover:shadow flex items-center justify-center gap-2"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
            </svg>
            Fund This Event
          </button>

          {/* Contributors link */}
          <div className="mt-3 pt-3 border-t border-slate-100 text-center">
            <div className="text-sm text-slate-600 mb-1">
              <span className="font-semibold text-indigo-600">{contributorCount}</span> contributors
            </div>
            <button
              onClick={handleViewDebate}
              className="text-xs text-indigo-600 hover:text-indigo-800 font-medium hover:underline"
            >
              You can contribute easily →
            </button>
          </div>
        </div>
      </div>

      {/* Revision History */}
      <div className="bg-white rounded-lg border border-slate-200 shadow-sm overflow-hidden">
        <div className="px-4 py-3 border-b border-slate-100">
          <h3 className="font-semibold text-slate-800 text-sm">Revision History</h3>
        </div>

        <div className="p-3 max-h-48 overflow-y-auto">
          <div className="space-y-2">
            {mockRevisions.map((revision) => (
              <div key={revision.id} className="flex items-start gap-2 p-2 rounded-lg hover:bg-slate-50 transition-colors text-sm">
                <span className="flex-shrink-0">{getRevisionIcon(revision.type)}</span>
                <div className="flex-1 min-w-0">
                  <div className="text-slate-700">{revision.summary}</div>
                  <div className="text-xs text-slate-400 mt-0.5">
                    {formatTimeAgo(revision.timestamp)} • {revision.author}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="px-4 py-2 bg-slate-50 border-t border-slate-100">
          <button className="text-xs text-indigo-600 hover:text-indigo-800 font-medium">
            View full history →
          </button>
        </div>
      </div>

      {/* Quick Stats */}
      <div className="bg-white rounded-lg border border-slate-200 shadow-sm p-4">
        <div className="grid grid-cols-2 gap-4">
          <div className="text-center">
            <div className="text-2xl font-bold text-indigo-600">{claimCount}</div>
            <div className="text-xs text-slate-500">Verified Claims</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-purple-600">{sourceCount}</div>
            <div className="text-xs text-slate-500">Sources</div>
          </div>
        </div>
      </div>

      {/* Fund Modal - z-[9999] to ensure it's above Leaflet map layers */}
      {showFundModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-[9999]" onClick={() => setShowFundModal(false)}>
          <div className="bg-white rounded-xl shadow-xl p-6 w-96 max-w-[90vw]" onClick={e => e.stopPropagation()}>
            <h2 className="text-xl font-bold text-slate-900 mb-4">Fund This Event</h2>
            <p className="text-slate-600 text-sm mb-4">
              Support continued coverage and verification of this event. Your credits help maintain accuracy and depth.
            </p>

            {/* Amount selector */}
            <div className="mb-4">
              <label className="block text-sm font-medium text-slate-700 mb-2">Amount (credits)</label>
              <div className="flex gap-2">
                {[5, 10, 25, 50].map(amount => (
                  <button
                    key={amount}
                    onClick={() => setFundAmount(amount)}
                    className={`flex-1 py-2 rounded-lg font-medium transition-all ${
                      fundAmount === amount
                        ? 'bg-amber-500 text-white'
                        : 'bg-slate-100 text-slate-700 hover:bg-slate-200'
                    }`}
                  >
                    {amount}
                  </button>
                ))}
              </div>
            </div>

            {/* Actions */}
            <div className="flex gap-3">
              <button
                onClick={() => setShowFundModal(false)}
                className="flex-1 py-2.5 border border-slate-300 text-slate-700 rounded-lg font-medium hover:bg-slate-50 transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={handleSubmitFund}
                className="flex-1 py-2.5 bg-gradient-to-r from-amber-500 to-yellow-500 text-white rounded-lg font-medium hover:from-amber-600 hover:to-yellow-600 transition-all"
              >
                Fund {fundAmount} Credits
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default EventSidebar;
