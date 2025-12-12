import React, { useState, useMemo } from 'react';

interface Claim {
  id: string;
  text: string;
  event_time?: string;
  confidence?: number;
}

interface TimelineCardProps {
  claims: Claim[];
  eventSlug: string;
}

interface TimelineItem {
  id: string;
  text: string;
  time: Date;
  confidence: number;
}

const TimelineCard: React.FC<TimelineCardProps> = ({ claims, eventSlug: _eventSlug }) => {
  const [zoomLevel, setZoomLevel] = useState<'compact' | 'expanded'>('compact');

  // Convert claims to timeline items
  const timelineItems: TimelineItem[] = useMemo(() => {
    return claims
      .filter(claim => claim.event_time)
      .map(claim => ({
        id: claim.id,
        text: claim.text,
        time: new Date(claim.event_time!),
        confidence: claim.confidence || 0.5
      }))
      .sort((a, b) => a.time.getTime() - b.time.getTime());
  }, [claims]);

  // Group items by day for compact view, or show individual for expanded
  const groupedByDay = useMemo(() => {
    const groups: Map<string, TimelineItem[]> = new Map();
    timelineItems.forEach(item => {
      const dayKey = item.time.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
      if (!groups.has(dayKey)) {
        groups.set(dayKey, []);
      }
      groups.get(dayKey)!.push(item);
    });
    return Array.from(groups.entries());
  }, [timelineItems]);

  const handleZoomIn = (e: React.MouseEvent) => {
    e.stopPropagation();
    setZoomLevel('expanded');
  };

  const handleZoomOut = (e: React.MouseEvent) => {
    e.stopPropagation();
    setZoomLevel('compact');
  };

  const handleWheel = (e: React.WheelEvent) => {
    if (e.deltaY < 0) {
      setZoomLevel('expanded');
    } else {
      setZoomLevel('compact');
    }
  };

  const formatTime = (date: Date) => {
    return date.toLocaleString('en-US', {
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  // No timeline data
  if (timelineItems.length === 0) {
    return null;
  }

  const isExpanded = zoomLevel === 'expanded';
  const displayLimit = isExpanded ? 8 : 4;

  return (
    <div
      className={`bg-white rounded-lg border border-slate-200 shadow-md overflow-hidden w-64 transition-all duration-300 ${
        isExpanded ? 'max-h-[400px]' : 'max-h-[240px]'
      }`}
      onWheel={handleWheel}
    >
      {/* Header */}
      <div className="px-3 py-2 bg-slate-50 border-b border-slate-200 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-slate-600 text-sm font-medium">Timeline</span>
          <span className="text-xs text-slate-400">({timelineItems.length})</span>
        </div>
        <div className="flex items-center gap-1">
          <button
            onClick={handleZoomIn}
            className={`p-1 rounded hover:bg-slate-200 transition-colors ${isExpanded ? 'text-indigo-600' : 'text-slate-400'}`}
            title="Zoom in (more detail)"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0zM10 7v3m0 0v3m0-3h3m-3 0H7" />
            </svg>
          </button>
          <button
            onClick={handleZoomOut}
            className={`p-1 rounded hover:bg-slate-200 transition-colors ${!isExpanded ? 'text-indigo-600' : 'text-slate-400'}`}
            title="Zoom out (summary)"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0zM13 10H7" />
            </svg>
          </button>
        </div>
      </div>

      {/* Timeline Content */}
      <div className={`overflow-y-auto ${isExpanded ? 'max-h-[340px]' : 'max-h-[180px]'}`}>
        <div className="relative pl-6 pr-3 py-3">
          {/* Vertical line */}
          <div className="absolute left-[18px] top-3 bottom-3 w-0.5 bg-gradient-to-b from-indigo-400 via-purple-400 to-slate-200" />

          {/* Timeline items */}
          {isExpanded ? (
            // Expanded: show individual items
            <div className="space-y-3">
              {timelineItems.slice(0, displayLimit).map((item) => (
                <div key={item.id} className="relative">
                  {/* Dot */}
                  <div className="absolute -left-[14px] top-1.5 w-2.5 h-2.5 rounded-full bg-indigo-500 border-2 border-white shadow-sm" />

                  {/* Content */}
                  <div className="bg-slate-50 rounded-lg p-2 border border-slate-100">
                    <div className="text-xs text-slate-400 mb-1">
                      {item.time.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })} {formatTime(item.time)}
                    </div>
                    <p className="text-xs text-slate-700 line-clamp-2 leading-relaxed">
                      {item.text}
                    </p>
                    <div className="mt-1">
                      <span className="text-[10px] px-1.5 py-0.5 bg-indigo-100 text-indigo-600 rounded">
                        {Math.round(item.confidence * 100)}%
                      </span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            // Compact: show grouped by day
            <div className="space-y-2">
              {groupedByDay.slice(0, displayLimit).map(([day, items]) => (
                <div key={day} className="relative">
                  {/* Dot */}
                  <div className="absolute -left-[14px] top-1 w-2.5 h-2.5 rounded-full bg-indigo-500 border-2 border-white shadow-sm" />

                  {/* Content */}
                  <div className="flex items-baseline gap-2">
                    <span className="text-xs font-medium text-slate-600 whitespace-nowrap">{day}</span>
                    <span className="text-xs text-slate-400">
                      {items.length} event{items.length > 1 ? 's' : ''}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Show more indicator */}
          {(isExpanded ? timelineItems.length : groupedByDay.length) > displayLimit && (
            <div className="mt-3 text-center">
              <span className="text-xs text-slate-400">
                +{(isExpanded ? timelineItems.length : groupedByDay.length) - displayLimit} more
              </span>
            </div>
          )}
        </div>
      </div>

      {/* Footer */}
      <div className="px-3 py-2 bg-slate-50 border-t border-slate-200 text-xs text-slate-400">
        Scroll to zoom • {isExpanded ? 'Detailed view' : 'Summary view'}
      </div>
    </div>
  );
};

export default TimelineCard;
