import React, { useRef, useEffect, useState } from 'react';

interface Claim {
  id: string;
  text: string;
  event_time?: string;
  confidence?: number;
}

interface TimelineViewProps {
  claims: Claim[];
}

interface TimelineItem {
  id: string;
  text: string;
  time: Date;
  confidence: number;
}

const TimelineView: React.FC<TimelineViewProps> = ({ claims }) => {
  const [scale, setScale] = useState(1);
  const [translateX, setTranslateX] = useState(0);
  const [isDragging, setIsDragging] = useState(false);
  const [startX, setStartX] = useState(0);
  const containerRef = useRef<HTMLDivElement>(null);

  // Convert claims to timeline items
  const timelineItems: TimelineItem[] = claims
    .filter(claim => claim.event_time)
    .map(claim => ({
      id: claim.id,
      text: claim.text,
      time: new Date(claim.event_time!),
      confidence: claim.confidence || 0.5
    }))
    .sort((a, b) => a.time.getTime() - b.time.getTime());

  // Group claims by proximity (within 1 hour)
  const groupedItems: TimelineItem[][] = [];
  timelineItems.forEach(item => {
    const lastGroup = groupedItems[groupedItems.length - 1];
    if (lastGroup && lastGroup.length > 0) {
      const lastTime = lastGroup[lastGroup.length - 1].time.getTime();
      const currentTime = item.time.getTime();
      const hoursDiff = Math.abs(currentTime - lastTime) / (1000 * 60 * 60);

      if (hoursDiff < 1) {
        lastGroup.push(item);
      } else {
        groupedItems.push([item]);
      }
    } else {
      groupedItems.push([item]);
    }
  });

  // Drag handlers
  const handleMouseDown = (e: React.MouseEvent) => {
    setIsDragging(true);
    setStartX(e.clientX - translateX);
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDragging) return;
    setTranslateX(e.clientX - startX);
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? 0.9 : 1.1;
    setScale(prev => Math.max(0.5, Math.min(3, prev * delta)));
  };

  const formatTime = (date: Date) => {
    return date.toLocaleString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  if (timelineItems.length === 0) {
    return (
      <div className="h-[600px] bg-gray-900 rounded-lg flex items-center justify-center">
        <p className="text-gray-400">No timeline data available</p>
      </div>
    );
  }

  return (
    <div className="relative h-[600px] bg-gray-900 rounded-lg overflow-hidden">
      {/* Controls */}
      <div className="absolute top-4 right-4 z-10 flex gap-2">
        <button
          onClick={() => setScale(prev => Math.min(3, prev * 1.2))}
          className="px-3 py-1 bg-gray-800 hover:bg-gray-700 rounded text-sm"
        >
          +
        </button>
        <button
          onClick={() => setScale(1)}
          className="px-3 py-1 bg-gray-800 hover:bg-gray-700 rounded text-sm"
        >
          Reset
        </button>
        <button
          onClick={() => setScale(prev => Math.max(0.5, prev * 0.8))}
          className="px-3 py-1 bg-gray-800 hover:bg-gray-700 rounded text-sm"
        >
          -
        </button>
      </div>

      {/* Timeline Canvas */}
      <div
        ref={containerRef}
        className="w-full h-full overflow-hidden cursor-grab active:cursor-grabbing"
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onWheel={handleWheel}
      >
        <div
          className="relative h-full"
          style={{
            transform: `translateX(${translateX}px) scale(${scale})`,
            transformOrigin: 'left center',
            transition: isDragging ? 'none' : 'transform 0.2s ease-out',
            width: `${groupedItems.length * 400}px`,
            padding: '100px 50px'
          }}
        >
          {/* Timeline axis */}
          <div className="absolute left-0 right-0 top-1/2 h-0.5 bg-gradient-to-r from-blue-500/30 via-purple-500/30 to-blue-500/30" />

          {/* Timeline items */}
          {groupedItems.map((group, groupIdx) => {
            const x = groupIdx * 400;
            const y = 200 + (groupIdx % 2 === 0 ? 0 : 100); // Alternate heights
            const isEven = groupIdx % 2 === 0;

            return (
              <div
                key={groupIdx}
                className="absolute"
                style={{ left: `${x}px`, top: `${y}px` }}
              >
                {/* Connector line */}
                <div
                  className={`absolute left-1/2 w-0.5 bg-gradient-to-b from-blue-500 to-transparent ${
                    isEven ? 'bottom-full mb-2' : 'top-full mt-2'
                  }`}
                  style={{ height: '60px' }}
                />

                {/* Time marker */}
                <div className="absolute -translate-x-1/2 left-1/2 -bottom-16 text-center">
                  <div className="w-3 h-3 rounded-full bg-blue-500 mx-auto mb-2" />
                  <div className="text-xs text-gray-400 whitespace-nowrap">
                    {formatTime(group[0].time)}
                  </div>
                </div>

                {/* Card */}
                <div className="w-80 bg-gray-800 rounded-lg p-4 shadow-xl border border-gray-700">
                  <div className="space-y-2">
                    {group.map((item, idx) => (
                      <div key={item.id} className={idx > 0 ? 'pt-2 border-t border-gray-700' : ''}>
                        <p className="text-sm text-gray-300 leading-relaxed">{item.text}</p>
                        <div className="mt-1 flex items-center gap-2 text-xs text-gray-500">
                          <span className="px-2 py-0.5 bg-blue-500/20 text-blue-400 rounded">
                            {Math.round(item.confidence * 100)}%
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Instructions */}
      <div className="absolute bottom-4 left-4 text-xs text-gray-500">
        Drag to pan • Scroll to zoom
      </div>
    </div>
  );
};

export default TimelineView;
