import React, { useEffect, useState, useRef, useCallback } from 'react';
import ForceGraph2D from 'react-force-graph-2d';

// Types matching the /api/event/{id}/topology response
interface TopologyClaim {
  id: string;
  text: string;
  plausibility: number;
  prior: number;
  is_superseded: boolean;
  event_time?: string;
  source_type?: string;
  corroboration_count: number;
}

interface TopologyRelationship {
  source: string;
  target: string;
  type: 'CORROBORATES' | 'CONTRADICTS' | 'UPDATES';
  similarity?: number;
}

interface OrganismState {
  coherence: number;
  temperature: number;
  active_tensions: number;
  last_updated?: string;
}

interface TopologyData {
  event_id: string;
  pattern: string;
  consensus_date?: string;
  claims: TopologyClaim[];
  relationships: TopologyRelationship[];
  update_chains: Array<{ metric: string; chain: string[]; current: string }>;
  contradictions: Array<{ claim1_id: string; claim2_id: string }>;
  source_diversity: Record<string, { count: number; avg_prior: number }>;
  organism_state: OrganismState;
}

interface TopologyViewProps {
  eventId: string;
  eventName?: string;
}

interface EventNodeData {
  coherence: number;
  temperature: number;
  claimCount: number;
  pattern: string;
  version: number; // Mock version based on claim count
}

interface GraphNode {
  id: string;
  name: string;
  type: 'event' | 'claim';
  val: number;
  color: string;
  claim?: TopologyClaim;
  eventData?: EventNodeData;
  isContradicted?: boolean;
  x?: number;
  y?: number;
  fx?: number;
  fy?: number;
}

interface GraphLink {
  source: string;
  target: string;
  type: string;
  color: string;
  width: number;
  dashed?: boolean;
}

interface GraphData {
  nodes: GraphNode[];
  links: GraphLink[];
}

const TopologyView: React.FC<TopologyViewProps> = ({ eventId, eventName }) => {
  const [topology, setTopology] = useState<TopologyData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedClaim, setSelectedClaim] = useState<TopologyClaim | null>(null);
  const [graphData, setGraphData] = useState<GraphData>({ nodes: [], links: [] });
  const [pulsePhase, setPulsePhase] = useState(0);

  const graphRef = useRef<any>();
  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 500 });

  // Animation for event node pulse
  useEffect(() => {
    const interval = setInterval(() => {
      setPulsePhase(prev => (prev + 0.05) % (2 * Math.PI));
    }, 50);
    return () => clearInterval(interval);
  }, []);

  // Handle resize
  useEffect(() => {
    const updateDimensions = () => {
      if (containerRef.current) {
        const rect = containerRef.current.getBoundingClientRect();
        setDimensions({
          width: rect.width || containerRef.current.offsetWidth || 800,
          height: Math.max(500, rect.height || containerRef.current.offsetHeight || 500)
        });
      }
    };

    window.addEventListener('resize', updateDimensions);
    // Delay initial measurement to ensure CSS is applied
    const timer = setTimeout(updateDimensions, 100);
    updateDimensions();

    return () => {
      window.removeEventListener('resize', updateDimensions);
      clearTimeout(timer);
    };
  }, []);

  // Load topology data
  useEffect(() => {
    loadTopology();
  }, [eventId]);

  const loadTopology = async () => {
    try {
      setLoading(true);
      setError(null);
      const response = await fetch(`/api/event/${eventId}/topology`);
      if (!response.ok) {
        if (response.status === 404) {
          setError('Topology not yet computed for this event');
        } else {
          throw new Error('Failed to load topology');
        }
        return;
      }
      const data = await response.json();
      setTopology(data);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // Build graph data when topology changes
  useEffect(() => {
    if (!topology) return;

    const nodes: GraphNode[] = [];
    const links: GraphLink[] = [];

    // Create a set of contradicted claim IDs
    const contradictedIds = new Set<string>();
    topology.contradictions.forEach(c => {
      contradictedIds.add(c.claim1_id);
      contradictedIds.add(c.claim2_id);
    });

    // Add EVENT node at center (fixed position)
    // Truncate event name for display
    const displayName = eventName
      ? (eventName.length > 30 ? eventName.slice(0, 30) + '...' : eventName)
      : 'Event';

    nodes.push({
      id: 'event-center',
      name: displayName,
      type: 'event',
      val: 45,
      color: '#6366f1',
      fx: 0,  // Fixed at center
      fy: -80,
      eventData: {
        coherence: topology.organism_state.coherence,
        temperature: topology.organism_state.temperature,
        claimCount: topology.claims.length,
        pattern: topology.pattern,
        version: Math.floor(topology.claims.length / 3) + 1 // Mock version
      }
    });

    // Add claim nodes
    topology.claims.forEach(claim => {
      const isContradicted = contradictedIds.has(claim.id);

      // Color based on state
      let color = '#f59e0b'; // Default amber
      if (claim.is_superseded) {
        color = '#6b7280'; // Gray
      } else if (isContradicted) {
        color = '#ef4444'; // Red
      } else if (claim.plausibility >= 0.8) {
        color = '#10b981'; // Green
      } else if (claim.plausibility >= 0.6) {
        color = '#3b82f6'; // Blue
      } else if (claim.plausibility >= 0.4) {
        color = '#8b5cf6'; // Purple
      }

      // Size based on plausibility and corroboration
      const baseSize = 8 + claim.plausibility * 12;
      const bonus = Math.min(claim.corroboration_count * 3, 10);

      nodes.push({
        id: claim.id,
        name: claim.text.length > 60 ? claim.text.slice(0, 60) + '...' : claim.text,
        type: 'claim',
        val: baseSize + bonus,
        color,
        claim,
        isContradicted
      });

      // Link from event to high-confidence claims
      if (claim.plausibility >= 0.6 && !claim.is_superseded) {
        links.push({
          source: 'event-center',
          target: claim.id,
          type: 'SUPPORTS',
          color: 'rgba(99, 102, 241, 0.3)',
          width: 1
        });
      }
    });

    // Add relationship links
    topology.relationships.forEach(rel => {
      let color = 'rgba(16, 185, 129, 0.5)'; // Green for corroborates
      let width = 1.5;
      let dashed = false;

      if (rel.type === 'CONTRADICTS') {
        color = 'rgba(239, 68, 68, 0.8)'; // Red
        width = 2.5;
        dashed = true;
      } else if (rel.type === 'UPDATES') {
        color = 'rgba(59, 130, 246, 0.6)'; // Blue
        width = 2;
        dashed = true;
      }

      links.push({
        source: rel.source,
        target: rel.target,
        type: rel.type,
        color,
        width,
        dashed
      });
    });

    setGraphData({ nodes, links });
  }, [topology]);

  // Configure forces and center view
  useEffect(() => {
    if (graphRef.current && graphData.nodes.length > 0) {
      const fg = graphRef.current;

      // Stronger repulsion for claims
      fg.d3Force('charge').strength((node: GraphNode) => {
        return node.type === 'event' ? -300 : -150;
      });

      // Keep claims close to event
      fg.d3Force('link').distance((link: GraphLink) => {
        if (link.type === 'SUPPORTS') return 80;
        if (link.type === 'CONTRADICTS') return 120;
        return 100;
      }).strength(0.5);

      // Center force
      fg.d3Force('center').strength(0.1);

      // Center view after simulation settles
      setTimeout(() => {
        fg.centerAt(0, 50, 500);
        fg.zoom(0.9, 500);
      }, 300);
    }
  }, [graphData]);

  // Node click handler
  const handleNodeClick = useCallback((node: GraphNode) => {
    if (node.type === 'claim' && node.claim) {
      setSelectedClaim(prev => prev?.id === node.claim?.id ? null : node.claim!);
    }
  }, []);

  // Custom node rendering
  const nodeCanvasObject = useCallback((node: GraphNode, ctx: CanvasRenderingContext2D, globalScale: number) => {
    if (typeof node.x !== 'number' || typeof node.y !== 'number') return;

    // Use definite values after the guard (TypeScript narrowing)
    const x = node.x;
    const y = node.y;

    const fontSize = Math.max(10 / globalScale, 8);

    if (node.type === 'event') {
      // Event node - rounded rectangle with event info
      const eventData = node.eventData;
      const baseWidth = 160 / globalScale;
      const baseHeight = 80 / globalScale;
      const padding = 8 / globalScale;

      // Animated pulse for glow
      const pulseScale = 1 + Math.sin(pulsePhase) * 0.08;
      const glowAlpha = 0.15 + Math.sin(pulsePhase) * 0.1;

      // Outer glow (animated)
      ctx.save();
      ctx.shadowColor = '#6366f1';
      ctx.shadowBlur = (15 + Math.sin(pulsePhase) * 8) / globalScale;
      ctx.fillStyle = `rgba(99, 102, 241, ${glowAlpha})`;
      ctx.beginPath();
      ctx.roundRect(
        x - (baseWidth * pulseScale) / 2 - 6 / globalScale,
        y - (baseHeight * pulseScale) / 2 - 6 / globalScale,
        baseWidth * pulseScale + 12 / globalScale,
        baseHeight * pulseScale + 12 / globalScale,
        12 / globalScale
      );
      ctx.fill();
      ctx.restore();

      // Main background
      const gradient = ctx.createLinearGradient(x - baseWidth / 2, y, x + baseWidth / 2, y);
      gradient.addColorStop(0, '#4f46e5');
      gradient.addColorStop(0.5, '#6366f1');
      gradient.addColorStop(1, '#818cf8');
      ctx.fillStyle = gradient;
      ctx.beginPath();
      ctx.roundRect(x - baseWidth / 2, y - baseHeight / 2, baseWidth, baseHeight, 8 / globalScale);
      ctx.fill();

      // Border
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.4)';
      ctx.lineWidth = 2 / globalScale;
      ctx.stroke();

      // Event name (title)
      ctx.font = `bold ${fontSize * 1.1}px Sans-Serif`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      ctx.fillStyle = 'white';
      ctx.fillText(node.name, x, y - baseHeight / 2 + padding);

      // Stats row
      if (eventData) {
        const statsY = y + padding / 2;
        ctx.font = `${fontSize * 0.75}px Sans-Serif`;
        ctx.textBaseline = 'middle';

        // Coherence
        const coherenceColor = eventData.coherence >= 0.7 ? '#4ade80' : eventData.coherence >= 0.4 ? '#fbbf24' : '#f87171';
        ctx.fillStyle = coherenceColor;
        ctx.textAlign = 'left';
        ctx.fillText(`⚡${Math.round(eventData.coherence * 100)}%`, x - baseWidth / 2 + padding, statsY);

        // Version (mock)
        ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
        ctx.textAlign = 'center';
        ctx.fillText(`v${eventData.version}`, x, statsY);

        // Claim count
        ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
        ctx.textAlign = 'right';
        ctx.fillText(`${eventData.claimCount} claims`, x + baseWidth / 2 - padding, statsY);

        // Pattern badge at bottom
        ctx.font = `${fontSize * 0.65}px Sans-Serif`;
        ctx.fillStyle = 'rgba(255, 255, 255, 0.5)';
        ctx.textAlign = 'center';
        ctx.fillText(eventData.pattern.toUpperCase(), x, y + baseHeight / 2 - padding);
      }

    } else {
      // Claim node - rounded rectangle
      const claim = node.claim;
      const isSelected = selectedClaim?.id === node.id;
      const size = node.val;

      // Calculate box dimensions based on text
      ctx.font = `${fontSize * 0.9}px Sans-Serif`;
      const maxWidth = 140 / globalScale;
      const padding = 8 / globalScale;

      // Word wrap
      const words = node.name.split(' ');
      const lines: string[] = [];
      let currentLine = '';

      for (const word of words) {
        const testLine = currentLine + (currentLine ? ' ' : '') + word;
        const metrics = ctx.measureText(testLine);
        if (metrics.width > maxWidth && currentLine) {
          lines.push(currentLine);
          currentLine = word;
        } else {
          currentLine = testLine;
        }
      }
      if (currentLine) lines.push(currentLine);
      const displayLines = lines.slice(0, 3);

      const lineHeight = fontSize * 1.2;
      let boxWidth = 0;
      displayLines.forEach(line => {
        const w = ctx.measureText(line).width;
        if (w > boxWidth) boxWidth = w;
      });
      boxWidth += padding * 2;
      const boxHeight = displayLines.length * lineHeight + padding * 2;

      // Selection highlight
      if (isSelected) {
        ctx.fillStyle = 'rgba(255, 255, 255, 0.15)';
        ctx.fillRect(
          x - boxWidth / 2 - 4,
          y - boxHeight / 2 - 4,
          boxWidth + 8,
          boxHeight + 8
        );
      }

      // Background
      ctx.fillStyle = claim?.is_superseded
        ? 'rgba(107, 114, 128, 0.3)'
        : node.isContradicted
          ? 'rgba(239, 68, 68, 0.2)'
          : 'rgba(30, 41, 59, 0.8)';
      ctx.fillRect(x - boxWidth / 2, y - boxHeight / 2, boxWidth, boxHeight);

      // Left accent bar
      ctx.fillStyle = node.color;
      ctx.fillRect(x - boxWidth / 2, y - boxHeight / 2, 3 / globalScale, boxHeight);

      // Border
      ctx.strokeStyle = node.color;
      ctx.lineWidth = (isSelected ? 2 : 1) / globalScale;
      ctx.strokeRect(x - boxWidth / 2, y - boxHeight / 2, boxWidth, boxHeight);

      // Text
      ctx.fillStyle = claim?.is_superseded ? 'rgba(156, 163, 175, 0.8)' : 'rgba(226, 232, 240, 0.95)';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';

      let textY = y - ((displayLines.length - 1) * lineHeight) / 2;
      displayLines.forEach(line => {
        ctx.fillText(line, x, textY);
        textY += lineHeight;
      });

      // Plausibility badge
      if (claim) {
        const badgeText = `${Math.round(claim.plausibility * 100)}%`;
        ctx.font = `bold ${fontSize * 0.7}px Sans-Serif`;
        const badgeWidth = ctx.measureText(badgeText).width + 6 / globalScale;
        const badgeX = x + boxWidth / 2 - badgeWidth - 2 / globalScale;
        const badgeY = y - boxHeight / 2 + 2 / globalScale;

        ctx.fillStyle = node.color;
        ctx.globalAlpha = 0.3;
        ctx.fillRect(badgeX, badgeY, badgeWidth, fontSize);
        ctx.globalAlpha = 1;

        ctx.fillStyle = node.color;
        ctx.textAlign = 'center';
        ctx.fillText(badgeText, badgeX + badgeWidth / 2, badgeY + fontSize / 2);
      }

      // Contradiction indicator
      if (node.isContradicted) {
        ctx.font = `${fontSize}px Sans-Serif`;
        ctx.fillStyle = '#ef4444';
        ctx.textAlign = 'left';
        ctx.fillText('⚡', x - boxWidth / 2 + 6 / globalScale, y - boxHeight / 2 + fontSize);
      }
    }
  }, [selectedClaim, pulsePhase]);

  // Link rendering
  const linkCanvasObject = useCallback((link: any, ctx: CanvasRenderingContext2D, globalScale: number) => {
    const source = link.source;
    const target = link.target;

    if (!source.x || !source.y || !target.x || !target.y) return;

    ctx.beginPath();
    ctx.moveTo(source.x, source.y);
    ctx.lineTo(target.x, target.y);

    ctx.strokeStyle = link.color;
    ctx.lineWidth = link.width / globalScale;

    if (link.dashed) {
      ctx.setLineDash([6 / globalScale, 4 / globalScale]);
    } else {
      ctx.setLineDash([]);
    }

    ctx.stroke();
    ctx.setLineDash([]);

    // Arrow for CONTRADICTS
    if (link.type === 'CONTRADICTS') {
      const angle = Math.atan2(target.y - source.y, target.x - source.x);
      const midX = (source.x + target.x) / 2;
      const midY = (source.y + target.y) / 2;
      const arrowSize = 8 / globalScale;

      ctx.beginPath();
      ctx.moveTo(midX, midY);
      ctx.lineTo(
        midX - arrowSize * Math.cos(angle - Math.PI / 6),
        midY - arrowSize * Math.sin(angle - Math.PI / 6)
      );
      ctx.lineTo(
        midX - arrowSize * Math.cos(angle + Math.PI / 6),
        midY - arrowSize * Math.sin(angle + Math.PI / 6)
      );
      ctx.closePath();
      ctx.fillStyle = link.color;
      ctx.fill();
    }
  }, []);

  // Helper functions for display
  const getPatternStyle = (pattern: string) => {
    switch (pattern) {
      case 'consensus': return 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30';
      case 'progressive': return 'bg-blue-500/20 text-blue-400 border-blue-500/30';
      case 'contradictory': return 'bg-red-500/20 text-red-400 border-red-500/30';
      case 'mixed': return 'bg-amber-500/20 text-amber-400 border-amber-500/30';
      default: return 'bg-gray-500/20 text-gray-400 border-gray-500/30';
    }
  };

  const getTemperatureInfo = (temp: number) => {
    if (temp < 0.2) return { label: 'Stable', color: 'text-emerald-400' };
    if (temp < 0.4) return { label: 'Cool', color: 'text-blue-400' };
    if (temp < 0.6) return { label: 'Warm', color: 'text-amber-400' };
    if (temp < 0.8) return { label: 'Hot', color: 'text-orange-400' };
    return { label: 'Volatile', color: 'text-red-400' };
  };

  if (loading) {
    return (
      <div className="h-[500px] bg-gray-900 rounded-lg flex items-center justify-center">
        <div className="text-center">
          <div className="w-8 h-8 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin mx-auto mb-3"></div>
          <p className="text-gray-400 text-sm">Loading topology...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="h-[400px] bg-gray-900 rounded-lg flex items-center justify-center">
        <div className="text-center">
          <p className="text-gray-400 mb-2">{error}</p>
          <p className="text-gray-500 text-sm">Topology analysis runs when claims are processed.</p>
        </div>
      </div>
    );
  }

  if (!topology || topology.claims.length === 0) {
    return (
      <div className="h-[400px] bg-gray-900 rounded-lg flex items-center justify-center">
        <p className="text-gray-400">No claims available</p>
      </div>
    );
  }

  const tempInfo = getTemperatureInfo(topology.organism_state.temperature);
  const contradictions = topology.relationships.filter(r => r.type === 'CONTRADICTS');
  const corroborations = topology.relationships.filter(r => r.type === 'CORROBORATES');

  return (
    <div className="bg-gray-900 rounded-lg overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-800">
        <div className="flex items-center gap-3">
          <span className={`px-3 py-1 rounded-full text-sm font-medium border ${getPatternStyle(topology.pattern)}`}>
            {topology.pattern.charAt(0).toUpperCase() + topology.pattern.slice(1)}
          </span>
          <span className="text-gray-400 text-sm">
            φ {Math.round(topology.organism_state.coherence * 100)}%
          </span>
          <span className={`text-sm ${tempInfo.color}`}>
            {tempInfo.label}
          </span>
        </div>
        <div className="flex items-center gap-3 text-sm text-gray-500">
          {contradictions.length > 0 && (
            <span className="text-red-400">
              <span className="inline-block w-2 h-2 rounded-full bg-red-500 mr-1"></span>
              {contradictions.length} conflicts
            </span>
          )}
          {corroborations.length > 0 && (
            <span className="text-emerald-400">
              <span className="inline-block w-2 h-2 rounded-full bg-emerald-500 mr-1"></span>
              {corroborations.length} support
            </span>
          )}
          <span>{topology.claims.length} claims</span>
        </div>
      </div>

      {/* Graph container */}
      <div ref={containerRef} className="relative w-full" style={{ minHeight: '500px', height: '60vh' }}>
        <ForceGraph2D
          ref={graphRef}
          width={dimensions.width}
          height={dimensions.height}
          graphData={graphData}
          nodeCanvasObject={nodeCanvasObject}
          linkCanvasObject={linkCanvasObject}
          nodePointerAreaPaint={(node: GraphNode, color, ctx) => {
            if (node.type === 'event') {
              ctx.beginPath();
              ctx.arc(node.x!, node.y!, node.val, 0, 2 * Math.PI);
              ctx.fillStyle = color;
              ctx.fill();
            } else {
              ctx.fillStyle = color;
              ctx.fillRect(node.x! - 70, node.y! - 30, 140, 60);
            }
          }}
          onNodeClick={handleNodeClick}
          backgroundColor="#111827"
          d3AlphaDecay={0.02}
          d3VelocityDecay={0.3}
          cooldownTicks={100}
        />

        {/* Legend */}
        <div className="absolute bottom-4 left-4 bg-gray-800/90 rounded-lg p-3 text-xs space-y-1.5">
          <div className="font-medium text-gray-300 mb-2">Legend</div>
          <div className="flex items-center gap-2">
            <span className="w-3 h-3 rounded-full bg-emerald-500"></span>
            <span className="text-gray-400">High confidence</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-3 h-3 rounded-full bg-blue-500"></span>
            <span className="text-gray-400">Good confidence</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-3 h-3 rounded-full bg-amber-500"></span>
            <span className="text-gray-400">Low confidence</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-3 h-3 rounded-full bg-red-500"></span>
            <span className="text-gray-400">In contradiction</span>
          </div>
          <div className="border-t border-gray-700 pt-1.5 mt-1.5">
            <div className="flex items-center gap-2">
              <span className="w-4 h-0.5 bg-red-500"></span>
              <span className="text-gray-400">Contradicts</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="w-4 h-0.5 bg-emerald-500"></span>
              <span className="text-gray-400">Supports</span>
            </div>
          </div>
        </div>
      </div>

      {/* Selected claim detail */}
      {selectedClaim && (
        <div className="border-t border-gray-800 p-4 bg-gray-800/80">
          <div className="flex justify-between items-start mb-2">
            <div className="flex items-center gap-2">
              <span className="text-white font-medium">
                {Math.round(selectedClaim.plausibility * 100)}% plausible
              </span>
              <span className="text-gray-500 text-sm">
                (prior: {Math.round(selectedClaim.prior * 100)}%)
              </span>
            </div>
            <button
              onClick={() => setSelectedClaim(null)}
              className="text-gray-500 hover:text-white"
            >
              ×
            </button>
          </div>
          <p className="text-gray-200 text-sm mb-2">{selectedClaim.text}</p>
          <div className="flex flex-wrap gap-2 text-xs">
            {selectedClaim.is_superseded && (
              <span className="px-2 py-1 bg-gray-700 rounded text-gray-400">Superseded</span>
            )}
            {selectedClaim.corroboration_count > 0 && (
              <span className="px-2 py-1 bg-emerald-900/50 text-emerald-400 rounded">
                +{selectedClaim.corroboration_count} corroborating
              </span>
            )}
            {selectedClaim.event_time && (
              <span className="px-2 py-1 bg-gray-700 text-gray-400 rounded">
                {new Date(selectedClaim.event_time).toLocaleDateString()}
              </span>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default TopologyView;
