import React, { useEffect, useState, useRef, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
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
  page_id?: string;
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

interface ContradictionData {
  claim1_id: string;
  claim2_id: string;
  note?: string; // LLM-generated explanation of why claims contradict
}

interface TopologyData {
  event_id: string;
  pattern: string;
  consensus_date?: string;
  claims: TopologyClaim[];
  relationships: TopologyRelationship[];
  update_chains: Array<{ metric: string; chain: string[]; current: string }>;
  contradictions: ContradictionData[];
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
  note?: string; // Contradiction explanation
}

interface GraphData {
  nodes: GraphNode[];
  links: GraphLink[];
}

const TopologyView: React.FC<TopologyViewProps> = ({ eventId, eventName }) => {
  const navigate = useNavigate();
  const [topology, setTopology] = useState<TopologyData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [hoveredNode, setHoveredNode] = useState<GraphNode | null>(null);
  const [hoveredLink, setHoveredLink] = useState<GraphLink | null>(null);
  const [linkTooltipPos, setLinkTooltipPos] = useState<{ x: number; y: number } | null>(null);
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

  // Handle resize with ResizeObserver for reliable dimension tracking
  useEffect(() => {
    const updateDimensions = () => {
      if (containerRef.current) {
        const rect = containerRef.current.getBoundingClientRect();
        const newWidth = rect.width || containerRef.current.offsetWidth || 800;
        const newHeight = Math.max(500, rect.height || containerRef.current.offsetHeight || 500);

        // Only update if dimensions actually changed
        setDimensions(prev => {
          if (prev.width !== newWidth || prev.height !== newHeight) {
            return { width: newWidth, height: newHeight };
          }
          return prev;
        });
      }
    };

    // Use ResizeObserver for more reliable tracking
    const resizeObserver = new ResizeObserver(() => {
      updateDimensions();
    });

    if (containerRef.current) {
      resizeObserver.observe(containerRef.current);
    }

    window.addEventListener('resize', updateDimensions);

    // Multiple delayed measurements to catch tab switches
    const timers = [
      setTimeout(updateDimensions, 0),
      setTimeout(updateDimensions, 100),
      setTimeout(updateDimensions, 300),
      setTimeout(updateDimensions, 500),
    ];

    updateDimensions();

    return () => {
      window.removeEventListener('resize', updateDimensions);
      resizeObserver.disconnect();
      timers.forEach(t => clearTimeout(t));
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

    // Prepare EVENT node data (will be added last for top z-index)
    // Truncate event name for display (wider box now fits ~45 chars)
    const displayName = eventName
      ? (eventName.length > 45 ? eventName.slice(0, 45) + '...' : eventName)
      : 'Event';

    const eventNode: GraphNode = {
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
    };

    // Add claim nodes first (so event renders on top)
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

      // Link from event to high-confidence claims - light mode: visible on light bg
      if (claim.plausibility >= 0.6 && !claim.is_superseded) {
        links.push({
          source: 'event-center',
          target: claim.id,
          type: 'INTAKES',
          color: 'rgba(99, 102, 241, 0.5)',
          width: 1.5
        });
      }
    });

    // Build a map of contradiction notes for quick lookup
    const contradictionNotes = new Map<string, string>();
    topology.contradictions.forEach(c => {
      if (c.note) {
        // Key by both directions since edges can be either way
        contradictionNotes.set(`${c.claim1_id}-${c.claim2_id}`, c.note);
        contradictionNotes.set(`${c.claim2_id}-${c.claim1_id}`, c.note);
      }
    });

    // Add relationship links - light mode: solid colors for visibility
    topology.relationships.forEach(rel => {
      let color = '#10b981'; // Solid emerald-500 for corroborates
      let width = 2;
      let dashed = false;
      let note: string | undefined;

      if (rel.type === 'CONTRADICTS') {
        color = '#dc2626'; // Solid red-600 for visibility
        width = 2.5;
        dashed = true;
        // Look up note for this contradiction
        note = contradictionNotes.get(`${rel.source}-${rel.target}`);
      } else if (rel.type === 'UPDATES') {
        color = '#2563eb'; // Solid blue-600 for visibility
        width = 2;
        dashed = true;
      }

      links.push({
        source: rel.source,
        target: rel.target,
        type: rel.type,
        color,
        width,
        dashed,
        note
      });
    });

    // Add EVENT node last so it renders on top of all claims
    nodes.push(eventNode);

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
        if (link.type === 'INTAKES') return 80;
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

  // Node click handler - navigate to source page
  const handleNodeClick = useCallback((node: GraphNode) => {
    if (node.type === 'claim' && node.claim) {
      // If claim has page_id, navigate to it
      if (node.claim.page_id) {
        navigate(`/page/${node.claim.page_id}#${node.claim.id}`);
      }
    }
  }, [navigate]);

  // Node hover handler - track hovered node for highlighting
  const handleNodeHover = useCallback((node: GraphNode | null, _prevNode: GraphNode | null) => {
    setHoveredNode(node);
  }, []);

  // Track mouse position for link tooltip
  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (hoveredLink && containerRef.current) {
        const rect = containerRef.current.getBoundingClientRect();
        setLinkTooltipPos({
          x: e.clientX - rect.left,
          y: e.clientY - rect.top
        });
      }
    };

    if (hoveredLink) {
      window.addEventListener('mousemove', handleMouseMove);
    }

    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
    };
  }, [hoveredLink]);

  // Link hover handler - show tooltip for contradiction notes
  const handleLinkHover = useCallback((link: any | null, _prevLink: any | null) => {
    if (link && link.type === 'CONTRADICTS' && link.note) {
      setHoveredLink(link);
    } else {
      setHoveredLink(null);
      setLinkTooltipPos(null);
    }
  }, []);

  // Custom node rendering
  const nodeCanvasObject = useCallback((node: GraphNode, ctx: CanvasRenderingContext2D, _globalScale: number) => {
    if (typeof node.x !== 'number' || typeof node.y !== 'number') return;

    // Use definite values after the guard (TypeScript narrowing)
    const x = node.x;
    const y = node.y;

    if (node.type === 'event') {
      // Event node - rounded rectangle with event info
      // Use fixed graph coordinates for proportional zoom scaling
      const eventData = node.eventData;
      const baseWidth = 220;  // Wider to fit longer event names
      const baseHeight = 90;
      const padding = 10;
      const eventFontSize = 11;

      // Animated pulse for glow
      const pulseScale = 1 + Math.sin(pulsePhase) * 0.08;
      const glowAlpha = 0.15 + Math.sin(pulsePhase) * 0.1;

      // Outer glow (animated)
      ctx.save();
      ctx.shadowColor = '#6366f1';
      ctx.shadowBlur = 15 + Math.sin(pulsePhase) * 8;
      ctx.fillStyle = `rgba(99, 102, 241, ${glowAlpha})`;
      ctx.beginPath();
      ctx.roundRect(
        x - (baseWidth * pulseScale) / 2 - 6,
        y - (baseHeight * pulseScale) / 2 - 6,
        baseWidth * pulseScale + 12,
        baseHeight * pulseScale + 12,
        12
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
      ctx.roundRect(x - baseWidth / 2, y - baseHeight / 2, baseWidth, baseHeight, 8);
      ctx.fill();

      // Border
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.4)';
      ctx.lineWidth = 2;
      ctx.stroke();

      // Event name (title)
      ctx.font = `bold ${eventFontSize * 1.1}px Sans-Serif`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      ctx.fillStyle = 'white';
      ctx.fillText(node.name, x, y - baseHeight / 2 + padding);

      // Stats row
      if (eventData) {
        const statsY = y + padding / 2;
        ctx.font = `${eventFontSize * 0.85}px Sans-Serif`;
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
        ctx.font = `${eventFontSize * 0.75}px Sans-Serif`;
        ctx.fillStyle = 'rgba(255, 255, 255, 0.5)';
        ctx.textAlign = 'center';
        ctx.fillText(eventData.pattern.toUpperCase(), x, y + baseHeight / 2 - padding);
      }

    } else {
      // Claim node - rounded rectangle
      const claim = node.claim;
      const isHovered = hoveredNode?.id === node.id;
      const size = node.val;

      // Calculate box dimensions based on text
      // Use fixed graph coordinates - these scale proportionally with zoom
      const claimFontSize = isHovered ? 11 : 12;  // Slightly smaller when expanded
      ctx.font = `${claimFontSize}px Sans-Serif`;
      const maxWidth = isHovered ? 220 : 140;  // Wider when hovered
      const padding = 8;

      // Word wrap - show more text when hovered
      const textToWrap = isHovered ? (claim?.text || node.name) : node.name;
      const words = textToWrap.split(' ');
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

      // Show more lines when hovered (up to 8), otherwise 3
      const maxLines = isHovered ? 8 : 3;
      const displayLines = lines.slice(0, maxLines);
      if (lines.length > maxLines) {
        displayLines[displayLines.length - 1] += '...';
      }

      const lineHeight = claimFontSize * 1.3;
      let boxWidth = 0;
      displayLines.forEach(line => {
        const w = ctx.measureText(line).width;
        if (w > boxWidth) boxWidth = w;
      });
      boxWidth = Math.max(boxWidth + padding * 2, 80);  // Minimum width

      // Add extra height for metadata when hovered
      const metaHeight = isHovered ? 36 : 0;  // Space for source type + click hint
      const boxHeight = displayLines.length * lineHeight + padding * 2 + metaHeight;

      // Selection highlight / glow effect when hovered
      if (isHovered) {
        ctx.shadowColor = 'rgba(99, 102, 241, 0.5)';
        ctx.shadowBlur = 15;
        ctx.fillStyle = 'rgba(99, 102, 241, 0.1)';
        ctx.fillRect(
          x - boxWidth / 2 - 4,
          y - boxHeight / 2 - 4,
          boxWidth + 8,
          boxHeight + 8
        );
        ctx.shadowBlur = 0;
      }

      // Background - light mode: white with subtle tint
      ctx.fillStyle = claim?.is_superseded
        ? 'rgba(241, 245, 249, 0.98)'  // slate-100
        : node.isContradicted
          ? 'rgba(254, 242, 242, 0.98)'  // red-50
          : isHovered
            ? 'rgba(255, 255, 255, 1)'  // solid white when hovered
            : 'rgba(255, 255, 255, 0.95)';  // white
      ctx.fillRect(x - boxWidth / 2, y - boxHeight / 2, boxWidth, boxHeight);

      // Left accent bar
      ctx.fillStyle = node.color;
      ctx.fillRect(x - boxWidth / 2, y - boxHeight / 2, 3, boxHeight);

      // Border
      ctx.strokeStyle = claim?.is_superseded ? 'rgba(148, 163, 184, 0.6)' : node.color;
      ctx.lineWidth = isHovered ? 2 : 1;
      ctx.strokeRect(x - boxWidth / 2, y - boxHeight / 2, boxWidth, boxHeight);

      // Text - dark text for light mode
      ctx.font = `${claimFontSize}px Sans-Serif`;
      ctx.fillStyle = claim?.is_superseded ? 'rgba(100, 116, 139, 0.8)' : 'rgba(30, 41, 59, 0.95)';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';

      // Offset text up when showing metadata
      const textStartY = y - ((displayLines.length - 1) * lineHeight) / 2 - (metaHeight / 2);
      let textY = textStartY;
      displayLines.forEach(line => {
        ctx.fillText(line, x, textY);
        textY += lineHeight;
      });

      // Plausibility badge (top right)
      if (claim) {
        const badgeText = `${Math.round(claim.plausibility * 100)}%`;
        const badgeFontSize = claimFontSize * 0.8;
        ctx.font = `bold ${badgeFontSize}px Sans-Serif`;
        const badgeWidth = ctx.measureText(badgeText).width + 6;
        const badgeX = x + boxWidth / 2 - badgeWidth - 2;
        const badgeY = y - boxHeight / 2 + 2;

        ctx.fillStyle = node.color;
        ctx.globalAlpha = 0.3;
        ctx.fillRect(badgeX, badgeY, badgeWidth, badgeFontSize * 1.2);
        ctx.globalAlpha = 1;

        ctx.fillStyle = node.color;
        ctx.textAlign = 'center';
        ctx.fillText(badgeText, badgeX + badgeWidth / 2, badgeY + badgeFontSize * 0.6);
      }

      // Contradiction indicator (top left)
      if (node.isContradicted) {
        ctx.font = `${claimFontSize}px Sans-Serif`;
        ctx.fillStyle = '#ef4444';
        ctx.textAlign = 'left';
        ctx.fillText('⚡', x - boxWidth / 2 + 6, y - boxHeight / 2 + claimFontSize);
      }

      // Metadata section when hovered
      if (isHovered && claim) {
        const metaY = y + boxHeight / 2 - metaHeight + 4;

        // Divider line
        ctx.strokeStyle = 'rgba(148, 163, 184, 0.3)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(x - boxWidth / 2 + 8, metaY);
        ctx.lineTo(x + boxWidth / 2 - 8, metaY);
        ctx.stroke();

        // Source type
        const metaFontSize = 10;
        ctx.font = `${metaFontSize}px Sans-Serif`;
        ctx.textAlign = 'left';
        ctx.fillStyle = 'rgba(100, 116, 139, 0.9)';

        const sourceText = claim.source_type
          ? claim.source_type.replace(/_/g, ' ')
          : 'source';
        ctx.fillText(sourceText, x - boxWidth / 2 + 8, metaY + 14);

        // Click to view hint (if has page_id)
        if (claim.page_id) {
          ctx.fillStyle = 'rgba(99, 102, 241, 0.9)';
          ctx.textAlign = 'right';
          ctx.fillText('click to view →', x + boxWidth / 2 - 8, metaY + 14);
        }

        // Date if available
        if (claim.event_time) {
          ctx.fillStyle = 'rgba(100, 116, 139, 0.7)';
          ctx.textAlign = 'left';
          const dateStr = new Date(claim.event_time).toLocaleDateString();
          ctx.fillText(dateStr, x - boxWidth / 2 + 8, metaY + 26);
        }
      }
    }
  }, [hoveredNode, pulsePhase]);

  // Link rendering
  const linkCanvasObject = useCallback((link: any, ctx: CanvasRenderingContext2D, globalScale: number) => {
    const source = link.source;
    const target = link.target;

    if (!source.x || !source.y || !target.x || !target.y) return;

    // Check if this link is hovered - handle both string IDs and resolved node objects
    const getNodeId = (node: any): string => typeof node === 'string' ? node : node?.id || '';
    const hoveredSourceId = hoveredLink ? getNodeId(hoveredLink.source) : '';
    const hoveredTargetId = hoveredLink ? getNodeId(hoveredLink.target) : '';
    const linkSourceId = getNodeId(link.source);
    const linkTargetId = getNodeId(link.target);
    const isHovered = hoveredLink &&
      (hoveredSourceId === linkSourceId && hoveredTargetId === linkTargetId);

    ctx.beginPath();
    ctx.moveTo(source.x, source.y);
    ctx.lineTo(target.x, target.y);

    // Highlight hovered links
    ctx.strokeStyle = isHovered ? '#991b1b' : link.color; // darker red when hovered
    ctx.lineWidth = (isHovered ? link.width * 1.5 : link.width) / globalScale;

    if (link.dashed) {
      ctx.setLineDash([6 / globalScale, 4 / globalScale]);
    } else {
      ctx.setLineDash([]);
    }

    ctx.stroke();
    ctx.setLineDash([]);

    // Arrow and info icon for CONTRADICTS
    if (link.type === 'CONTRADICTS') {
      const angle = Math.atan2(target.y - source.y, target.x - source.x);
      const midX = (source.x + target.x) / 2;
      const midY = (source.y + target.y) / 2;
      const arrowSize = 8 / globalScale;

      // Draw arrow
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
      ctx.fillStyle = isHovered ? '#991b1b' : link.color;
      ctx.fill();

      // Draw info icon if link has a note
      if (link.note) {
        const iconSize = 12 / globalScale;
        // Position icon slightly offset from midpoint
        const iconX = midX + 15 / globalScale;
        const iconY = midY - 15 / globalScale;

        // Circle background
        ctx.beginPath();
        ctx.arc(iconX, iconY, iconSize, 0, 2 * Math.PI);
        ctx.fillStyle = isHovered ? '#fef2f2' : '#fee2e2'; // red-50 or red-100
        ctx.fill();
        ctx.strokeStyle = isHovered ? '#991b1b' : '#dc2626';
        ctx.lineWidth = 1.5 / globalScale;
        ctx.stroke();

        // "i" text
        ctx.font = `bold ${iconSize * 1.2}px Sans-Serif`;
        ctx.fillStyle = isHovered ? '#991b1b' : '#dc2626';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('i', iconX, iconY + 1 / globalScale);
      }
    }
  }, [hoveredLink]);

  // Helper functions for display
  const getPatternStyle = (pattern: string) => {
    switch (pattern) {
      case 'consensus': return 'bg-emerald-100 text-emerald-700 border-emerald-300';
      case 'progressive': return 'bg-blue-100 text-blue-700 border-blue-300';
      case 'contradictory': return 'bg-red-100 text-red-700 border-red-300';
      case 'mixed': return 'bg-amber-100 text-amber-700 border-amber-300';
      default: return 'bg-slate-100 text-slate-600 border-slate-300';
    }
  };

  const getTemperatureInfo = (temp: number) => {
    if (temp < 0.2) return { label: 'Stable', color: 'text-emerald-600' };
    if (temp < 0.4) return { label: 'Cool', color: 'text-blue-600' };
    if (temp < 0.6) return { label: 'Warm', color: 'text-amber-600' };
    if (temp < 0.8) return { label: 'Hot', color: 'text-orange-600' };
    return { label: 'Volatile', color: 'text-red-600' };
  };

  if (loading) {
    return (
      <div className="h-[500px] bg-slate-50 rounded-lg flex items-center justify-center border border-slate-200">
        <div className="text-center">
          <div className="w-8 h-8 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin mx-auto mb-3"></div>
          <p className="text-slate-500 text-sm">Loading topology...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="h-[400px] bg-slate-50 rounded-lg flex items-center justify-center border border-slate-200">
        <div className="text-center">
          <p className="text-slate-600 mb-2">{error}</p>
          <p className="text-slate-500 text-sm">Topology analysis runs when claims are processed.</p>
        </div>
      </div>
    );
  }

  if (!topology || topology.claims.length === 0) {
    return (
      <div className="h-[400px] bg-slate-50 rounded-lg flex items-center justify-center border border-slate-200">
        <p className="text-slate-500">No claims available</p>
      </div>
    );
  }

  const tempInfo = getTemperatureInfo(topology.organism_state.temperature);
  const contradictions = topology.relationships.filter(r => r.type === 'CONTRADICTS');
  const corroborations = topology.relationships.filter(r => r.type === 'CORROBORATES');

  return (
    <div className="bg-white rounded-lg overflow-hidden border border-slate-200">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-slate-200 bg-slate-50">
        <div className="flex items-center gap-3">
          <span className={`px-3 py-1 rounded-full text-sm font-medium border ${getPatternStyle(topology.pattern)}`}>
            {topology.pattern.charAt(0).toUpperCase() + topology.pattern.slice(1)}
          </span>
          <span className="text-slate-600 text-sm font-medium">
            φ {Math.round(topology.organism_state.coherence * 100)}%
          </span>
          <span className={`text-sm ${tempInfo.color}`}>
            {tempInfo.label}
          </span>
        </div>
        <div className="flex items-center gap-3 text-sm text-slate-500">
          {contradictions.length > 0 && (
            <span className="text-red-600">
              <span className="inline-block w-2 h-2 rounded-full bg-red-500 mr-1"></span>
              {contradictions.length} conflicts
            </span>
          )}
          {corroborations.length > 0 && (
            <span className="text-emerald-600">
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
          nodeVal={(node: GraphNode) => node.val}
          onNodeClick={handleNodeClick}
          onNodeHover={handleNodeHover}
          onLinkHover={handleLinkHover}
          linkWidth={(link: GraphLink) => link.note ? 4 : link.width}
          nodeLabel={() => ''}
          backgroundColor="#f8fafc"
          d3AlphaDecay={0.02}
          d3VelocityDecay={0.3}
          cooldownTicks={100}
          nodeCanvasObjectMode={() => 'replace'}
        />

        {/* Contradiction note tooltip */}
        {hoveredLink && hoveredLink.note && linkTooltipPos && (
          <div
            className="absolute z-50 pointer-events-none"
            style={{
              left: linkTooltipPos.x,
              top: linkTooltipPos.y,
              transform: 'translate(-50%, -100%)',
              marginTop: '-12px'
            }}
          >
            <div className="bg-slate-900 text-white px-3 py-2 rounded-lg shadow-lg max-w-xs">
              <div className="flex items-center gap-2 mb-1">
                <span className="text-red-400 text-sm">⚡ Contradiction</span>
              </div>
              <p className="text-sm text-slate-200 leading-snug">
                {hoveredLink.note}
              </p>
            </div>
            {/* Arrow pointing down */}
            <div
              className="absolute left-1/2 -translate-x-1/2"
              style={{
                width: 0,
                height: 0,
                borderLeft: '6px solid transparent',
                borderRight: '6px solid transparent',
                borderTop: '6px solid #0f172a'
              }}
            />
          </div>
        )}

        {/* Legend */}
        <div className="absolute bottom-4 left-4 bg-white/95 backdrop-blur-sm rounded-lg p-3 text-xs space-y-1.5 shadow-sm border border-slate-200">
          <div className="font-medium text-slate-700 mb-2">Legend</div>
          <div className="flex items-center gap-2">
            <span className="w-3 h-3 rounded-full bg-emerald-500"></span>
            <span className="text-slate-600">High confidence</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-3 h-3 rounded-full bg-blue-500"></span>
            <span className="text-slate-600">Good confidence</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-3 h-3 rounded-full bg-amber-500"></span>
            <span className="text-slate-600">Low confidence</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-3 h-3 rounded-full bg-red-500"></span>
            <span className="text-slate-600">In contradiction</span>
          </div>
          <div className="border-t border-slate-200 pt-1.5 mt-1.5">
            <div className="flex items-center gap-2">
              <span className="w-4 h-0.5 bg-red-500"></span>
              <span className="text-slate-600">Contradicts</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="w-4 h-0.5 bg-emerald-500"></span>
              <span className="text-slate-600">Supports</span>
            </div>
          </div>
        </div>
      </div>

      {/* Topology Details Section */}
      <div className="border-t border-slate-200 p-4 space-y-4 bg-slate-50">
        {/* Summary Cards */}
        <div className="grid grid-cols-3 gap-3">
          <div className="bg-white rounded-lg p-3 border border-slate-200 shadow-sm">
            <div className="text-xs text-slate-500 uppercase tracking-wide mb-1">Pattern</div>
            <div className="text-lg font-semibold text-slate-800 capitalize">{topology.pattern}</div>
            {topology.pattern === 'progressive' && (
              <div className="text-xs text-amber-600 mt-1">Metrics evolving ↑</div>
            )}
            {topology.pattern === 'contradictory' && (
              <div className="text-xs text-red-600 mt-1">Active conflicts</div>
            )}
            {topology.pattern === 'consensus' && (
              <div className="text-xs text-emerald-600 mt-1">Sources agree</div>
            )}
          </div>

          <div className="bg-white rounded-lg p-3 border border-slate-200 shadow-sm">
            <div className="text-xs text-slate-500 uppercase tracking-wide mb-1">Consensus Date</div>
            <div className="text-lg font-semibold text-slate-800">
              {topology.consensus_date
                ? new Date(topology.consensus_date).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })
                : '—'}
            </div>
            {topology.consensus_date && (
              <div className="text-xs text-slate-500 mt-1">Most agreed timeline</div>
            )}
          </div>

          <div className="bg-white rounded-lg p-3 border border-slate-200 shadow-sm">
            <div className="text-xs text-slate-500 uppercase tracking-wide mb-1">Contradictions</div>
            <div className={`text-lg font-semibold ${topology.contradictions.length > 0 ? 'text-red-600' : 'text-emerald-600'}`}>
              {topology.contradictions.length} active
            </div>
            <div className="text-xs text-slate-500 mt-1">
              Temp: {tempInfo.label}
            </div>
          </div>
        </div>

        {/* Update Chains */}
        {topology.update_chains.length > 0 && (
          <div className="bg-white rounded-lg p-4 border border-slate-200 shadow-sm">
            <h4 className="text-sm font-medium text-slate-700 mb-3 flex items-center gap-2">
              <span className="text-blue-500">📈</span> Update Chains (Metric Progression)
            </h4>
            <div className="space-y-3">
              {topology.update_chains.map((chain, idx) => {
                // Get claim texts for the chain
                const chainClaims = chain.chain.map(claimId => {
                  const claim = topology.claims.find(c => c.id === claimId);
                  return claim;
                }).filter(Boolean) as TopologyClaim[];

                // Extract numeric values from claims if possible
                const values = chainClaims.map(c => {
                  const match = c.text.match(/(\d+(?:,\d+)?)/);
                  return match ? match[1] : '?';
                });

                return (
                  <div key={idx} className="bg-slate-50 rounded p-3 border border-slate-100">
                    <div className="text-xs text-slate-500 uppercase tracking-wide mb-2 font-medium">
                      {chain.metric}
                    </div>
                    <div className="flex items-center gap-2 flex-wrap text-sm">
                      {chainClaims.map((claim, i) => {
                        const isCurrent = claim.id === chain.current;
                        return (
                          <React.Fragment key={claim.id}>
                            <span
                              className={`px-2 py-1 rounded cursor-pointer transition-colors ${
                                isCurrent
                                  ? 'bg-blue-600 text-white font-medium'
                                  : 'bg-slate-200 text-slate-700 hover:bg-slate-300'
                              }`}
                              onClick={() => claim.page_id && navigate(`/page/${claim.page_id}#${claim.id}`)}
                              title={claim.text}
                            >
                              {values[i]}
                            </span>
                            {i < chainClaims.length - 1 && (
                              <span className="text-slate-400">→</span>
                            )}
                          </React.Fragment>
                        );
                      })}
                    </div>
                    <div className="flex items-center gap-2 mt-2 text-xs text-slate-500">
                      {chainClaims.map((claim, i) => (
                        <React.Fragment key={claim.id}>
                          <span className={claim.id === chain.current ? 'text-blue-600' : ''}>
                            {Math.round(claim.plausibility * 100)}%
                          </span>
                          {i < chainClaims.length - 1 && <span className="w-4" />}
                        </React.Fragment>
                      ))}
                      <span className="ml-auto text-slate-400">(plausibility)</span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Active Contradictions */}
        {topology.contradictions.length > 0 && (
          <div className="bg-white rounded-lg p-4 border border-slate-200 shadow-sm">
            <h4 className="text-sm font-medium text-slate-700 mb-3 flex items-center gap-2">
              <span className="text-red-500">⚡</span> Contradictions (Active Tensions)
            </h4>
            <div className="space-y-3">
              {topology.contradictions.slice(0, 5).map((contradiction, idx) => {
                const claim1 = topology.claims.find(c => c.id === contradiction.claim1_id);
                const claim2 = topology.claims.find(c => c.id === contradiction.claim2_id);
                if (!claim1 || !claim2) return null;

                return (
                  <div key={idx} className="bg-red-50 rounded p-3 border border-red-200">
                    <div className="flex items-start gap-3">
                      <div className="flex-1">
                        <div
                          className="text-sm text-slate-700 cursor-pointer hover:text-slate-900 hover:text-indigo-600 line-clamp-2"
                          onClick={() => claim1.page_id && navigate(`/page/${claim1.page_id}#${claim1.id}`)}
                        >
                          "{claim1.text.length > 80 ? claim1.text.slice(0, 80) + '...' : claim1.text}"
                        </div>
                        <div className="text-xs text-slate-500 mt-1">
                          {Math.round(claim1.plausibility * 100)}% plausible
                        </div>
                      </div>
                      <div className="text-red-500 text-lg font-bold px-2">vs</div>
                      <div className="flex-1">
                        <div
                          className="text-sm text-slate-700 cursor-pointer hover:text-slate-900 hover:text-indigo-600 line-clamp-2"
                          onClick={() => claim2.page_id && navigate(`/page/${claim2.page_id}#${claim2.id}`)}
                        >
                          "{claim2.text.length > 80 ? claim2.text.slice(0, 80) + '...' : claim2.text}"
                        </div>
                        <div className="text-xs text-slate-500 mt-1">
                          {Math.round(claim2.plausibility * 100)}% plausible
                        </div>
                      </div>
                    </div>
                    {/* Contradiction note/explanation */}
                    {contradiction.note && (
                      <div className="mt-2 pt-2 border-t border-red-200">
                        <div className="flex items-start gap-2 text-xs">
                          <span className="text-red-500 flex-shrink-0">📝</span>
                          <span className="text-slate-600 italic">{contradiction.note}</span>
                        </div>
                      </div>
                    )}
                  </div>
                );
              })}
              {topology.contradictions.length > 5 && (
                <div className="text-xs text-slate-500 text-center">
                  +{topology.contradictions.length - 5} more contradictions
                </div>
              )}
            </div>
          </div>
        )}

        {/* Source Diversity */}
        {Object.keys(topology.source_diversity).length > 0 && (
          <div className="bg-white rounded-lg p-4 border border-slate-200 shadow-sm">
            <h4 className="text-sm font-medium text-slate-700 mb-3 flex items-center gap-2">
              <span className="text-purple-500">📊</span> Source Diversity
            </h4>
            <div className="space-y-2">
              {Object.entries(topology.source_diversity)
                .sort((a, b) => b[1].avg_prior - a[1].avg_prior)
                .map(([sourceType, data]) => {
                  const maxCount = Math.max(...Object.values(topology.source_diversity).map(d => d.count));
                  const barWidth = (data.count / maxCount) * 100;
                  const priorColor = data.avg_prior >= 0.7 ? 'text-emerald-600' :
                                     data.avg_prior >= 0.5 ? 'text-amber-600' : 'text-red-600';

                  return (
                    <div key={sourceType} className="flex items-center gap-3">
                      <div className="w-24 text-sm text-slate-600 capitalize">
                        {sourceType.replace(/_/g, ' ')}
                      </div>
                      <div className={`w-12 text-sm font-medium ${priorColor}`}>
                        {Math.round(data.avg_prior * 100)}%
                      </div>
                      <div className="flex-1 h-5 bg-slate-100 rounded overflow-hidden relative">
                        <div
                          className="h-full bg-gradient-to-r from-purple-500 to-purple-400 transition-all duration-300"
                          style={{ width: `${barWidth}%` }}
                        />
                        <span className="absolute right-2 top-1/2 -translate-y-1/2 text-xs text-slate-600">
                          {data.count} claims
                        </span>
                      </div>
                    </div>
                  );
                })}
            </div>
            <div className="text-xs text-slate-500 mt-3 flex items-center gap-4">
              <span>Prior = credibility score</span>
              <span className="text-emerald-600">■ High (≥70%)</span>
              <span className="text-amber-600">■ Medium (≥50%)</span>
              <span className="text-red-600">■ Low (&lt;50%)</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default TopologyView;
