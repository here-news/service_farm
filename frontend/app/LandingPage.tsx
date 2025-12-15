import React, { useEffect, useState, useRef, useCallback } from 'react';
import ForceGraph2D from 'react-force-graph-2d';

interface Entity {
    id: string;
    canonical_name: string;
    entity_type?: string;
    image_url?: string;
}

interface Event {
    event_id: string;
    id: string;
    title: string;
}

interface GraphNode {
    id: string;
    name: string;
    type: 'entity';
    val: number;
    color?: string;
    imgUrl?: string;
    img?: HTMLImageElement;
    eventIds?: string[];
    recency: number; // 0 = most recent, 1 = oldest
    x?: number;
    y?: number;
    vx?: number;
    vy?: number;
    fx?: number; // Fixed x position (for center attraction)
    fy?: number; // Fixed y position
}

interface GraphLink {
    source: string | GraphNode;
    target: string | GraphNode;
    color?: string;
    width?: number;
}

interface GraphData {
    nodes: GraphNode[];
    links: GraphLink[];
}

const COLORS: Record<string, string> = {
    PERSON: '#a78bfa',
    ORGANIZATION: '#f59e0b',
    LOCATION: '#10b981',
    DEFAULT: '#64748b'
};

const MAX_ENTITIES = 100;

// Lightning bolt drawing (from GraphPage)
function drawLightningBolt(
    ctx: CanvasRenderingContext2D,
    x1: number, y1: number,
    x2: number, y2: number,
    progress: number,
    globalScale: number
) {
    const segments = 8;
    const jitter = 20 / globalScale;

    const points: { x: number; y: number }[] = [{ x: x1, y: y1 }];
    const dx = x2 - x1;
    const dy = y2 - y1;
    const dist = Math.sqrt(dx * dx + dy * dy);

    for (let i = 1; i < segments; i++) {
        const t = i / segments;
        if (t > progress) break;

        const baseX = x1 + dx * t;
        const baseY = y1 + dy * t;

        const perpX = -dy / dist;
        const perpY = dx / dist;
        const offset = (Math.random() - 0.5) * 2 * jitter;

        points.push({
            x: baseX + perpX * offset,
            y: baseY + perpY * offset
        });
    }

    if (progress >= 0.9) {
        points.push({ x: x2, y: y2 });
    }

    if (points.length < 2) return;

    ctx.save();
    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);

    for (let i = 1; i < points.length; i++) {
        ctx.lineTo(points[i].x, points[i].y);
    }

    // Glow
    ctx.shadowColor = '#fbbf24';
    ctx.shadowBlur = 8;
    ctx.strokeStyle = '#fef08a';
    ctx.lineWidth = 2 / globalScale;
    ctx.stroke();

    // Inner line
    ctx.shadowBlur = 0;
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 1 / globalScale;
    ctx.stroke();

    ctx.restore();
}

const LandingPage: React.FC = () => {
    const [data, setData] = useState<GraphData>({ nodes: [], links: [] });
    const [loading, setLoading] = useState(true);
    const graphRef = useRef<any>();
    const containerRef = useRef<HTMLDivElement>(null);
    const [dimensions, setDimensions] = useState({ width: 800, height: 600 });

    // Zap state
    const zapRef = useRef<{
        sourceId: string;
        targetId: string;
        progress: number;
    } | null>(null);

    // Handle resize
    useEffect(() => {
        const updateDimensions = () => {
            if (containerRef.current) {
                setDimensions({
                    width: containerRef.current.offsetWidth,
                    height: containerRef.current.offsetHeight
                });
            }
        };

        window.addEventListener('resize', updateDimensions);
        updateDimensions();

        return () => window.removeEventListener('resize', updateDimensions);
    }, []);

    // Fetch data
    useEffect(() => {
        const fetchData = async () => {
            try {
                const feedResponse = await fetch('/api/coherence/feed?limit=50');
                const feedResult = await feedResponse.json();

                const events = feedResult.events || [];
                if (events.length === 0) {
                    setLoading(false);
                    return;
                }

                const eventDetailsPromises = events.slice(0, 25).map((event: Event) =>
                    fetch(`/api/event/${event.event_id || event.id}`)
                        .then(res => res.json())
                        .catch(() => null)
                );

                const eventDetailsResults = await Promise.all(eventDetailsPromises);
                const validEventDetails = eventDetailsResults.filter(r => r && r.event);

                const entityMap = new Map<string, { entity: Entity; eventIds: string[]; firstEventIndex: number }>();
                const totalEvents = validEventDetails.length;

                validEventDetails.forEach((result, eventIndex) => {
                    const event = result.event;
                    const eventId = event.event_id || event.id;

                    const entities = result.entities || [];
                    entities.forEach((entity: Entity) => {
                        if (entity.id && entity.canonical_name) {
                            if (!entityMap.has(entity.id)) {
                                // First appearance - eventIndex 0 is most recent
                                entityMap.set(entity.id, { entity, eventIds: [], firstEventIndex: eventIndex });
                            }
                            entityMap.get(entity.id)!.eventIds.push(eventId);
                        }
                    });
                });

                // Sort by importance
                let maxMentions = 1;
                entityMap.forEach(({ eventIds }) => {
                    maxMentions = Math.max(maxMentions, eventIds.length);
                });

                const sortedEntities = Array.from(entityMap.entries())
                    .map(([id, { entity, eventIds, firstEventIndex }]) => ({
                        id,
                        entity,
                        eventIds,
                        firstEventIndex,
                        // Score includes recency bonus (lower index = more recent = higher score)
                        score: eventIds.length * 2 + (entity.image_url ? 5 : 0) + (entity.entity_type === 'PERSON' ? 3 : 0) + (totalEvents - firstEventIndex)
                    }))
                    .sort((a, b) => b.score - a.score)
                    .slice(0, MAX_ENTITIES);

                // Build nodes with recency-based initial positioning
                const centerX = dimensions.width / 2;
                const centerY = dimensions.height / 2;
                const maxRadius = Math.min(dimensions.width, dimensions.height) * 0.4;

                const nodes: GraphNode[] = sortedEntities.map(({ id, entity, eventIds, firstEventIndex }) => {
                    const mentionRatio = eventIds.length / maxMentions;
                    const hasImage = !!entity.image_url;
                    const nodeSize = hasImage ? (18 + mentionRatio * 22) : (6 + mentionRatio * 8);
                    const color = COLORS[entity.entity_type || 'DEFAULT'] || COLORS.DEFAULT;

                    // Recency: 0 = most recent, 1 = oldest
                    const recency = totalEvents > 1 ? firstEventIndex / (totalEvents - 1) : 0;

                    // Position based on recency - recent entities start closer to center
                    const angle = Math.random() * Math.PI * 2;
                    const radius = (0.1 + recency * 0.9) * maxRadius; // 10%-100% of maxRadius based on recency
                    const initialX = centerX + Math.cos(angle) * radius;
                    const initialY = centerY + Math.sin(angle) * radius;

                    return {
                        id,
                        name: entity.canonical_name,
                        type: 'entity' as const,
                        val: nodeSize,
                        color,
                        imgUrl: entity.image_url,
                        eventIds,
                        recency,
                        x: initialX,
                        y: initialY
                    };
                });

                // Build links between co-occurring entities
                const links: GraphLink[] = [];
                for (let i = 0; i < nodes.length; i++) {
                    for (let j = i + 1; j < nodes.length; j++) {
                        const shared = nodes[i].eventIds?.filter(e => nodes[j].eventIds?.includes(e)) || [];
                        if (shared.length > 0) {
                            links.push({
                                source: nodes[i].id,
                                target: nodes[j].id,
                                color: `rgba(148, 163, 184, ${Math.min(0.5, shared.length * 0.2)})`,
                                width: Math.min(2, shared.length * 0.5)
                            });
                        }
                    }
                }

                // Pre-load images
                nodes.forEach(node => {
                    if (node.imgUrl) {
                        const img = new Image();
                        img.crossOrigin = 'anonymous';
                        img.src = node.imgUrl;
                        img.onload = () => {
                            node.img = img;
                        };
                    }
                });

                setData({ nodes, links });
                console.log(`Loaded ${nodes.length} entities, ${links.length} links`);

            } catch (error) {
                console.error('Error fetching graph data:', error);
            } finally {
                setLoading(false);
            }
        };

        fetchData();
    }, [dimensions.width, dimensions.height]);

    // Configure forces - recent entities pulled to center
    useEffect(() => {
        if (graphRef.current && data.nodes.length > 0) {
            const fg = graphRef.current;
            const centerX = dimensions.width / 2;
            const centerY = dimensions.height / 2;

            // Weak repulsion
            fg.d3Force('charge').strength(-50);
            fg.d3Force('link').distance(150).strength(0.05);

            // Custom radial force based on recency
            // Recent entities (recency=0) get pulled to center
            // Old entities (recency=1) get pushed to edges
            const d3 = (window as any).d3;
            if (d3 && d3.forceRadial) {
                fg.d3Force('recency', d3.forceRadial(
                    (node: GraphNode) => {
                        // Target radius based on recency
                        const maxRadius = Math.min(dimensions.width, dimensions.height) * 0.35;
                        return node.recency * maxRadius;
                    },
                    centerX,
                    centerY
                ).strength((node: GraphNode) => {
                    // Stronger pull for recent entities
                    return 0.1 - node.recency * 0.05;
                }));
            } else {
                // Fallback: adjust center force
                fg.d3Force('center').strength(0.02);
            }
        }
    }, [data.nodes.length, dimensions]);

    // Add Brownian motion - nudge nodes randomly
    useEffect(() => {
        if (data.nodes.length === 0 || !graphRef.current) return;

        const interval = setInterval(() => {
            if (!graphRef.current) return;

            // Pick a few random nodes to nudge
            const numToNudge = Math.max(3, Math.floor(data.nodes.length * 0.1));
            for (let i = 0; i < numToNudge; i++) {
                const node = data.nodes[Math.floor(Math.random() * data.nodes.length)] as any;
                if (typeof node.vx === 'number') {
                    node.vx += (Math.random() - 0.5) * 2;
                    node.vy += (Math.random() - 0.5) * 2;
                }
            }

            // Reheat simulation slightly to process the velocity changes
            graphRef.current.d3ReheatSimulation();
        }, 500);

        return () => clearInterval(interval);
    }, [data.nodes]);

    // Trigger zaps
    useEffect(() => {
        if (data.nodes.length < 2) return;

        const triggerZap = () => {
            // Pick from linked pairs preferentially
            if (data.links.length > 0 && Math.random() < 0.7) {
                const link = data.links[Math.floor(Math.random() * data.links.length)];
                const sourceId = typeof link.source === 'string' ? link.source : link.source.id;
                const targetId = typeof link.target === 'string' ? link.target : link.target.id;

                zapRef.current = { sourceId, targetId, progress: 0 };
            } else {
                // Random pair
                const withImages = data.nodes.filter(n => n.img);
                const candidates = withImages.length >= 2 ? withImages : data.nodes;
                const idx1 = Math.floor(Math.random() * candidates.length);
                let idx2 = Math.floor(Math.random() * candidates.length);
                while (idx2 === idx1) idx2 = Math.floor(Math.random() * candidates.length);

                zapRef.current = {
                    sourceId: candidates[idx1].id,
                    targetId: candidates[idx2].id,
                    progress: 0
                };
            }

            // Animate progress
            const interval = setInterval(() => {
                if (zapRef.current) {
                    zapRef.current.progress += 0.12;
                    if (zapRef.current.progress >= 1) {
                        clearInterval(interval);
                        setTimeout(() => { zapRef.current = null; }, 100);
                    }
                }
            }, 25);
        };

        const scheduleZap = () => {
            const delay = 2000 + Math.random() * 3000;
            return setTimeout(() => {
                triggerZap();
                zapTimeout = scheduleZap();
            }, delay);
        };

        let zapTimeout = scheduleZap();
        setTimeout(triggerZap, 2500); // Initial zap

        return () => clearTimeout(zapTimeout);
    }, [data.nodes, data.links]);

    // Node rendering (same quality as GraphPage)
    const nodeCanvasObject = useCallback((node: any, ctx: CanvasRenderingContext2D, globalScale: number) => {
        if (typeof node.x !== 'number' || typeof node.y !== 'number') return;

        const label = node.name;
        const fontSize = 11 / globalScale;
        const size = node.val;

        // Check if this node is part of active zap
        const zap = zapRef.current;
        const isZapSource = zap && zap.sourceId === node.id;
        const isZapTarget = zap && zap.targetId === node.id && zap.progress > 0.8;

        if (node.img) {
            ctx.save();
            ctx.beginPath();
            ctx.arc(node.x, node.y, size, 0, 2 * Math.PI, false);
            ctx.fillStyle = node.color || '#a78bfa';
            ctx.fill();
            ctx.clip();

            try {
                const imgWidth = node.img.width;
                const imgHeight = node.img.height;
                const circleDiameter = size * 2;
                const scale = Math.max(circleDiameter / imgWidth, circleDiameter / imgHeight);
                const scaledWidth = imgWidth * scale;
                const scaledHeight = imgHeight * scale;
                const offsetX = (circleDiameter - scaledWidth) / 2;
                const offsetY = (circleDiameter - scaledHeight) / 2;

                ctx.drawImage(
                    node.img,
                    node.x - size + offsetX,
                    node.y - size + offsetY,
                    scaledWidth,
                    scaledHeight
                );
            } catch (e) {
                ctx.fillStyle = node.color;
                ctx.fill();
            }

            ctx.restore();
        } else {
            ctx.beginPath();
            ctx.arc(node.x, node.y, size, 0, 2 * Math.PI, false);
            ctx.fillStyle = node.color || '#a78bfa';
            ctx.fill();
        }

        // Border (glow if zap target)
        ctx.beginPath();
        ctx.arc(node.x, node.y, size, 0, 2 * Math.PI, false);
        ctx.lineWidth = (isZapTarget ? 3 : 1.5) / globalScale;
        ctx.strokeStyle = isZapTarget ? '#fbbf24' : (node.color || '#8b5cf6');
        ctx.stroke();

        // Label
        ctx.font = `${fontSize}px Sans-Serif`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillStyle = 'rgba(255, 255, 255, 0.85)';
        ctx.fillText(label, node.x, node.y + size + fontSize + 2);

        // Draw zap from this node if it's the source
        if (isZapSource && zap) {
            const targetNode = data.nodes.find(n => n.id === zap.targetId);
            if (targetNode && typeof targetNode.x === 'number' && typeof targetNode.y === 'number') {
                drawLightningBolt(ctx, node.x, node.y, targetNode.x, targetNode.y, zap.progress, globalScale);
            }
        }
    }, [data.nodes]);

    const linkColor = useCallback((link: any) => link.color || 'rgba(148, 163, 184, 0.3)', []);
    const linkWidth = useCallback((link: any) => link.width || 1, []);

    return (
        <div className="fixed inset-0 bg-[#0a0a0f] overflow-hidden">
            {/* Background "HERE" text - ABOVE the graph with pointer-events-none */}
            <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none select-none z-20">
                <div className="text-center">
                    <div className="inline-flex items-center px-5 py-2 mb-6 rounded-full text-sm uppercase tracking-widest text-indigo-300/60 border border-white/10 bg-black/30 backdrop-blur-sm">
                        <span className="w-2 h-2 bg-indigo-400 rounded-full mr-3 animate-pulse"></span>
                        System Breathing
                    </div>

                    <h1 className="text-[10rem] md:text-[16rem] font-black tracking-tighter leading-none text-white/10">
                        HERE
                    </h1>

                    <p className="text-xl md:text-2xl text-slate-400/50 font-light -mt-6 md:-mt-10">
                        a breathing knowledge system
                    </p>

                    <p className="text-base text-slate-500/40 mt-3 italic">
                        (coming soon...)
                    </p>
                </div>
            </div>

            {/* Graph */}
            <div ref={containerRef} className="absolute inset-0 z-10">
                {loading ? (
                    <div className="absolute inset-0 flex items-center justify-center">
                        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-400"></div>
                    </div>
                ) : data.nodes.length === 0 ? (
                    <div className="absolute inset-0 flex items-center justify-center text-slate-500">
                        Loading...
                    </div>
                ) : (
                    <ForceGraph2D
                        ref={graphRef}
                        width={dimensions.width}
                        height={dimensions.height}
                        graphData={data}
                        nodeCanvasObject={nodeCanvasObject}
                        nodePointerAreaPaint={(node: any, color, ctx) => {
                            ctx.fillStyle = color;
                            ctx.beginPath();
                            ctx.arc(node.x, node.y, node.val, 0, 2 * Math.PI, false);
                            ctx.fill();
                        }}
                        onNodeClick={() => {}}
                        backgroundColor="transparent"
                        linkColor={linkColor}
                        linkWidth={linkWidth}
                        d3AlphaDecay={0.01}
                        d3VelocityDecay={0.2}
                        cooldownTicks={Infinity}
                        cooldownTime={Infinity}
                        nodeRelSize={1}
                        enableZoomInteraction={true}
                        enablePanInteraction={true}
                        enableNodeDrag={true}
                    />
                )}
            </div>
        </div>
    );
};

export default LandingPage;
