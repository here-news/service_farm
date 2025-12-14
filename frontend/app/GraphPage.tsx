import React, { useEffect, useState, useRef, useCallback } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import { useNavigate } from 'react-router-dom';

interface Event {
    event_id: string;
    id: string;
    title: string;
    coherence?: number;
    status?: string;
}

interface Entity {
    id: string;
    canonical_name: string;
    entity_type?: string;
    wikidata_qid?: string;
    image_url?: string;
}

interface GraphNode {
    id: string;
    name: string;
    type: 'entity' | 'event' | 'me';
    val: number;
    color?: string;
    imgUrl?: string;
    img?: HTMLImageElement;
    data?: any;
    eventIds?: string[];
    mentionCount?: number;
    x?: number;
    y?: number;
    fx?: number; // Fixed x position
    fy?: number; // Fixed y position
}

interface GraphLink {
    source: string;
    target: string;
    color?: string;
    width?: number;
    isZap?: boolean;
}

interface GraphData {
    nodes: GraphNode[];
    links: GraphLink[];
}

// Draw a lightning bolt effect between two points
function drawLightningBolt(
    ctx: CanvasRenderingContext2D,
    x1: number, y1: number,
    x2: number, y2: number,
    progress: number,
    globalScale: number
) {
    const segments = 6;
    const jitter = 15 / globalScale;

    // Calculate points along the path with random offsets
    const points: { x: number; y: number }[] = [{ x: x1, y: y1 }];
    const dx = x2 - x1;
    const dy = y2 - y1;

    for (let i = 1; i < segments; i++) {
        const t = i / segments;
        // Only draw up to current progress
        if (t > progress) break;

        const baseX = x1 + dx * t;
        const baseY = y1 + dy * t;

        // Perpendicular offset for zigzag effect
        const perpX = -dy / Math.sqrt(dx * dx + dy * dy);
        const perpY = dx / Math.sqrt(dx * dx + dy * dy);
        const offset = (Math.random() - 0.5) * 2 * jitter;

        points.push({
            x: baseX + perpX * offset,
            y: baseY + perpY * offset
        });
    }

    // Add endpoint if we've progressed far enough
    if (progress >= 0.9) {
        points.push({ x: x2, y: y2 });
    }

    if (points.length < 2) return;

    // Draw the lightning bolt
    ctx.save();
    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);

    for (let i = 1; i < points.length; i++) {
        ctx.lineTo(points[i].x, points[i].y);
    }

    // Glow effect - thinner line
    ctx.shadowColor = '#fbbf24';
    ctx.shadowBlur = 6;
    ctx.strokeStyle = '#fef08a';
    ctx.lineWidth = 1.5 / globalScale;
    ctx.stroke();

    // Inner bright line
    ctx.shadowBlur = 0;
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 0.8 / globalScale;
    ctx.stroke();

    ctx.restore();
}

interface UserInfo {
    id: string;
    name: string;
    picture?: string;
}

const GraphPage: React.FC = () => {
    const [data, setData] = useState<GraphData>({ nodes: [], links: [] });
    const [loading, setLoading] = useState(true);
    const navigate = useNavigate();
    const graphRef = useRef<any>();
    const containerRef = useRef<HTMLDivElement>(null);
    const [dimensions, setDimensions] = useState({ width: 800, height: 600 });

    // Me node state - using refs to avoid React rerenders during animation
    const [userInfo, setUserInfo] = useState<UserInfo | null>(null);
    const meCanvasRef = useRef<HTMLCanvasElement>(null);
    const meAvatarImgRef = useRef<HTMLImageElement | null>(null);
    const entityNodesRef = useRef<GraphNode[]>([]);
    // Animation state stored in refs to avoid rerenders
    const meAnimRef = useRef({
        startTime: Date.now(),
        lastBlink: Date.now(),
        nextBlinkDelay: 3000 + Math.random() * 4000,
        zapTarget: null as string | null,
        zapProgress: 0
    });

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

    // Fetch user info for "Me" node avatar
    useEffect(() => {
        const fetchUser = async () => {
            try {
                // Try to get user from /api/events/mine endpoint which may have user info
                const response = await fetch('/api/auth/me', { credentials: 'include' });
                if (response.ok) {
                    const data = await response.json();
                    setUserInfo(data);
                    // Pre-load avatar image
                    if (data.picture) {
                        const img = new Image();
                        img.crossOrigin = 'anonymous';
                        img.src = data.picture;
                        img.onload = () => {
                            meAvatarImgRef.current = img;
                        };
                    }
                }
            } catch (err) {
                // No auth, use default
                console.log('No user auth, using default Me node');
            }
        };
        fetchUser();
    }, []);

    // Me node animation loop - draws directly to canvas without React state updates
    useEffect(() => {
        let animationFrame: number;
        const anim = meAnimRef.current;

        const drawMeNode = () => {
            const canvas = meCanvasRef.current;
            if (!canvas) {
                animationFrame = requestAnimationFrame(drawMeNode);
                return;
            }

            const ctx = canvas.getContext('2d');
            if (!ctx) {
                animationFrame = requestAnimationFrame(drawMeNode);
                return;
            }

            const now = Date.now();
            const elapsed = (now - anim.startTime) / 1000;

            // Calculate animation values
            const breathScale = 1 + Math.sin(elapsed * 1.2) * 0.04;
            const shakeX = Math.sin(elapsed * 0.6) * 1.5 + Math.sin(elapsed * 1.7) * 0.8;
            const shakeY = Math.cos(elapsed * 0.8) * 1.5 + Math.cos(elapsed * 1.4) * 0.8;
            const glowIntensity = 0.5 + Math.sin(elapsed * 2) * 0.2;

            // Eye blinking logic
            let eyeOpen = true;
            if (now - anim.lastBlink > anim.nextBlinkDelay) {
                if (now - anim.lastBlink < anim.nextBlinkDelay + 150) {
                    eyeOpen = false;
                } else {
                    anim.lastBlink = now;
                    anim.nextBlinkDelay = 3000 + Math.random() * 4000;
                }
            }

            // Clear canvas
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            // Position
            const baseSize = 35;
            const meX = 70 + shakeX;
            const meY = canvas.height - 100 + shakeY;
            const size = baseSize * breathScale;

            // Outer glow ring
            ctx.save();
            ctx.shadowColor = '#06b6d4';
            ctx.shadowBlur = 15 * glowIntensity;
            ctx.beginPath();
            ctx.arc(meX, meY, size + 5, 0, 2 * Math.PI);
            ctx.strokeStyle = `rgba(6, 182, 212, ${glowIntensity * 0.6})`;
            ctx.lineWidth = 2;
            ctx.stroke();
            ctx.restore();

            // Main circle
            ctx.save();
            ctx.beginPath();
            ctx.arc(meX, meY, size, 0, 2 * Math.PI);

            if (meAvatarImgRef.current) {
                ctx.clip();
                const img = meAvatarImgRef.current;
                const imgSize = size * 2;
                ctx.drawImage(img, meX - size, meY - size, imgSize, imgSize);
            } else {
                // Default gradient with face
                const gradient = ctx.createRadialGradient(meX, meY - size * 0.3, 0, meX, meY, size);
                gradient.addColorStop(0, '#67e8f9');
                gradient.addColorStop(1, '#0891b2');
                ctx.fillStyle = gradient;
                ctx.fill();
                ctx.restore();
                ctx.save();

                // Eyes
                const eyeY = meY - 5;
                const eyeSpacing = 10;

                if (eyeOpen) {
                    ctx.fillStyle = '#1e3a5f';
                    ctx.beginPath();
                    ctx.arc(meX - eyeSpacing, eyeY, 4, 0, 2 * Math.PI);
                    ctx.arc(meX + eyeSpacing, eyeY, 4, 0, 2 * Math.PI);
                    ctx.fill();

                    ctx.fillStyle = '#ffffff';
                    ctx.beginPath();
                    ctx.arc(meX - eyeSpacing + 1.5, eyeY - 1.5, 1.5, 0, 2 * Math.PI);
                    ctx.arc(meX + eyeSpacing + 1.5, eyeY - 1.5, 1.5, 0, 2 * Math.PI);
                    ctx.fill();
                } else {
                    ctx.strokeStyle = '#1e3a5f';
                    ctx.lineWidth = 2;
                    ctx.beginPath();
                    ctx.moveTo(meX - eyeSpacing - 4, eyeY);
                    ctx.lineTo(meX - eyeSpacing + 4, eyeY);
                    ctx.moveTo(meX + eyeSpacing - 4, eyeY);
                    ctx.lineTo(meX + eyeSpacing + 4, eyeY);
                    ctx.stroke();
                }

                // Smile
                ctx.strokeStyle = '#1e3a5f';
                ctx.lineWidth = 2;
                ctx.lineCap = 'round';
                ctx.beginPath();
                ctx.arc(meX, meY + 2, 8, 0.2 * Math.PI, 0.8 * Math.PI);
                ctx.stroke();
            }
            ctx.restore();

            // Border
            ctx.beginPath();
            ctx.arc(meX, meY, size, 0, 2 * Math.PI);
            ctx.strokeStyle = '#67e8f9';
            ctx.lineWidth = 2.5;
            ctx.stroke();

            // Labels
            ctx.textAlign = 'center';
            ctx.font = 'bold 13px Sans-Serif';
            ctx.fillStyle = '#e2e8f0';
            ctx.fillText('Me', meX, meY + size + 14);

            // Lightning zap
            if (anim.zapTarget && anim.zapProgress > 0 && graphRef.current) {
                const targetNode = data.nodes.find(n => n.id === anim.zapTarget);
                if (targetNode && typeof targetNode.x === 'number' && typeof targetNode.y === 'number') {
                    const screenCoords = graphRef.current.graph2ScreenCoords(targetNode.x, targetNode.y);
                    if (screenCoords) {
                        drawLightningBolt(ctx, meX, meY, screenCoords.x, screenCoords.y, anim.zapProgress, 1);
                    }
                }
            }

            animationFrame = requestAnimationFrame(drawMeNode);
        };

        drawMeNode();
        return () => cancelAnimationFrame(animationFrame);
    }, [dimensions, data.nodes]);

    // Lightning zap effect - trigger occasionally when we have entities
    // Uses ref to avoid React rerenders
    useEffect(() => {
        if (entityNodesRef.current.length === 0) return;

        const anim = meAnimRef.current;

        const triggerZap = () => {
            const entities = entityNodesRef.current.filter(n => n.type === 'entity' && n.img);
            if (entities.length === 0) return;

            // Pick a random entity (preferring those with images)
            const target = entities[Math.floor(Math.random() * entities.length)];
            anim.zapTarget = target.id;
            anim.zapProgress = 0;

            // Animate zap progress using ref (no React rerenders)
            const zapAnimation = setInterval(() => {
                anim.zapProgress += 0.1;
                if (anim.zapProgress >= 1) {
                    clearInterval(zapAnimation);
                    setTimeout(() => {
                        anim.zapTarget = null;
                        anim.zapProgress = 0;
                    }, 200);
                }
            }, 30);
        };

        // Trigger zap every 4-8 seconds
        const scheduleNextZap = () => {
            const delay = 4000 + Math.random() * 4000;
            return setTimeout(() => {
                triggerZap();
                zapTimeoutRef = scheduleNextZap();
            }, delay);
        };

        let zapTimeoutRef = scheduleNextZap();
        return () => clearTimeout(zapTimeoutRef);
    }, [data.nodes]);

    useEffect(() => {
        const fetchData = async () => {
            try {
                // 1. Fetch events from coherence feed
                const feedResponse = await fetch('/api/coherence/feed?limit=30');
                const feedResult = await feedResponse.json();

                const events = feedResult.events || [];
                if (events.length === 0) {
                    setLoading(false);
                    return;
                }

                // 2. Fetch details for events to get entities
                const eventDetailsPromises = events.slice(0, 15).map((event: Event) =>
                    fetch(`/api/event/${event.event_id || event.id}`)
                        .then(res => res.json())
                        .catch(err => {
                            console.error(`Failed to fetch event ${event.event_id}:`, err);
                            return null;
                        })
                );

                const eventDetailsResults = await Promise.all(eventDetailsPromises);
                const validEventDetails = eventDetailsResults.filter(r => r && r.event);

                // 3. Extract all entities across all events
                const entityMap = new Map<string, { entity: Entity; eventIds: string[] }>();
                const eventMap = new Map<string, Event>();

                validEventDetails.forEach((result) => {
                    const event = result.event;
                    const eventId = event.event_id || event.id;
                    eventMap.set(eventId, event);

                    // Extract entities (people, orgs, locations)
                    const entities = result.entities || [];
                    entities.forEach((entity: Entity) => {
                        const entityId = entity.id;
                        const entityName = entity.canonical_name;

                        if (entityId && entityName) {
                            if (!entityMap.has(entityId)) {
                                entityMap.set(entityId, {
                                    entity: entity,
                                    eventIds: []
                                });
                            }
                            entityMap.get(entityId)!.eventIds.push(eventId);
                        }
                    });
                });

                // 4. Build graph data
                const nodes: GraphNode[] = [];
                const links: GraphLink[] = [];
                const addedNodeIds = new Set<string>();

                // Calculate mention count stats for size discrimination
                let maxMentions = 1;
                let minMentions = Infinity;
                entityMap.forEach(({ eventIds }) => {
                    maxMentions = Math.max(maxMentions, eventIds.length);
                    minMentions = Math.min(minMentions, eventIds.length);
                });

                // Add entity nodes - filter to entities appearing in multiple events or persons
                entityMap.forEach(({ entity, eventIds }, entityId) => {
                    const isPerson = entity.entity_type === 'PERSON';
                    const isImportant = eventIds.length >= 2 || isPerson;

                    if (isImportant) {
                        // Size discrimination: scale based on mention count
                        // Low mentions: 8-12px, High mentions: 25-40px
                        const mentionRatio = (eventIds.length - minMentions) / Math.max(1, maxMentions - minMentions);
                        const hasImage = !!entity.image_url;

                        // Entities with images get larger base size, no-image entities stay small
                        let nodeSize: number;
                        if (hasImage) {
                            // With image: 18-40px based on mentions
                            nodeSize = 18 + mentionRatio * 22;
                        } else {
                            // Without image: 6-14px (much smaller to reduce visual noise)
                            nodeSize = 6 + mentionRatio * 8;
                        }

                        // Color by entity type
                        let color = '#a78bfa'; // Purple for people
                        if (entity.entity_type === 'ORGANIZATION') color = '#f59e0b'; // Orange
                        if (entity.entity_type === 'LOCATION') color = '#10b981'; // Green

                        nodes.push({
                            id: entityId,
                            name: entity.canonical_name,
                            type: 'entity',
                            val: nodeSize,
                            color,
                            imgUrl: entity.image_url,
                            data: entity,
                            eventIds,
                            mentionCount: eventIds.length
                        });
                        addedNodeIds.add(entityId);
                    }
                });

                // Add event nodes
                eventMap.forEach((event, eventId) => {
                    nodes.push({
                        id: eventId,
                        name: event.title || 'Untitled Event',
                        type: 'event',
                        val: 1,
                        color: '#3b82f6', // Blue for events
                        data: event
                    });
                    addedNodeIds.add(eventId);
                });

                // Add links: entities -> events they appear in
                entityMap.forEach(({ eventIds }, entityId) => {
                    if (addedNodeIds.has(entityId)) {
                        eventIds.forEach(eventId => {
                            if (addedNodeIds.has(eventId)) {
                                links.push({
                                    source: entityId,
                                    target: eventId,
                                    color: 'rgba(167, 139, 250, 0.3)',
                                    width: 1
                                });
                            }
                        });
                    }
                });

                // Pre-load images
                nodes.forEach(node => {
                    if (node.imgUrl) {
                        const img = new Image();
                        img.src = node.imgUrl;
                        img.onload = () => {
                            node.img = img;
                        };
                    }
                });

                // Store entity nodes for zap targeting
                entityNodesRef.current = nodes.filter(n => n.type === 'entity');

                setData({ nodes, links });
                console.log(`GraphPage: Loaded ${nodes.length} nodes and ${links.length} links`);

            } catch (error) {
                console.error('Error fetching graph data:', error);
            } finally {
                setLoading(false);
            }
        };

        fetchData();
    }, []);

    const handleNodeClick = useCallback((node: any) => {
        if (node.type === 'event') {
            // Navigate to event page using full event ID
            const eventId = node.data?.event_id || node.id;
            navigate(`/event/${eventId}`);
        } else if (node.type === 'entity') {
            // Navigate to entity page
            const entityId = node.id;
            navigate(`/entity/${entityId}`);
        }
    }, [navigate]);

    // Configure force simulation
    useEffect(() => {
        if (graphRef.current && data.nodes.length > 0) {
            const fg = graphRef.current;

            fg.d3Force('charge').strength((node: any) => {
                return node.type === 'entity' ? -600 : -200;
            });

            fg.d3Force('link').distance(() => 100).strength(0.6);
        }
    }, [data.nodes]);

    const nodeCanvasObject = useCallback((node: any, ctx: CanvasRenderingContext2D, globalScale: number) => {
        if (typeof node.x !== 'number' || typeof node.y !== 'number' ||
            !isFinite(node.x) || !isFinite(node.y)) {
            return;
        }

        const label = node.name;
        const fontSize = 12 / globalScale;

        if (node.type === 'entity') {
            const size = node.val;

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

            // Border
            ctx.beginPath();
            ctx.arc(node.x, node.y, size, 0, 2 * Math.PI, false);
            ctx.lineWidth = 1.5 / globalScale;
            ctx.strokeStyle = node.color || '#8b5cf6';
            ctx.stroke();

            // Label
            ctx.font = `${fontSize}px Sans-Serif`;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
            ctx.fillText(label, node.x, node.y + size + fontSize + 2);

        } else if (node.type === 'event') {
            // Event node - text box
            ctx.font = `${fontSize * 0.8}px Sans-Serif`;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';

            const maxWidth = 150 / globalScale;
            const lineHeight = fontSize * 0.8 * 1.2;

            // Word wrap
            const words = label.split(' ');
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

            const displayLines = lines.slice(0, 2);
            if (lines.length > 2) {
                displayLines[1] = displayLines[1].substring(0, displayLines[1].length - 3) + '...';
            }

            // Calculate box size
            const padding = 6 / globalScale;
            let maxLineWidth = 0;
            for (const line of displayLines) {
                const metrics = ctx.measureText(line);
                maxLineWidth = Math.max(maxLineWidth, metrics.width);
            }
            const width = maxLineWidth + padding * 2;
            const height = displayLines.length * lineHeight + padding * 2;

            // Draw box
            ctx.fillStyle = 'rgba(30, 58, 138, 0.4)';
            ctx.fillRect(node.x - width / 2, node.y - height / 2, width, height);

            ctx.strokeStyle = 'rgba(59, 130, 246, 0.5)';
            ctx.lineWidth = 1 / globalScale;
            ctx.strokeRect(node.x - width / 2, node.y - height / 2, width, height);

            // Draw text
            ctx.fillStyle = 'rgba(255, 255, 255, 0.85)';
            let textY = node.y - ((displayLines.length - 1) * lineHeight) / 2;

            for (const line of displayLines) {
                ctx.fillText(line, node.x, textY);
                textY += lineHeight;
            }
        }
    }, [data.nodes]);

    const linkColor = useCallback((link: any) => {
        return link.color || 'rgba(167, 139, 250, 0.2)';
    }, []);

    const linkWidth = useCallback((link: any) => {
        return link.width || 1;
    }, []);

    return (
        <div className="h-screen flex flex-col bg-gray-900 text-white">
            <div className="p-4 border-b border-gray-800 flex justify-between items-center z-10 bg-gray-900">
                <h1 className="text-xl font-bold">Knowledge Graph: Entities & Events</h1>
                <div className="text-sm text-gray-400">
                    <span className="mr-4">
                        <span className="inline-block w-3 h-3 rounded-full bg-purple-400 mr-1"></span> People
                    </span>
                    <span className="mr-4">
                        <span className="inline-block w-3 h-3 rounded-full bg-orange-400 mr-1"></span> Organizations
                    </span>
                    <span className="mr-4">
                        <span className="inline-block w-3 h-3 rounded-full bg-green-400 mr-1"></span> Locations
                    </span>
                    <span className="mr-4">
                        <span className="inline-block w-3 h-3 bg-blue-500 mr-1"></span> Events
                    </span>
                    <span className="mr-4">
                        <span className="inline-block w-3 h-3 rounded-full bg-cyan-400 mr-1"></span> Me
                    </span>
                    <span className="text-gray-500 text-xs">(click any node to view details)</span>
                </div>
            </div>

            <div ref={containerRef} className="flex-1 relative overflow-hidden" style={{ minHeight: '500px' }}>
                {/* "Me" node overlay - stays fixed, doesn't move with graph */}
                <canvas
                    ref={meCanvasRef}
                    width={dimensions.width}
                    height={dimensions.height}
                    className="absolute inset-0 pointer-events-none z-10"
                    style={{ width: dimensions.width, height: dimensions.height }}
                />

                {loading ? (
                    <div className="absolute inset-0 flex items-center justify-center">
                        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-white"></div>
                    </div>
                ) : data.nodes.length === 0 ? (
                    <div className="absolute inset-0 flex items-center justify-center text-gray-400">
                        No data found to visualize.
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
                            if (node.type === 'entity') {
                                ctx.beginPath();
                                ctx.arc(node.x, node.y, node.val, 0, 2 * Math.PI, false);
                                ctx.fill();
                            } else {
                                const approxWidth = 120;
                                const approxHeight = 40;
                                ctx.fillRect(node.x - approxWidth / 2, node.y - approxHeight / 2, approxWidth, approxHeight);
                            }
                        }}
                        onNodeClick={handleNodeClick}
                        backgroundColor="#111827"
                        linkColor={linkColor}
                        linkWidth={linkWidth}
                        d3AlphaDecay={0.02}
                        d3VelocityDecay={0.3}
                        cooldownTicks={100}
                        nodeRelSize={1}
                    />
                )}
            </div>
        </div>
    );
};

export default GraphPage;
