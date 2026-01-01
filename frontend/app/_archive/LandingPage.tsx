import React, { useEffect, useState, useRef, useCallback } from 'react';
import ForceGraph2D from 'react-force-graph-2d';

interface Entity {
    id: string;
    canonical_name: string;
    entity_type?: string;
    image_url?: string;
    mention_count?: number; // Total mentions from API
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
    mentionCount: number; // Total mentions across all events
    mentionRatio: number; // 0-1 ratio vs max mentions
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

    // Track loaded images separately to preserve them across updates
    const imageCache = useRef<Map<string, HTMLImageElement>>(new Map());

    // Fetch data
    useEffect(() => {
        const fetchData = async (isInitial: boolean = false) => {
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

                const entityMap = new Map<string, { entity: Entity; eventIds: string[]; firstEventIndex: number; totalMentions: number }>();
                const totalEvents = validEventDetails.length;

                validEventDetails.forEach((result, eventIndex) => {
                    const event = result.event;
                    const eventId = event.event_id || event.id;

                    const entities = result.entities || [];
                    entities.forEach((entity: Entity) => {
                        if (entity.id && entity.canonical_name) {
                            const mentionCount = entity.mention_count || 1;
                            if (!entityMap.has(entity.id)) {
                                // First appearance - eventIndex 0 is most recent
                                entityMap.set(entity.id, { entity, eventIds: [], firstEventIndex: eventIndex, totalMentions: 0 });
                            }
                            const entry = entityMap.get(entity.id)!;
                            entry.eventIds.push(eventId);
                            entry.totalMentions += mentionCount; // Accumulate mentions across events
                        }
                    });
                });

                // Sort by importance - now using totalMentions
                let maxMentions = 1;
                entityMap.forEach(({ totalMentions }) => {
                    maxMentions = Math.max(maxMentions, totalMentions);
                });

                const sortedEntities = Array.from(entityMap.entries())
                    .map(([id, { entity, eventIds, firstEventIndex, totalMentions }]) => ({
                        id,
                        entity,
                        eventIds,
                        firstEventIndex,
                        totalMentions,
                        // Score includes mentions and recency bonus
                        score: totalMentions * 3 + eventIds.length * 2 + (entity.image_url ? 5 : 0) + (entity.entity_type === 'PERSON' ? 3 : 0) + (totalEvents - firstEventIndex)
                    }))
                    .sort((a, b) => b.score - a.score)
                    .slice(0, MAX_ENTITIES);

                // Build nodes with recency-based positioning
                const centerX = dimensions.width / 2;
                const centerY = dimensions.height / 2;
                const maxRadius = Math.min(dimensions.width, dimensions.height) * 0.4;

                // Get existing node positions from the actual graph (not React state)
                const existingPositions = new Map<string, { x: number; y: number; vx?: number; vy?: number }>();
                if (graphRef.current && !isInitial) {
                    try {
                        const fg = graphRef.current;
                        const currentData = typeof fg.graphData === 'function' ? fg.graphData() : null;
                        if (currentData && currentData.nodes) {
                            currentData.nodes.forEach((n: any) => {
                                if (typeof n.x === 'number' && typeof n.y === 'number') {
                                    existingPositions.set(n.id, { x: n.x, y: n.y, vx: n.vx, vy: n.vy });
                                }
                            });
                        }
                    } catch (e) {
                        console.log('Could not get existing positions');
                    }
                }

                const nodes: GraphNode[] = sortedEntities.map(({ id, entity, eventIds, firstEventIndex, totalMentions }) => {
                    const mentionRatio = totalMentions / maxMentions;
                    const hasImage = !!entity.image_url;
                    // Size based on mentions - more mentions = bigger
                    const nodeSize = hasImage ? (16 + mentionRatio * 28) : (5 + mentionRatio * 10);
                    const color = COLORS[entity.entity_type || 'DEFAULT'] || COLORS.DEFAULT;

                    // Recency: 0 = most recent, 1 = oldest
                    const recency = totalEvents > 1 ? firstEventIndex / (totalEvents - 1) : 0;

                    const existingPos = existingPositions.get(id);
                    const cachedImg = imageCache.current.get(id);

                    if (existingPos) {
                        // Entity exists - preserve exact position from graph
                        return {
                            id,
                            name: entity.canonical_name,
                            type: 'entity' as const,
                            val: nodeSize,
                            color,
                            imgUrl: entity.image_url,
                            img: cachedImg,
                            eventIds,
                            recency,
                            mentionCount: totalMentions,
                            mentionRatio,
                            x: existingPos.x,
                            y: existingPos.y,
                            vx: existingPos.vx,
                            vy: existingPos.vy
                        };
                    } else {
                        // New entity - position based on recency (recent = closer to center)
                        const angle = Math.random() * Math.PI * 2;
                        // High mentions + recent = close to center
                        const radiusFactor = 0.1 + (1 - mentionRatio) * 0.4 + recency * 0.5;
                        const radius = radiusFactor * maxRadius;
                        const initialX = centerX + Math.cos(angle) * radius;
                        const initialY = centerY + Math.sin(angle) * radius;

                        return {
                            id,
                            name: entity.canonical_name,
                            type: 'entity' as const,
                            val: nodeSize,
                            color,
                            imgUrl: entity.image_url,
                            img: cachedImg,
                            eventIds,
                            recency,
                            mentionCount: totalMentions,
                            mentionRatio,
                            x: initialX,
                            y: initialY
                        };
                    }
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

                // Pre-load images (use cache)
                nodes.forEach(node => {
                    if (node.imgUrl && !imageCache.current.has(node.id)) {
                        const img = new Image();
                        img.crossOrigin = 'anonymous';
                        img.src = node.imgUrl;
                        img.onload = () => {
                            imageCache.current.set(node.id, img);
                            node.img = img;
                        };
                    } else if (imageCache.current.has(node.id)) {
                        node.img = imageCache.current.get(node.id);
                    }
                });

                if (isInitial) {
                    // Initial load - set React state to trigger render
                    setData({ nodes, links });
                    console.log(`Initial load: ${nodes.length} entities, ${links.length} links`);
                } else {
                    // Refresh - check for new/changed data
                    const fg = graphRef.current;
                    if (fg && typeof fg.graphData === 'function') {
                        const currentData = fg.graphData();
                        const currentNodeMap = new Map(currentData.nodes.map((n: GraphNode) => [n.id, n]));
                        const newNodeIds = new Set(nodes.map(n => n.id));

                        let addedCount = 0;
                        let removedCount = 0;
                        let updatedCount = 0;

                        // Update existing nodes, track new ones
                        const nodesToAdd: GraphNode[] = [];
                        nodes.forEach(newNode => {
                            const existing = currentNodeMap.get(newNode.id) as GraphNode | undefined;
                            if (existing) {
                                // Update properties but keep position/velocity
                                existing.val = newNode.val;
                                existing.color = newNode.color;
                                existing.mentionCount = newNode.mentionCount;
                                existing.mentionRatio = newNode.mentionRatio;
                                existing.eventIds = newNode.eventIds;
                                existing.recency = newNode.recency;
                                if (newNode.img) existing.img = newNode.img;
                                updatedCount++;
                            } else {
                                // New node
                                nodesToAdd.push(newNode);
                                addedCount++;
                            }
                        });

                        // Count removed nodes
                        currentData.nodes.forEach((n: GraphNode) => {
                            if (!newNodeIds.has(n.id)) removedCount++;
                        });

                        // If there are new or removed nodes, update via graphData() method
                        if (addedCount > 0 || removedCount > 0) {
                            // Filter out removed nodes
                            const updatedNodes = currentData.nodes.filter((n: GraphNode) => newNodeIds.has(n.id));
                            // Add new nodes
                            nodesToAdd.forEach(n => updatedNodes.push(n));

                            // Use graphData setter to update - this properly informs the graph
                            fg.graphData({ nodes: updatedNodes, links: links });

                            console.log(`Refreshed: +${addedCount} new, -${removedCount} removed, ${updatedCount} updated`);
                        } else {
                            console.log(`Refreshed: ${updatedCount} entities updated (no structural changes)`);
                        }
                    }
                }

            } catch (error) {
                console.error('Error fetching graph data:', error);
            } finally {
                setLoading(false);
            }
        };

        fetchData(true); // Initial load

        // Periodic refresh - preserve positions
        const refreshInterval = setInterval(() => fetchData(false), 60000);
        return () => clearInterval(refreshInterval);
    }, [dimensions.width, dimensions.height]);

    // Configure forces - spread out with more distance
    useEffect(() => {
        if (graphRef.current && data.nodes.length > 0) {
            const fg = graphRef.current;

            // Strong repulsion to spread nodes far apart
            fg.d3Force('charge').strength(-500);
            // Longer link distance
            fg.d3Force('link').distance(200).strength(0.2);
            // Weak center pull for all nodes
            fg.d3Force('center').strength(0.02);
        }
    }, [data.nodes.length]);

    // Store data in a ref so ghost dragger can access latest without re-running effect
    const dataRef = useRef(data);
    dataRef.current = data;

    // Simulate random entity dragging - like a ghost user moving nodes around
    useEffect(() => {
        let isRunning = true;
        let animationFrame: number;
        let dragTimeout: NodeJS.Timeout | null = null;

        // Current drag state
        let currentDrag: {
            node: any;
            startX: number;
            startY: number;
            targetX: number;
            targetY: number;
            progress: number;
        } | null = null;

        const startDrag = () => {
            if (!isRunning) return;

            const fg = graphRef.current;
            if (!fg) {
                console.log('Ghost drag: graph not ready, retrying...');
                dragTimeout = setTimeout(startDrag, 1000);
                return;
            }

            // Use nodes from React state via ref (they have x,y positions after initial render)
            const currentData = dataRef.current;
            if (!currentData || currentData.nodes.length < 2) {
                console.log('Ghost drag: no data yet, retrying...');
                dragTimeout = setTimeout(startDrag, 1000);
                return;
            }

            // The nodes in React state should have positions after ForceGraph processes them
            // But we need the actual graph nodes which have live positions
            let nodes = currentData.nodes;

            // Try to get live positions from graph
            try {
                const gd = fg.graphData();
                if (gd && gd.nodes && gd.nodes.length > 0) {
                    nodes = gd.nodes;
                    console.log('Ghost drag: got live nodes from graph');
                }
            } catch (e) {
                // Use state nodes
            }

            // Pick random node (weighted by mentions)
            const weights = nodes.map((n: any) => 0.3 + (n.mentionRatio || 0) * 0.7);
            const totalWeight = weights.reduce((a: number, b: number) => a + b, 0);
            let r = Math.random() * totalWeight;
            let picked = nodes[0];
            for (let i = 0; i < nodes.length; i++) {
                r -= weights[i];
                if (r <= 0) {
                    picked = nodes[i];
                    break;
                }
            }

            // Get current position (might be undefined initially)
            const startX = typeof picked.x === 'number' ? picked.x : dimensions.width / 2;
            const startY = typeof picked.y === 'number' ? picked.y : dimensions.height / 2;

            // Random movement direction and distance
            const angle = Math.random() * Math.PI * 2;
            const dist = 50 + Math.random() * 100;

            // Bias toward center for high-mention nodes
            const cx = dimensions.width / 2;
            const cy = dimensions.height / 2;
            const biasX = (cx - startX) * 0.2 * (picked.mentionRatio || 0);
            const biasY = (cy - startY) * 0.2 * (picked.mentionRatio || 0);

            const targetX = startX + Math.cos(angle) * dist + biasX;
            const targetY = startY + Math.sin(angle) * dist + biasY;

            currentDrag = {
                node: picked,
                startX,
                startY,
                targetX,
                targetY,
                progress: 0
            };

            console.log(`👻 Dragging: ${picked.name || picked.id}`);
        };

        const animate = () => {
            if (!isRunning) return;

            if (currentDrag) {
                const prevX = currentDrag.node.x ?? currentDrag.startX;
                const prevY = currentDrag.node.y ?? currentDrag.startY;

                currentDrag.progress += 0.008;

                if (currentDrag.progress >= 1) {
                    // Release node
                    currentDrag.node.fx = undefined;
                    currentDrag.node.fy = undefined;
                    currentDrag = null;

                    // Schedule next drag
                    const delay = 2000 + Math.random() * 2000;
                    dragTimeout = setTimeout(startDrag, delay);
                } else {
                    // Smooth easing
                    const t = currentDrag.progress;
                    const ease = t < 0.5
                        ? 4 * t * t * t
                        : 1 - Math.pow(-2 * t + 2, 3) / 2;

                    const x = currentDrag.startX + (currentDrag.targetX - currentDrag.startX) * ease;
                    const y = currentDrag.startY + (currentDrag.targetY - currentDrag.startY) * ease;

                    // Calculate movement delta
                    const dx = x - prevX;
                    const dy = y - prevY;

                    // Update dragged node position
                    currentDrag.node.fx = x;
                    currentDrag.node.fy = y;
                    currentDrag.node.x = x;
                    currentDrag.node.y = y;

                    // Push nearby nodes (simulate force)
                    const currentData = dataRef.current;
                    if (currentData && currentData.nodes && (Math.abs(dx) > 0.1 || Math.abs(dy) > 0.1)) {
                        const draggedId = currentDrag.node.id;
                        currentData.nodes.forEach((node: any) => {
                            if (node.id === draggedId || node.fx !== undefined) return;

                            const nodeX = node.x ?? 0;
                            const nodeY = node.y ?? 0;
                            const distX = nodeX - x;
                            const distY = nodeY - y;
                            const dist = Math.sqrt(distX * distX + distY * distY);

                            // Push nodes within 150px radius
                            if (dist > 10 && dist < 150) {
                                const force = (150 - dist) / 150 * 0.3; // Stronger when closer
                                const pushX = (distX / dist) * force * Math.abs(dx) * 2;
                                const pushY = (distY / dist) * force * Math.abs(dy) * 2;

                                node.x = nodeX + pushX;
                                node.y = nodeY + pushY;
                            }
                        });
                    }
                }
            }

            animationFrame = requestAnimationFrame(animate);
        };

        // Start animation loop immediately
        animationFrame = requestAnimationFrame(animate);

        // Start first drag after 4 seconds
        dragTimeout = setTimeout(startDrag, 4000);
        console.log('👻 Ghost dragger initialized, first drag in 4s');

        return () => {
            isRunning = false;
            if (dragTimeout) clearTimeout(dragTimeout);
            cancelAnimationFrame(animationFrame);
        };
    }, [dimensions.width, dimensions.height]);

    // Weighted random selection - higher mention entities get picked more often
    const weightedRandomPick = useCallback((nodes: GraphNode[]): GraphNode => {
        // Weight by mentionRatio - entities with more mentions are more likely to be picked
        const weights = nodes.map(n => 0.2 + (n.mentionRatio || 0) * 0.8);
        const totalWeight = weights.reduce((a, b) => a + b, 0);
        let random = Math.random() * totalWeight;
        for (let i = 0; i < nodes.length; i++) {
            random -= weights[i];
            if (random <= 0) return nodes[i];
        }
        return nodes[nodes.length - 1];
    }, []);

    // Trigger zaps
    useEffect(() => {
        if (data.nodes.length < 2) return;

        const triggerZap = () => {
            // Pick from linked pairs preferentially (70% of the time)
            if (data.links.length > 0 && Math.random() < 0.7) {
                // Weight links by the combined mention count of their nodes
                const linksWithWeight = data.links.map(link => {
                    const sourceId = typeof link.source === 'string' ? link.source : link.source.id;
                    const targetId = typeof link.target === 'string' ? link.target : link.target.id;
                    const sourceNode = data.nodes.find(n => n.id === sourceId);
                    const targetNode = data.nodes.find(n => n.id === targetId);
                    const weight = ((sourceNode?.mentionRatio || 0) + (targetNode?.mentionRatio || 0)) / 2 + 0.3;
                    return { link, sourceId, targetId, weight };
                });

                const totalWeight = linksWithWeight.reduce((a, b) => a + b.weight, 0);
                let random = Math.random() * totalWeight;
                let selected = linksWithWeight[0];
                for (const lw of linksWithWeight) {
                    random -= lw.weight;
                    if (random <= 0) {
                        selected = lw;
                        break;
                    }
                }

                zapRef.current = { sourceId: selected.sourceId, targetId: selected.targetId, progress: 0 };
            } else {
                // Random pair - weighted by mentions
                const withImages = data.nodes.filter(n => n.img);
                const candidates = withImages.length >= 2 ? withImages : data.nodes;
                const node1 = weightedRandomPick(candidates);
                let node2 = weightedRandomPick(candidates.filter(n => n.id !== node1.id));

                zapRef.current = {
                    sourceId: node1.id,
                    targetId: node2.id,
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
            // Zap more frequently when there are high-mention entities
            const hasHighMention = data.nodes.some(n => n.mentionRatio > 0.5);
            const delay = hasHighMention ? (1500 + Math.random() * 2000) : (2000 + Math.random() * 3000);
            return setTimeout(() => {
                triggerZap();
                zapTimeout = scheduleZap();
            }, delay);
        };

        let zapTimeout = scheduleZap();
        setTimeout(triggerZap, 2500); // Initial zap

        return () => clearTimeout(zapTimeout);
    }, [data.nodes, data.links, weightedRandomPick]);

    // Animation time for pulse effects
    const animTimeRef = useRef(0);
    useEffect(() => {
        const animate = () => {
            animTimeRef.current = Date.now() / 1000;
            animFrameRef.current = requestAnimationFrame(animate);
        };
        const animFrameRef = { current: 0 };
        animFrameRef.current = requestAnimationFrame(animate);
        return () => cancelAnimationFrame(animFrameRef.current);
    }, []);

    // Node rendering (same quality as GraphPage)
    const nodeCanvasObject = useCallback((node: any, ctx: CanvasRenderingContext2D, globalScale: number) => {
        if (typeof node.x !== 'number' || typeof node.y !== 'number') return;

        const label = node.name;
        const fontSize = 11 / globalScale;
        const size = node.val;
        const mentionRatio = node.mentionRatio || 0;

        // Check if this node is part of active zap
        const zap = zapRef.current;
        const isZapSource = zap && zap.sourceId === node.id;
        const isZapTarget = zap && zap.targetId === node.id && zap.progress > 0.8;

        // Subtle pulse glow for high-mention entities
        const isHighMention = mentionRatio > 0.4;
        const pulseIntensity = isHighMention ? 0.3 + Math.sin(animTimeRef.current * 2 + node.id.charCodeAt(0)) * 0.2 : 0;

        // Draw outer glow for high-mention entities
        if (isHighMention && !isZapTarget) {
            const glowSize = size + 4 / globalScale;
            const gradient = ctx.createRadialGradient(node.x, node.y, size * 0.8, node.x, node.y, glowSize);
            gradient.addColorStop(0, `rgba(251, 191, 36, ${pulseIntensity * mentionRatio})`);
            gradient.addColorStop(1, 'rgba(251, 191, 36, 0)');
            ctx.beginPath();
            ctx.arc(node.x, node.y, glowSize, 0, 2 * Math.PI, false);
            ctx.fillStyle = gradient;
            ctx.fill();
        }

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

        // Border (glow if zap target, golden for high mention)
        ctx.beginPath();
        ctx.arc(node.x, node.y, size, 0, 2 * Math.PI, false);
        const borderWidth = isZapTarget ? 3 : (isHighMention ? 2 : 1.5);
        ctx.lineWidth = borderWidth / globalScale;
        ctx.strokeStyle = isZapTarget ? '#fbbf24' : (isHighMention ? `rgba(251, 191, 36, ${0.5 + pulseIntensity})` : (node.color || '#8b5cf6'));
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
                    <h1 className="text-[10rem] md:text-[16rem] font-black tracking-tighter leading-none text-white/10">
                        HERE
                    </h1>

                    <p className="text-xl md:text-2xl text-slate-400/50 font-light -mt-6 md:-mt-10">
                        The first breathing knowledge system
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
                        onNodeDragEnd={(node: any) => {
                            // Fix the node position after drag so it doesn't bounce back
                            node.fx = node.x;
                            node.fy = node.y;
                            // Unfix after a short delay to allow organic movement later
                            setTimeout(() => {
                                node.fx = undefined;
                                node.fy = undefined;
                            }, 3000);
                        }}
                        backgroundColor="transparent"
                        linkColor={linkColor}
                        linkWidth={linkWidth}
                        d3AlphaDecay={0.01}
                        d3VelocityDecay={0.3}
                        cooldownTicks={Infinity}
                        cooldownTime={Infinity}
                        warmupTicks={0}
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
