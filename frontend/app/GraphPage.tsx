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
    type: 'entity' | 'event';
    val: number;
    color?: string;
    imgUrl?: string;
    img?: HTMLImageElement;
    data?: any;
    eventIds?: string[];
    x?: number;
    y?: number;
}

interface GraphLink {
    source: string;
    target: string;
    color?: string;
    width?: number;
}

interface GraphData {
    nodes: GraphNode[];
    links: GraphLink[];
}

const GraphPage: React.FC = () => {
    const [data, setData] = useState<GraphData>({ nodes: [], links: [] });
    const [loading, setLoading] = useState(true);
    const navigate = useNavigate();
    const graphRef = useRef<any>();
    const containerRef = useRef<HTMLDivElement>(null);
    const [dimensions, setDimensions] = useState({ width: 800, height: 600 });

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

                // Add entity nodes - filter to entities appearing in multiple events or persons
                entityMap.forEach(({ entity, eventIds }, entityId) => {
                    const isPerson = entity.entity_type === 'PERSON';
                    const isImportant = eventIds.length >= 2 || isPerson;

                    if (isImportant) {
                        const baseSize = isPerson ? 25 : 18;
                        const bonus = Math.min(eventIds.length * 5, 25);

                        // Color by entity type
                        let color = '#a78bfa'; // Purple for people
                        if (entity.entity_type === 'ORGANIZATION') color = '#f59e0b'; // Orange
                        if (entity.entity_type === 'LOCATION') color = '#10b981'; // Green

                        nodes.push({
                            id: entityId,
                            name: entity.canonical_name,
                            type: 'entity',
                            val: baseSize + bonus,
                            color,
                            imgUrl: entity.image_url,
                            data: entity,
                            eventIds
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
                    <span className="text-gray-500 text-xs">(click any node to view details)</span>
                </div>
            </div>

            <div ref={containerRef} className="flex-1 relative overflow-hidden" style={{ minHeight: '500px' }}>
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
