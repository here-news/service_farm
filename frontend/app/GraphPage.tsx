import React, { useEffect, useState, useRef, useCallback } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import { useNavigate } from 'react-router-dom';

interface Story {
    story_id: string;
    title: string;
    coherence?: number;
    tcf_score?: number;
    cover_image?: string;
}

interface Person {
    id: string;
    name: string;
    canonical_id?: string;
    wikidata_thumbnail?: string;
}

interface GraphNode {
    id: string;
    name: string;
    type: 'person' | 'story';
    val: number;
    color?: string;
    imgUrl?: string;
    img?: HTMLImageElement;
    data?: any;
    x?: number;
    y?: number;
    storyIds?: string[]; // For people: which stories they appear in
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
                // 1. Fetch more stories to get better people overlap
                const feedResponse = await fetch('/api/coherence/feed?limit=30');
                const feedResult = await feedResponse.json();

                if (!feedResult.stories || feedResult.stories.length === 0) {
                    setLoading(false);
                    return;
                }

                const stories = feedResult.stories;

                // 2. Fetch details for more stories to find people overlaps
                const storyDetailsPromises = stories.slice(0, 20).map((story: Story) =>
                    fetch(`/api/stories/${story.story_id}`)
                        .then(res => res.json())
                        .catch(err => {
                            console.error(`Failed to fetch story ${story.story_id}:`, err);
                            return null;
                        })
                );

                const storyDetailsResults = await Promise.all(storyDetailsPromises);
                const validStoryDetails = storyDetailsResults.filter(r => r && r.story);

                // 3. Extract all people across all stories
                const peopleMap = new Map<string, { person: Person; storyIds: string[] }>();
                const storyMap = new Map<string, Story>();

                validStoryDetails.forEach((result) => {
                    const story = result.story;
                    storyMap.set(story.id, story);

                    // Extract people entities
                    const people = story.entities?.people || [];
                    people.forEach((person: any) => {
                        const personId = person.canonical_id || person.id;
                        const personName = person.canonical_name || person.name;

                        if (personId && personName) {
                            if (!peopleMap.has(personId)) {
                                peopleMap.set(personId, {
                                    person: {
                                        id: personId,
                                        name: personName,
                                        canonical_id: person.canonical_id,
                                        wikidata_thumbnail: person.wikidata_thumbnail
                                    },
                                    storyIds: []
                                });
                            }
                            peopleMap.get(personId)!.storyIds.push(story.id);
                        }
                    });
                });

                // 4. Build graph data: People as primary nodes, Stories as secondary
                const nodes: GraphNode[] = [];
                const links: GraphLink[] = [];
                const addedNodeIds = new Set<string>();

                // Add people nodes (primary, larger, circular)
                peopleMap.forEach(({ person, storyIds }, personId) => {
                    // Show all people with BIG circles (people are the focus!)
                    if (storyIds.length >= 1) {
                        // Make everyone BIG by default, then scale up for importance
                        const baseSize = 22; // Big baseline (was 12 - too small!)
                        const bonus = Math.min(storyIds.length * 6, 30); // 6 per story

                        nodes.push({
                            id: personId,
                            name: person.name,
                            type: 'person',
                            val: baseSize + bonus,
                            color: '#a78bfa', // Purple for people
                            imgUrl: person.wikidata_thumbnail,
                            data: person,
                            storyIds: storyIds
                        });
                        addedNodeIds.add(personId);
                    }
                });

                // Add story nodes (secondary, smaller, rectangular)
                storyMap.forEach((story, storyId) => {
                    nodes.push({
                        id: storyId,
                        name: story.title,
                        type: 'story',
                        val: 1, // Base size, actual size determined by text
                        color: '#3b82f6', // Blue for stories
                        data: story
                    });
                    addedNodeIds.add(storyId);
                });

                // Add links: people -> stories they appear in
                peopleMap.forEach(({ storyIds }, personId) => {
                    if (addedNodeIds.has(personId)) {
                        storyIds.forEach(storyId => {
                            if (addedNodeIds.has(storyId)) {
                                links.push({
                                    source: personId,
                                    target: storyId,
                                    color: 'rgba(167, 139, 250, 0.2)', // Purple, transparent
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
                console.log(`GraphPage: Loaded ${nodes.length} nodes (${Array.from(peopleMap.keys()).length} people, ${storyMap.size} stories) and ${links.length} links`);

            } catch (error) {
                console.error('Error fetching graph data:', error);
            } finally {
                setLoading(false);
            }
        };

        fetchData();
    }, []);

    const handleNodeClick = useCallback((node: any) => {
        // Only story boxes are clickable - navigate to story page
        if (node.type === 'story') {
            navigate(`/story/${node.id}`);
        }
        // People are not clickable - just decorative
    }, [navigate]);

    // Configure force simulation after graph loads
    useEffect(() => {
        if (graphRef.current && data.nodes.length > 0) {
            const fg = graphRef.current;

            // Moderate repulsion to spread people out without scattering too far
            fg.d3Force('charge').strength((node: any) => {
                return node.type === 'person' ? -800 : -100;
            });

            // Medium link distance - not too close, not too far
            fg.d3Force('link').distance(() => {
                return 120; // Medium distance for balanced spacing
            }).strength(0.8); // Moderate pull toward connected nodes
        }
    }, [data.nodes]);

    const nodeCanvasObject = useCallback((node: any, ctx: CanvasRenderingContext2D, globalScale: number) => {
        // Skip rendering if node position is not yet defined or invalid
        if (typeof node.x !== 'number' || typeof node.y !== 'number' ||
            !isFinite(node.x) || !isFinite(node.y)) {
            return;
        }

        const label = node.name;
        const fontSize = 12 / globalScale;

        // No highlighting - people are just visual, stories are interactive
        const opacity = 1.0;

        ctx.globalAlpha = opacity;

        if (node.type === 'person') {
            // --- Draw Person Node (Circle) ---
            const size = node.val;

            if (node.img) {
                // Draw circular image
                ctx.save();
                ctx.beginPath();
                ctx.arc(node.x, node.y, size, 0, 2 * Math.PI, false);
                ctx.fillStyle = node.color || '#a78bfa';
                ctx.fill();
                ctx.clip();

                try {
                    // Maintain aspect ratio - cover style
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
                    ctx.fillStyle = '#a78bfa';
                    ctx.fill();
                }

                ctx.restore();
            } else {
                // No image, just colored circle
                ctx.beginPath();
                ctx.arc(node.x, node.y, size, 0, 2 * Math.PI, false);
                ctx.fillStyle = node.color || '#a78bfa';
                ctx.fill();
            }

            // Border (consistent - no selection state)
            ctx.beginPath();
            ctx.arc(node.x, node.y, size, 0, 2 * Math.PI, false);
            ctx.lineWidth = 1.5 / globalScale;
            ctx.strokeStyle = '#8b5cf6'; // Purple
            ctx.stroke();

            // Label (always show for people)
            ctx.font = `${fontSize}px Sans-Serif`;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
            ctx.fillText(label, node.x, node.y + size + fontSize + 2);

        } else if (node.type === 'story') {
            // --- Draw Story Node (Text Only - No Box) ---
            // Small, faded text that doesn't obscure people
            ctx.font = `${fontSize * 0.7}px Sans-Serif`;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';

            // Measure text to size for hit detection
            const maxWidth = 150 / globalScale;
            const lineHeight = fontSize * 0.7 * 1.2;

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
            if (currentLine) {
                lines.push(currentLine);
            }

            // Limit to 2 lines for compactness
            const displayLines = lines.slice(0, 2);
            if (lines.length > 2) {
                displayLines[1] = displayLines[1].substring(0, displayLines[1].length - 3) + '...';
            }

            // Draw subtle box to show it's clickable
            const padding = 4 / globalScale;
            let maxLineWidth = 0;
            for (const line of displayLines) {
                const metrics = ctx.measureText(line);
                maxLineWidth = Math.max(maxLineWidth, metrics.width);
            }
            const width = maxLineWidth + padding * 2;
            const height = displayLines.length * lineHeight + padding * 2;

            // Very subtle box to indicate clickability
            ctx.fillStyle = 'rgba(30, 58, 138, 0.2)';
            ctx.fillRect(node.x - width / 2, node.y - height / 2, width, height);

            ctx.strokeStyle = 'rgba(59, 130, 246, 0.3)';
            ctx.lineWidth = 0.5 / globalScale;
            ctx.strokeRect(node.x - width / 2, node.y - height / 2, width, height);

            // Draw text - faded gray
            ctx.fillStyle = 'rgba(156, 163, 175, 0.7)';
            let textY = node.y - ((displayLines.length - 1) * lineHeight) / 2;

            for (const line of displayLines) {
                ctx.fillText(line, node.x, textY);
                textY += lineHeight;
            }
        }

        ctx.globalAlpha = 1.0;
    }, [data.nodes]);

    const linkColor = useCallback((link: any) => {
        // All links same color - no selection highlighting
        return link.color || 'rgba(167, 139, 250, 0.2)';
    }, []);

    const linkWidth = useCallback((link: any) => {
        // All links same width - no selection highlighting
        return link.width || 1;
    }, []);

    return (
        <div className="h-screen flex flex-col bg-gray-900 text-white">
            <div className="p-4 border-b border-gray-800 flex justify-between items-center z-10 bg-gray-900">
                <h1 className="text-xl font-bold">Knowledge Graph: People & Stories</h1>
                <div className="text-sm text-gray-400">
                    <span>
                        People (circles) • Stories (text boxes) • Click story to view details
                    </span>
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
                            if (node.type === 'person') {
                                ctx.beginPath();
                                ctx.arc(node.x, node.y, node.val, 0, 2 * Math.PI, false);
                                ctx.fill();
                            } else {
                                // Approximate hit area for story text boxes
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
