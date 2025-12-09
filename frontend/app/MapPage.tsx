import React, { useEffect, useState, useMemo } from 'react';
import { MapContainer, TileLayer, Marker, Popup, Polyline, useMap } from 'react-leaflet';
import { useNavigate } from 'react-router-dom';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

interface Story {
    id: string;
    title: string;
    coherence?: number;
    cover_image?: string;
}

interface Location {
    id: string;
    name: string;
    latitude: number;
    longitude: number;
    qid?: string;
    thumbnail?: string;
    description?: string;
    stories: Story[];
    story_count: number;
}

interface Connection {
    source: string;
    target: string;
    shared_story_count: number;
    shared_story_ids: string[];
}

interface MapData {
    locations: Location[];
    connections: Connection[];
    total_locations: number;
    total_connections: number;
}

// Component to fit map bounds to markers
function FitBounds({ locations }: { locations: Location[] }) {
    const map = useMap();

    useEffect(() => {
        if (locations.length > 0) {
            const bounds = locations.map(loc => [loc.latitude, loc.longitude] as [number, number]);
            map.fitBounds(bounds, { padding: [50, 50] });
        }
    }, [locations, map]);

    return null;
}

const MapPage: React.FC = () => {
    const [data, setData] = useState<MapData>({ locations: [], connections: [], total_locations: 0, total_connections: 0 });
    const [loading, setLoading] = useState(false); // Don't block map rendering
    const [selectedLocation, setSelectedLocation] = useState<string | null>(null);
    const navigate = useNavigate();

    useEffect(() => {
        // Start loading data in background
        setLoading(true);

        const fetchData = async () => {
            try {
                const response = await fetch('/api/map/locations?limit=100'); // Fetch more stories to get more locations
                const result = await response.json();

                if (result.status === 'success') {
                    setData(result);
                    console.log(`MapPage: Loaded ${result.total_locations} locations and ${result.total_connections} connections`);
                }
            } catch (error) {
                console.error('Error fetching map data:', error);
            } finally {
                setLoading(false);
            }
        };

        fetchData();
    }, []);

    // Create a lookup map for locations
    const locationMap = new Map<string, Location>();
    data.locations.forEach(loc => locationMap.set(loc.id, loc));

    // Get color based on story count
    const getLocationColor = (storyCount: number) => {
        if (storyCount >= 5) return '#ef4444'; // Red for hot spots
        if (storyCount >= 3) return '#f59e0b'; // Orange
        if (storyCount >= 2) return '#eab308'; // Yellow
        return '#3b82f6'; // Blue for single story
    };

    // Get marker size based on story count
    const getMarkerSize = (storyCount: number) => {
        return Math.min(8 + storyCount * 3, 25); // Size between 8 and 25
    };

    // Memoize location icons to avoid recreating on every render
    const locationIcons = useMemo(() => {
        const icons = new Map<string, L.DivIcon>();

        data.locations.forEach(location => {
            const color = getLocationColor(location.story_count);
            const size = getMarkerSize(location.story_count) * 2; // Double for better visibility

            if (location.thumbnail) {
                // Icon with thumbnail image
                icons.set(location.id, L.divIcon({
                    className: 'custom-location-marker',
                    html: `
                        <div style="
                            width: ${size}px;
                            height: ${size}px;
                            border-radius: 50%;
                            border: 3px solid ${color};
                            overflow: hidden;
                            background: white;
                            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
                        ">
                            <img
                                src="${location.thumbnail}"
                                style="
                                    width: 100%;
                                    height: 100%;
                                    object-fit: cover;
                                "
                                onerror="this.style.display='none'"
                            />
                        </div>
                    `,
                    iconSize: [size, size],
                    iconAnchor: [size / 2, size / 2],
                }));
            } else {
                // Fallback colored circle
                icons.set(location.id, L.divIcon({
                    className: 'custom-location-marker',
                    html: `
                        <div style="
                            width: ${size}px;
                            height: ${size}px;
                            border-radius: 50%;
                            background: ${color};
                            border: 3px solid white;
                            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
                        "></div>
                    `,
                    iconSize: [size, size],
                    iconAnchor: [size / 2, size / 2],
                }));
            }
        });

        return icons;
    }, [data.locations]);

    return (
        <div className="h-screen flex flex-col bg-gray-900 text-white">
            <div className="p-4 border-b border-gray-800 flex justify-between items-center z-10 bg-gray-900">
                <h1 className="text-xl font-bold">News Map: Global Story Locations</h1>
                <div className="text-sm text-gray-400 flex items-center gap-4">
                    {data.total_locations > 0 && (
                        <span>
                            {data.total_locations} locations • {data.total_connections} connections
                        </span>
                    )}
                    <div className="flex items-center gap-2">
                        <div className="flex items-center gap-1">
                            <div className="w-3 h-3 rounded-full bg-blue-500"></div>
                            <span className="text-xs">1-2 stories</span>
                        </div>
                        <div className="flex items-center gap-1">
                            <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
                            <span className="text-xs">3-4 stories</span>
                        </div>
                        <div className="flex items-center gap-1">
                            <div className="w-3 h-3 rounded-full bg-red-500"></div>
                            <span className="text-xs">5+ stories</span>
                        </div>
                    </div>
                </div>
            </div>

            <div className="flex-1 relative overflow-hidden">
                {/* Show loading indicator as overlay, not blocking */}
                {loading && (
                    <div className="absolute top-4 left-1/2 transform -translate-x-1/2 z-[1000] bg-gray-800 text-white px-4 py-2 rounded-lg shadow-lg flex items-center gap-2">
                        <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                        <span className="text-sm">Loading locations...</span>
                    </div>
                )}

                {/* Always show map, just without markers if no data yet */}
                <MapContainer
                        center={[20, 0]}
                        zoom={2}
                        className="h-full w-full"
                        style={{ background: '#111827' }}
                    >
                        <TileLayer
                            url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
                            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
                        />

                        {data.locations.length > 0 && <FitBounds locations={data.locations} />}

                        {/* Draw connections between locations */}
                        {data.locations.length > 0 && data.connections.map((conn, idx) => {
                            const source = locationMap.get(conn.source);
                            const target = locationMap.get(conn.target);

                            if (!source || !target) return null;

                            return (
                                <Polyline
                                    key={idx}
                                    positions={[
                                        [source.latitude, source.longitude],
                                        [target.latitude, target.longitude]
                                    ]}
                                    pathOptions={{
                                        color: '#a78bfa',
                                        weight: Math.min(conn.shared_story_count, 4),
                                        opacity: 0.3 + (conn.shared_story_count * 0.1)
                                    }}
                                />
                            );
                        })}

                        {/* Draw location markers */}
                        {data.locations.length > 0 && data.locations.map((location) => {
                            const icon = locationIcons.get(location.id);
                            if (!icon) return null;

                            return (
                                <Marker
                                    key={location.id}
                                    position={[location.latitude, location.longitude]}
                                    icon={icon}
                                    eventHandlers={{
                                        mouseover: () => setSelectedLocation(location.id),
                                        mouseout: () => setSelectedLocation(null)
                                    }}
                                >
                                <Popup>
                                    <div className="text-gray-900 max-w-xs">
                                        <h3 className="font-bold text-lg mb-1">{location.name}</h3>
                                        {location.description && (
                                            <p className="text-sm text-gray-600 mb-2">{location.description}</p>
                                        )}
                                        <div className="text-sm font-semibold mb-2">
                                            {location.story_count} {location.story_count === 1 ? 'story' : 'stories'}
                                        </div>
                                        <div className="space-y-1 max-h-48 overflow-y-auto">
                                            {location.stories.map((story) => (
                                                <div
                                                    key={story.id}
                                                    className="p-2 bg-gray-100 rounded hover:bg-gray-200 cursor-pointer transition-colors"
                                                    onClick={() => navigate(`/story/${story.id}`)}
                                                >
                                                    <div className="text-sm font-medium line-clamp-2">
                                                        {story.title}
                                                    </div>
                                                    {story.coherence !== undefined && (
                                                        <div className="text-xs text-gray-500 mt-1">
                                                            Coherence: {(story.coherence * 100).toFixed(0)}%
                                                        </div>
                                                    )}
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                </Popup>
                            </Marker>
                        );
                        })}
                    </MapContainer>
            </div>
        </div>
    );
};

export default MapPage;
