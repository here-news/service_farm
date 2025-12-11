import React, { useState, useEffect, useRef } from 'react';
import { useParams, Link } from 'react-router-dom';
import TimelineView from './components/event/TimelineView';
import GraphView from './components/event/GraphView';
import MapView from './components/event/MapView';
import EventNarrativeContent from './components/event/EventNarrativeContent';

interface Entity {
    id: string;
    canonical_name: string;
    entity_type: string;
    mention_count: number;
    wikidata_qid?: string;
    wikidata_label?: string;
    wikidata_description?: string;
    image_url?: string;
    confidence?: number;
}

interface Claim {
    id: string;
    text: string;
    event_time?: string;
    confidence?: number;
}

interface Event {
    id: string;
    canonical_name: string;
    event_type: string;
    event_scale: string;
    coherence: number;
    event_start?: string | null;
    event_end?: string | null;
    summary: string;
}

interface EventData {
    event: Event;
    entities: Entity[];
    claims: Claim[];
    children: any[];
    parent: any | null;
}

type TabType = 'narrative' | 'timeline' | 'graph' | 'map';

const EventPageNew: React.FC = () => {
    const { eventSlug } = useParams<{ eventSlug: string }>();
    const [activeTab, setActiveTab] = useState<TabType>('narrative');
    const [eventData, setEventData] = useState<EventData | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const graphRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        loadEvent();
    }, [eventSlug]);

    const loadEvent = async () => {
        try {
            setLoading(true);
            const response = await fetch(`/api/events/${eventSlug}`);
            if (!response.ok) throw new Error('Event not found');
            const data = await response.json();
            setEventData(data);
        } catch (err: any) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    const renderTimeline = () => {
        if (!eventData) return null;
        return <TimelineView claims={eventData.claims} />;
    };

    const renderGraph = () => {
        if (!eventData) return null;
        return <GraphView entities={eventData.entities} claims={eventData.claims} eventName={event.canonical_name} />;
    };

    const renderMap = () => {
        if (!eventData) return null;
        return <MapView entities={eventData.entities} claims={eventData.claims} eventName={event.canonical_name} />;
    };

    if (loading) {
        return (
            <div className="min-h-screen bg-black flex items-center justify-center">
                <div className="text-center">
                    <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
                    <p className="text-gray-400">Loading event...</p>
                </div>
            </div>
        );
    }

    if (error || !eventData) {
        return (
            <div className="min-h-screen bg-black flex items-center justify-center">
                <div className="text-center">
                    <p className="text-red-500 text-xl mb-4">Error: {error || 'Event not found'}</p>
                    <Link to="/app" className="text-blue-500 hover:text-blue-400">← Back to Home</Link>
                </div>
            </div>
        );
    }

    const { event, entities, claims } = eventData;

    return (
        <div className="min-h-screen bg-black text-white">
            {/* Header */}
            <header className="sticky top-0 z-50 bg-black/95 backdrop-blur-sm border-b border-gray-800">
                <div className="max-w-7xl mx-auto px-4 py-4">
                    <div className="flex items-center justify-between mb-3">
                        <Link to="/app" className="text-blue-500 hover:text-blue-400 text-sm">← Back</Link>
                        <div className="text-xs text-gray-500">Event ID: {event.id}</div>
                    </div>

                    <h1 className="text-3xl font-bold mb-3">{event.canonical_name}</h1>

                    <div className="flex flex-wrap gap-4 text-sm">
                        {event.event_start && (
                            <div className="flex items-center gap-2">
                                <span>📅</span>
                                <span className="text-gray-400">
                                    {new Date(event.event_start).toLocaleDateString('en-US', {
                                        year: 'numeric',
                                        month: 'long',
                                        day: 'numeric'
                                    })}
                                </span>
                            </div>
                        )}
                        <div className="flex items-center gap-2">
                            <span>📍</span>
                            <span className="text-gray-400">{event.event_type}</span>
                        </div>
                        <div className="flex items-center gap-2">
                            <span className="px-3 py-1 rounded-full bg-gradient-to-r from-blue-500 to-purple-600 text-white font-semibold">
                                φ {Math.round(event.coherence * 100)}%
                            </span>
                        </div>
                        <div className="flex items-center gap-2">
                            <span>🔗</span>
                            <span className="text-gray-400">{claims.length} claims</span>
                        </div>
                        <div className="flex items-center gap-2">
                            <span>👤</span>
                            <span className="text-gray-400">{entities.length} entities</span>
                        </div>
                    </div>
                </div>
            </header>

            {/* Tabs */}
            <div className="sticky top-[140px] z-40 bg-gray-900/95 backdrop-blur-sm border-b border-gray-800">
                <div className="max-w-7xl mx-auto px-4">
                    <div className="flex gap-1">
                        {[
                            { id: 'narrative', label: '📰 Narrative' },
                            { id: 'timeline', label: '⏱️ Timeline' },
                            { id: 'graph', label: '🕸️ Graph' },
                            { id: 'map', label: '🗺️ Map' }
                        ].map(tab => (
                            <button
                                key={tab.id}
                                onClick={() => setActiveTab(tab.id as TabType)}
                                className={`px-6 py-3 font-medium transition-all ${
                                    activeTab === tab.id
                                        ? 'text-white border-b-2 border-blue-500'
                                        : 'text-gray-400 hover:text-gray-200'
                                }`}
                            >
                                {tab.label}
                            </button>
                        ))}
                    </div>
                </div>
            </div>

            {/* Content */}
            <main className="max-w-7xl mx-auto px-4 py-8">
                {activeTab === 'narrative' && (
                    <div className="max-w-4xl mx-auto">
                        <div className="bg-gray-900/50 rounded-lg p-8">
                            <EventNarrativeContent
                                content={event.summary}
                                entities={entities}
                                claims={claims}
                            />
                        </div>
                    </div>
                )}

                {activeTab === 'timeline' && (
                    <div className="max-w-4xl mx-auto">
                        <div className="bg-gray-900/50 rounded-lg p-8">
                            {renderTimeline()}
                        </div>
                    </div>
                )}

                {activeTab === 'graph' && (
                    <div>
                        {renderGraph()}
                    </div>
                )}

                {activeTab === 'map' && (
                    <div>
                        {renderMap()}
                    </div>
                )}
            </main>
        </div>
    );
};

export default EventPageNew;
