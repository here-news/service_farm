import React, { useState, useEffect, useRef } from 'react';
import { useParams } from 'react-router-dom';
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

interface NarrativeSection {
    topic: string;
    title: string;
    content: string;
    claim_ids: string[];
}

interface KeyFigure {
    label: string;
    value: string;
    claim_id: string;
    supersedes?: string;
}

interface StructuredNarrative {
    sections: NarrativeSection[];
    key_figures: KeyFigure[];
    pattern: string;
    consensus_date?: string;
    generated_at?: string;
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
    narrative?: StructuredNarrative;
}

interface EventData {
    event: Event;
    entities: Entity[];
    claims: Claim[];
    children: any[];
    parent: any | null;
}

type TabType = 'narrative' | 'timeline' | 'graph' | 'map';

const EventPage: React.FC = () => {
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
            const response = await fetch(`/api/event/${eventSlug}`);
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
            <div className="flex items-center justify-center py-20">
                <div className="text-center">
                    <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600 mx-auto mb-4"></div>
                    <p className="text-slate-500">Loading event...</p>
                </div>
            </div>
        );
    }

    if (error || !eventData) {
        return (
            <div className="flex items-center justify-center py-20">
                <div className="text-center">
                    <p className="text-red-500 text-xl mb-4">Error: {error || 'Event not found'}</p>
                </div>
            </div>
        );
    }

    const { event, entities, claims } = eventData;

    return (
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
            {/* Event Header */}
            <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6 mb-6">
                <h1 className="text-3xl font-bold text-slate-900 mb-4">{event.canonical_name}</h1>

                <div className="flex flex-wrap gap-4 text-sm">
                    {event.event_start && (
                        <div className="flex items-center gap-2 text-slate-600">
                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
                            </svg>
                            <span>
                                {new Date(event.event_start).toLocaleDateString('en-US', {
                                    year: 'numeric',
                                    month: 'long',
                                    day: 'numeric'
                                })}
                            </span>
                        </div>
                    )}
                    <div className="flex items-center gap-2 text-slate-600">
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 7h.01M7 3h5c.512 0 1.024.195 1.414.586l7 7a2 2 0 010 2.828l-7 7a2 2 0 01-2.828 0l-7-7A1.994 1.994 0 013 12V7a4 4 0 014-4z" />
                        </svg>
                        <span>{event.event_type}</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <span className="px-3 py-1 rounded-full bg-gradient-to-r from-indigo-500 to-purple-600 text-white text-sm font-semibold">
                            φ {Math.round(event.coherence * 100)}%
                        </span>
                    </div>
                    <div className="flex items-center gap-2 text-slate-600">
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                        </svg>
                        <span>{claims.length} claims</span>
                    </div>
                    <div className="flex items-center gap-2 text-slate-600">
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
                        </svg>
                        <span>{entities.length} entities</span>
                    </div>
                </div>
            </div>

            {/* Tabs */}
            <div className="bg-white rounded-xl shadow-sm border border-slate-200 mb-6">
                <div className="border-b border-slate-200">
                    <div className="flex">
                        {[
                            { id: 'narrative', label: 'Narrative', icon: '📰' },
                            { id: 'timeline', label: 'Timeline', icon: '⏱️' },
                            { id: 'graph', label: 'Graph', icon: '🕸️' },
                            { id: 'map', label: 'Map', icon: '🗺️' }
                        ].map(tab => (
                            <button
                                key={tab.id}
                                onClick={() => setActiveTab(tab.id as TabType)}
                                className={`px-6 py-4 font-medium transition-all border-b-2 ${
                                    activeTab === tab.id
                                        ? 'text-indigo-600 border-indigo-600'
                                        : 'text-slate-500 border-transparent hover:text-slate-700 hover:border-slate-300'
                                }`}
                            >
                                <span className="mr-2">{tab.icon}</span>
                                {tab.label}
                            </button>
                        ))}
                    </div>
                </div>

                {/* Content */}
                <div className="p-6">
                    {activeTab === 'narrative' && (
                        <div className="max-w-4xl mx-auto">
                            {/* Key Figures bar */}
                            {event.narrative?.key_figures && event.narrative.key_figures.length > 0 && (
                                <div className="bg-slate-50 border border-slate-200 rounded-lg p-4 mb-6 flex flex-wrap gap-6">
                                    {event.narrative.key_figures.map((fig, idx) => (
                                        <div key={idx} className="flex items-center gap-2">
                                            <span className="text-slate-500 text-sm capitalize">
                                                {fig.label.replace(/_/g, ' ')}:
                                            </span>
                                            <span className="text-xl font-bold text-slate-900">{fig.value}</span>
                                        </div>
                                    ))}
                                </div>
                            )}

                            {/* Narrative content */}
                            <div className="prose prose-slate max-w-none">
                                {event.narrative?.sections ? (
                                    event.narrative.sections.map((section, idx) => (
                                        <div key={idx} className={idx > 0 ? 'mt-8' : ''}>
                                            {section.title && (
                                                <h2 className="text-xl font-bold text-slate-800 mb-4">
                                                    {section.title}
                                                </h2>
                                            )}
                                            <EventNarrativeContent
                                                content={section.content}
                                                entities={entities}
                                                claims={claims}
                                            />
                                        </div>
                                    ))
                                ) : (
                                    <EventNarrativeContent
                                        content={event.summary}
                                        entities={entities}
                                        claims={claims}
                                    />
                                )}
                            </div>
                        </div>
                    )}

                    {activeTab === 'timeline' && (
                        <div className="max-w-4xl mx-auto">
                            {renderTimeline()}
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
                </div>
            </div>
        </div>
    );
};

export default EventPage;
