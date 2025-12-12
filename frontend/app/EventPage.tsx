import React, { useState, useEffect, useRef } from 'react';
import { useParams } from 'react-router-dom';
import TimelineView from './components/event/TimelineView';
import TimelineCard from './components/event/TimelineCard';
import TopologyView from './components/event/TopologyView';
import EpicenterMapCard from './components/event/EpicenterMapCard';
import EventSidebar from './components/event/EventSidebar';
import EventNarrativeContent from './components/event/EventNarrativeContent';
import DebateThread from './components/event/DebateThread';

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
    page_id?: string;
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

interface Thought {
    id: string;
    type: string;
    content: string;
    temperature?: number;
    coherence?: number;
    created_at?: string;
}

interface PageThumbnail {
    page_id: string;
    thumbnail_url: string;
    title?: string;
}

interface EventData {
    event: Event;
    entities: Entity[];
    claims: Claim[];
    children: any[];
    parent: any | null;
    thought?: Thought | null;
    page_thumbnails?: PageThumbnail[];
}

type TabType = 'narrative' | 'debate' | 'topology';

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

    const renderDebate = () => {
        if (!eventData) return null;
        return (
            <div className="flex gap-6">
                {/* Main debate area */}
                <div className="flex-1">
                    <DebateThread eventId={event.id} />
                </div>

                {/* Contributors sidebar in debate view */}
                <div className="w-64 flex-shrink-0">
                    <div className="bg-white rounded-lg border border-slate-200 shadow-sm overflow-hidden">
                        <div className="px-4 py-3 bg-gradient-to-r from-indigo-50 to-purple-50 border-b border-indigo-100">
                            <h3 className="font-semibold text-slate-800">Top Contributors</h3>
                        </div>
                        <div className="p-4">
                            <div className="space-y-3">
                                {[
                                    { name: 'Alice Chen', credits: 45, role: 'Fact Checker' },
                                    { name: 'Bob Smith', credits: 32, role: 'Source Analyst' },
                                    { name: 'Carol Wu', credits: 28, role: 'Editor' },
                                ].map((contributor, idx) => (
                                    <div key={idx} className="flex items-center gap-3">
                                        <div className="w-10 h-10 rounded-full bg-gradient-to-br from-indigo-400 to-purple-400 flex items-center justify-center text-white font-semibold">
                                            {contributor.name.charAt(0)}
                                        </div>
                                        <div className="flex-1">
                                            <div className="font-medium text-slate-800">{contributor.name}</div>
                                            <div className="text-xs text-slate-500">{contributor.role}</div>
                                        </div>
                                        <div className="text-right">
                                            <div className="text-sm font-bold text-amber-600">{contributor.credits}</div>
                                            <div className="text-xs text-slate-400">earned</div>
                                        </div>
                                    </div>
                                ))}
                            </div>

                            <div className="mt-4 pt-4 border-t border-slate-100">
                                <div className="text-xs text-slate-500 text-center">
                                    Contribute quality insights to earn credits
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        );
    };

    const renderTopology = () => {
        if (!eventData) return null;
        return <TopologyView eventId={event.id} eventName={event.canonical_name} />;
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

    const { event, entities, claims, thought, page_thumbnails } = eventData;

    // Get first available thumbnail for header background
    const headerThumbnail = page_thumbnails?.find(t => t.thumbnail_url)?.thumbnail_url;

    // Format date range with ~ for missing start/end
    const formatEventDate = () => {
        const formatDate = (dateStr: string) => {
            const date = new Date(dateStr);
            return date.toLocaleDateString('en-US', {
                month: 'short',
                day: 'numeric',
                year: 'numeric'
            });
        };

        if (event.event_start && event.event_end) {
            return `${formatDate(event.event_start)} ~ ${formatDate(event.event_end)}`;
        }
        if (event.event_start) {
            return `${formatDate(event.event_start)} ~`;
        }
        if (event.event_end) {
            return `~ ${formatDate(event.event_end)}`;
        }
        return null;
    };


    return (
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
            {/* Event Header - Redesigned */}
            <div className="relative rounded-xl shadow-sm border border-slate-200 mb-6 overflow-hidden">
                {/* Background thumbnail watermark - fades from bottom-right to top-left */}
                {headerThumbnail && (
                    <div
                        className="absolute inset-0"
                        style={{
                            backgroundImage: `url(${headerThumbnail})`,
                            backgroundSize: 'cover',
                            backgroundPosition: 'center',
                            filter: 'grayscale(50%)',
                            maskImage: 'linear-gradient(to top left, rgba(0,0,0,0.2) 0%, rgba(0,0,0,0) 60%)',
                            WebkitMaskImage: 'linear-gradient(to top left, rgba(0,0,0,0.2) 0%, rgba(0,0,0,0) 60%)',
                        }}
                    />
                )}

                <div className="relative p-6">
                    {/* Top row: Type badge and date */}
                    <div className="flex items-center justify-between mb-3">
                        <span className="px-3 py-1 rounded-full bg-slate-100 text-slate-600 text-xs font-medium uppercase tracking-wide">
                            {event.event_type}
                        </span>
                        {formatEventDate() && (
                            <span className="text-sm text-slate-500 flex items-center gap-1.5">
                                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
                                </svg>
                                {formatEventDate()}
                            </span>
                        )}
                    </div>

                    {/* Title */}
                    <h1 className="text-3xl font-bold text-slate-900 mb-4 leading-tight">
                        {event.canonical_name}
                    </h1>

                    {/* Thought - the "voice" of the event organism */}
                    {thought && (
                        <div className="flex items-start gap-3 bg-white/40 rounded-lg p-3">
                            <span className="text-lg flex-shrink-0">
                                {thought.type === 'question' ? '🤔' :
                                 thought.type === 'anomaly' ? '⚠️' :
                                 thought.type === 'contradiction' ? '⚡' :
                                 thought.type === 'progress' ? '📈' :
                                 thought.type === 'emergence' ? '🌱' : '💭'}
                            </span>
                            <p className="text-slate-700 text-sm leading-relaxed flex-1 italic">
                                "{thought.content}"
                            </p>
                        </div>
                    )}
                </div>
            </div>

            {/* Main content area with sidebar */}
            <div className="flex gap-6">
                {/* Main content */}
                <div className="flex-1 min-w-0">
                    {/* Tabs */}
                    <div className="bg-white rounded-xl shadow-sm border border-slate-200">
                        <div className="border-b border-slate-200">
                            <div className="flex">
                                {[
                                    { id: 'narrative', label: 'Narrative', icon: '📰', suffix: null },
                                    { id: 'debate', label: 'Debate', icon: '💬', suffix: null },
                                    { id: 'topology', label: 'Topology', icon: '🔮', suffix: `φ ${Math.round(event.coherence * 100)}%` }
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
                                        {tab.suffix && (
                                            <span className={`ml-2 px-2 py-0.5 rounded-full text-xs font-semibold ${
                                                activeTab === tab.id
                                                    ? 'bg-indigo-100 text-indigo-700'
                                                    : 'bg-slate-100 text-slate-600'
                                            }`}>
                                                {tab.suffix}
                                            </span>
                                        )}
                                    </button>
                                ))}
                            </div>
                        </div>

                        {/* Content */}
                        <div className="p-6">
                            {activeTab === 'narrative' && (
                                <div className="relative">
                                    {/* Floating cards - Map and Timeline stacked */}
                                    <div className="float-right ml-6 mb-4 space-y-4 z-10">
                                        <EpicenterMapCard
                                            entities={entities}
                                            eventName={event.canonical_name}
                                        />
                                        <TimelineCard
                                            claims={claims}
                                            eventSlug={eventSlug || ''}
                                        />
                                    </div>

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

                                    {/* Narrative content - full width, no max constraint */}
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

                                    {/* Clear float */}
                                    <div className="clear-both" />
                                </div>
                            )}

                            {activeTab === 'debate' && renderDebate()}

                            {activeTab === 'topology' && (
                                <div>
                                    {renderTopology()}
                                </div>
                            )}
                        </div>
                    </div>
                </div>

                {/* Right Sidebar - only show on narrative tab */}
                {activeTab === 'narrative' && (
                    <EventSidebar
                        eventId={event.id}
                        eventSlug={eventSlug || ''}
                        currentFund={273003}
                        distributedPercent={34}
                        contributorCount={127}
                        claimCount={claims.length}
                        sourceCount={entities.filter(e => e.entity_type === 'ORG').length || 3}
                        onNavigateToDebate={() => setActiveTab('debate')}
                    />
                )}
            </div>
        </div>
    );
};

export default EventPage;
