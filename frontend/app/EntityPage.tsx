import React, { useState, useEffect } from 'react';
import { useParams, Link, useNavigate } from 'react-router-dom';

interface Claim {
    id: string;
    text: string;
    event_time?: string;
    confidence?: number;
    page_id?: string;
    event_id?: string;
}

interface RelatedEvent {
    id: string;
    slug: string;
    canonical_name: string;
    event_type: string;
    event_start?: string;
    event_end?: string;
    coherence?: number;
    claims: Claim[];
}

interface EntityData {
    entity: {
        id: string;
        canonical_name: string;
        entity_type: string;
        aliases?: string[];
        mention_count?: number;
        profile_summary?: string;
        wikidata_qid?: string;
        wikidata_label?: string;
        wikidata_description?: string;
        image_url?: string;
        status?: string;
        confidence?: number;
        metadata?: Record<string, any>;
    };
    claims: Claim[];
    claims_count: number;
    related_events?: RelatedEvent[];
}

const EntityPage: React.FC = () => {
    const { entityId } = useParams<{ entityId: string }>();
    const navigate = useNavigate();
    const [data, setData] = useState<EntityData | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [expandedEvents, setExpandedEvents] = useState<Set<string>>(new Set());
    const [entityVote, setEntityVote] = useState<'up' | 'down' | null>(null);
    const [entityScore, setEntityScore] = useState(0);

    useEffect(() => {
        loadEntity();
    }, [entityId]);

    const loadEntity = async () => {
        try {
            setLoading(true);
            setError(null);
            const response = await fetch(`/api/entity/${entityId}`);
            if (!response.ok) {
                throw new Error('Entity not found');
            }
            const result = await response.json();
            setData(result);
        } catch (err: any) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    const getEntityTypeColor = (type: string) => {
        switch (type) {
            case 'PERSON': return 'bg-purple-100 text-purple-800';
            case 'ORGANIZATION': return 'bg-orange-100 text-orange-800';
            case 'GPE':
            case 'LOC':
            case 'LOCATION': return 'bg-green-100 text-green-800';
            default: return 'bg-gray-100 text-gray-800';
        }
    };

    const getEntityTypeLabel = (type: string) => {
        switch (type) {
            case 'PERSON': return 'Person';
            case 'ORGANIZATION': return 'Organization';
            case 'GPE': return 'Location';
            case 'LOC': return 'Location';
            case 'LOCATION': return 'Location';
            default: return type;
        }
    };

    if (loading) {
        return (
            <div className="min-h-screen bg-gray-50 flex items-center justify-center">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
            </div>
        );
    }

    if (error || !data) {
        return (
            <div className="min-h-screen bg-gray-50 flex items-center justify-center">
                <div className="text-center">
                    <h1 className="text-2xl font-bold text-gray-800 mb-2">Entity Not Found</h1>
                    <p className="text-gray-600 mb-4">{error || 'The requested entity could not be found.'}</p>
                    <Link to="/" className="text-blue-600 hover:underline">Back to Home</Link>
                </div>
            </div>
        );
    }

    const { entity, claims, related_events } = data;

    // Group claims by event if related_events not provided
    const getEventsWithClaims = (): RelatedEvent[] => {
        if (related_events && related_events.length > 0) {
            return related_events;
        }
        // Fallback: group claims by event_id if available
        const eventMap = new Map<string, Claim[]>();
        claims.forEach(claim => {
            if (claim.event_id) {
                if (!eventMap.has(claim.event_id)) {
                    eventMap.set(claim.event_id, []);
                }
                eventMap.get(claim.event_id)!.push(claim);
            }
        });

        // If no claims have event_id, return empty (we'll show claims directly)
        if (eventMap.size === 0) {
            return [];
        }

        return Array.from(eventMap.entries()).map(([eventId, eventClaims]) => ({
            id: eventId,
            slug: eventId,
            canonical_name: `Event ${eventId.replace('ev_', '').slice(0, 8)}...`,
            event_type: 'event',
            claims: eventClaims,
        }));
    };

    const eventsWithClaims = getEventsWithClaims();

    const toggleEventExpanded = (eventId: string) => {
        setExpandedEvents(prev => {
            const newSet = new Set(prev);
            if (newSet.has(eventId)) {
                newSet.delete(eventId);
            } else {
                newSet.add(eventId);
            }
            return newSet;
        });
    };

    const formatEventDate = (event: RelatedEvent) => {
        if (!event.event_start && !event.event_end) return null;
        const formatDate = (dateStr: string) => {
            const date = new Date(dateStr);
            return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
        };
        if (event.event_start && event.event_end) {
            return `${formatDate(event.event_start)} ~ ${formatDate(event.event_end)}`;
        }
        if (event.event_start) return `${formatDate(event.event_start)} ~`;
        if (event.event_end) return `~ ${formatDate(event.event_end)}`;
        return null;
    };

    const handleUpvote = () => {
        if (entityVote === 'up') {
            // Remove upvote
            setEntityVote(null);
            setEntityScore(prev => prev - 1);
        } else {
            // Add upvote (remove downvote if exists)
            const delta = entityVote === 'down' ? 2 : 1;
            setEntityVote('up');
            setEntityScore(prev => prev + delta);
        }
        // TODO: API call to record vote
    };

    const handleDownvote = () => {
        if (entityVote === 'down') {
            // Remove downvote
            setEntityVote(null);
            setEntityScore(prev => prev + 1);
        } else {
            // Add downvote (remove upvote if exists)
            const delta = entityVote === 'up' ? 2 : 1;
            setEntityVote('down');
            setEntityScore(prev => prev - delta);
        }
        // TODO: API call to record vote
    };

    return (
        <div className="min-h-screen bg-gray-50">
            {/* Header */}
            <div className="bg-white border-b">
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
                    <div className="flex items-start gap-6">
                        {/* Image + Wiki Links */}
                        <div className="flex-shrink-0 flex flex-col items-center gap-2">
                            {entity.image_url ? (
                                <img
                                    src={entity.image_url}
                                    alt={entity.canonical_name}
                                    className="w-32 h-32 rounded-lg object-cover shadow-md"
                                    onError={(e) => {
                                        (e.target as HTMLImageElement).style.display = 'none';
                                    }}
                                />
                            ) : (
                                <div className="w-32 h-32 rounded-lg bg-gray-200 flex items-center justify-center">
                                    <span className="text-4xl text-gray-400">
                                        {entity.entity_type === 'PERSON' ? '👤' :
                                         entity.entity_type === 'ORGANIZATION' ? '🏢' : '📍'}
                                    </span>
                                </div>
                            )}
                            {/* Wiki links under image */}
                            {entity.wikidata_qid && (
                                <div className="flex gap-2">
                                    <a
                                        href={`https://en.wikipedia.org/wiki/${encodeURIComponent(entity.canonical_name.replace(/ /g, '_'))}`}
                                        target="_blank"
                                        rel="noopener noreferrer"
                                        className="px-2 py-1 rounded text-xs font-medium bg-slate-100 text-slate-600 hover:bg-slate-200 transition-colors flex items-center gap-1"
                                        title="View on Wikipedia"
                                    >
                                        <span>📖</span> Wiki
                                    </a>
                                    <a
                                        href={`https://www.wikidata.org/wiki/${entity.wikidata_qid}`}
                                        target="_blank"
                                        rel="noopener noreferrer"
                                        className="px-2 py-1 rounded text-xs font-medium bg-blue-50 text-blue-600 hover:bg-blue-100 transition-colors"
                                        title={`Wikidata: ${entity.wikidata_qid}`}
                                    >
                                        {entity.wikidata_qid}
                                    </a>
                                </div>
                            )}
                        </div>

                        {/* Info */}
                        <div className="flex-1">
                            <div className="flex items-center gap-3 mb-2">
                                <span className={`px-2 py-1 rounded text-sm font-medium ${getEntityTypeColor(entity.entity_type)}`}>
                                    {getEntityTypeLabel(entity.entity_type)}
                                </span>
                            </div>

                            <h1 className="text-3xl font-bold text-gray-900 mb-2">
                                {entity.canonical_name}
                            </h1>

                            {entity.wikidata_description && (
                                <p className="text-lg text-gray-600 mb-3">
                                    {entity.wikidata_description}
                                </p>
                            )}

                            {entity.aliases && entity.aliases.length > 0 && (
                                <p className="text-sm text-gray-500">
                                    Also known as: {entity.aliases.join(', ')}
                                </p>
                            )}

                            {entity.profile_summary && (
                                <p className="text-gray-700 mt-3">
                                    {entity.profile_summary}
                                </p>
                            )}
                        </div>

                        {/* Voting Section */}
                        <div className="flex-shrink-0 flex flex-col items-center gap-1 bg-slate-50 rounded-lg px-3 py-4 border border-slate-200">
                            <button
                                onClick={handleUpvote}
                                className={`w-10 h-10 flex items-center justify-center rounded-full transition-all hover:scale-110 active:scale-95 ${
                                    entityVote === 'up'
                                        ? 'bg-green-100 text-green-600'
                                        : 'bg-white text-slate-400 hover:bg-slate-100 hover:text-slate-600'
                                }`}
                                title="Upvote entity (costs 1 credit)"
                            >
                                <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 20 20">
                                    <path fillRule="evenodd" d="M3.293 9.707a1 1 0 010-1.414l6-6a1 1 0 011.414 0l6 6a1 1 0 01-1.414 1.414L11 5.414V17a1 1 0 11-2 0V5.414L4.707 9.707a1 1 0 01-1.414 0z" clipRule="evenodd" />
                                </svg>
                            </button>
                            <div className={`text-xl font-bold ${
                                entityScore > 0 ? 'text-green-600' :
                                entityScore < 0 ? 'text-red-600' :
                                'text-slate-600'
                            }`}>
                                {entityScore > 0 ? '+' : ''}{entityScore}
                            </div>
                            <button
                                onClick={handleDownvote}
                                className={`w-10 h-10 flex items-center justify-center rounded-full transition-all hover:scale-110 active:scale-95 ${
                                    entityVote === 'down'
                                        ? 'bg-red-100 text-red-600'
                                        : 'bg-white text-slate-400 hover:bg-slate-100 hover:text-slate-600'
                                }`}
                                title="Downvote entity (costs 1 credit)"
                            >
                                <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 20 20">
                                    <path fillRule="evenodd" d="M16.707 10.293a1 1 0 010 1.414l-6 6a1 1 0 01-1.414 0l-6-6a1 1 0 111.414-1.414L9 14.586V3a1 1 0 012 0v11.586l4.293-4.293a1 1 0 011.414 0z" clipRule="evenodd" />
                                </svg>
                            </button>
                            <div className="text-xs text-slate-500 mt-1">1⭐ each</div>
                        </div>
                    </div>
                </div>
            </div>

            {/* Stats */}
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
                <div className="flex gap-4">
                    <div className="bg-white rounded-lg px-4 py-3 shadow-sm">
                        <div className="text-2xl font-bold text-indigo-600">{eventsWithClaims.length}</div>
                        <div className="text-sm text-gray-500">Related Events</div>
                    </div>
                    <div className="bg-white rounded-lg px-4 py-3 shadow-sm">
                        <div className="text-2xl font-bold text-gray-900">{claims.length}</div>
                        <div className="text-sm text-gray-500">Claims</div>
                    </div>
                    <div className="bg-white rounded-lg px-4 py-3 shadow-sm">
                        <div className="text-2xl font-bold text-gray-900">{entity.mention_count || 0}</div>
                        <div className="text-sm text-gray-500">Mentions</div>
                    </div>
                    {entity.confidence && (
                        <div className="bg-white rounded-lg px-4 py-3 shadow-sm">
                            <div className="text-2xl font-bold text-gray-900">
                                {(entity.confidence * 100).toFixed(0)}%
                            </div>
                            <div className="text-sm text-gray-500">Confidence</div>
                        </div>
                    )}
                </div>
            </div>

            {/* Related Events Section */}
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
                <h2 className="text-xl font-bold text-gray-900 mb-4">
                    Related Events
                </h2>

                {eventsWithClaims.length === 0 ? (
                    <div className="bg-white rounded-lg p-6 text-center text-gray-500">
                        No events found for this entity.
                    </div>
                ) : (
                    <div className="space-y-4">
                        {eventsWithClaims.map((event) => {
                            const isExpanded = expandedEvents.has(event.id);
                            const dateStr = formatEventDate(event);

                            return (
                                <div
                                    key={event.id}
                                    className="bg-white rounded-lg shadow-sm border border-gray-100 overflow-hidden"
                                >
                                    {/* Event Header */}
                                    <div
                                        className="p-4 cursor-pointer hover:bg-gray-50 transition-colors"
                                        onClick={() => toggleEventExpanded(event.id)}
                                    >
                                        <div className="flex items-start justify-between">
                                            <div className="flex-1">
                                                <div className="flex items-center gap-2 mb-1">
                                                    <span className="px-2 py-0.5 rounded text-xs font-medium bg-slate-100 text-slate-600 uppercase">
                                                        {event.event_type}
                                                    </span>
                                                    {event.coherence && (
                                                        <span className="text-xs text-slate-500">
                                                            φ {Math.round(event.coherence * 100)}%
                                                        </span>
                                                    )}
                                                </div>
                                                <Link
                                                    to={`/event/${event.slug}`}
                                                    className="text-lg font-semibold text-indigo-600 hover:text-indigo-800 hover:underline"
                                                    onClick={(e) => e.stopPropagation()}
                                                >
                                                    {event.canonical_name}
                                                </Link>
                                                {dateStr && (
                                                    <div className="text-sm text-gray-500 mt-1">
                                                        {dateStr}
                                                    </div>
                                                )}
                                            </div>
                                            <div className="flex items-center gap-3">
                                                <span className="text-sm text-gray-500">
                                                    {event.claims.length} claim{event.claims.length !== 1 ? 's' : ''}
                                                </span>
                                                <svg
                                                    className={`w-5 h-5 text-gray-400 transition-transform ${isExpanded ? 'rotate-180' : ''}`}
                                                    fill="none"
                                                    stroke="currentColor"
                                                    viewBox="0 0 24 24"
                                                >
                                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                                                </svg>
                                            </div>
                                        </div>
                                    </div>

                                    {/* Claims under this event */}
                                    {isExpanded && event.claims.length > 0 && (
                                        <div className="border-t border-gray-100 bg-gray-50 p-4">
                                            <div className="space-y-2">
                                                {event.claims.map((claim) => (
                                                    <div
                                                        key={claim.id}
                                                        className="bg-white rounded-lg p-3 border border-gray-200 hover:border-indigo-300 hover:shadow-sm transition-all cursor-pointer group"
                                                        onClick={() => {
                                                            if (claim.page_id) {
                                                                navigate(`/page/${claim.page_id}#${claim.id}`);
                                                            }
                                                        }}
                                                    >
                                                        <p className="text-gray-800 text-sm group-hover:text-gray-900">
                                                            {claim.text}
                                                        </p>
                                                        <div className="flex items-center gap-3 mt-2 text-xs text-gray-500">
                                                            {claim.event_time && (
                                                                <span>
                                                                    {new Date(claim.event_time).toLocaleDateString()}
                                                                </span>
                                                            )}
                                                            {claim.confidence && (
                                                                <span className="text-emerald-600">
                                                                    {(claim.confidence * 100).toFixed(0)}% confidence
                                                                </span>
                                                            )}
                                                            {claim.page_id && (
                                                                <span className="ml-auto text-indigo-500 group-hover:text-indigo-700">
                                                                    View source →
                                                                </span>
                                                            )}
                                                        </div>
                                                    </div>
                                                ))}
                                            </div>
                                        </div>
                                    )}
                                </div>
                            );
                        })}
                    </div>
                )}
            </div>

            {/* Direct Claims Section - shown when no events but claims exist */}
            {eventsWithClaims.length === 0 && claims.length > 0 && (
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
                    <h2 className="text-xl font-bold text-gray-900 mb-4">
                        Related Claims
                    </h2>
                    <div className="space-y-2">
                        {claims.map((claim) => (
                            <div
                                key={claim.id}
                                className="bg-white rounded-lg p-4 border border-gray-200 hover:border-indigo-300 hover:shadow-sm transition-all cursor-pointer group"
                                onClick={() => {
                                    if (claim.page_id) {
                                        navigate(`/page/${claim.page_id}#${claim.id}`);
                                    }
                                }}
                            >
                                <p className="text-gray-800 group-hover:text-gray-900">
                                    {claim.text}
                                </p>
                                <div className="flex items-center gap-3 mt-2 text-xs text-gray-500">
                                    {claim.event_time && (
                                        <span>
                                            {new Date(claim.event_time).toLocaleDateString()}
                                        </span>
                                    )}
                                    {claim.confidence && (
                                        <span className="text-emerald-600">
                                            {(claim.confidence * 100).toFixed(0)}% confidence
                                        </span>
                                    )}
                                    {claim.page_id && (
                                        <span className="ml-auto text-indigo-500 group-hover:text-indigo-700">
                                            View source →
                                        </span>
                                    )}
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            )}

        </div>
    );
};

export default EntityPage;
