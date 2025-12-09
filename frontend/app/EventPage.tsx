import React, { useState, useEffect } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import {
    MapPin, Clock, Users, Building2, FileText, ExternalLink,
    Calendar, TrendingUp, Award, Eye, CheckCircle2, AlertTriangle,
    HelpCircle, DollarSign, MessageSquare, Image, ChevronDown,
    ChevronUp, Coins, Target, UserCircle, Shield, ThumbsUp, ThumbsDown
} from 'lucide-react';

interface Entity {
    id: string;
    name: string;
    type: string;
    thumbnail?: string;
    description?: string;
    role?: string;
    coordinates?: string;
}

interface Claim {
    id: string;
    text: string;
    stance?: string;
    date?: string;
    confidence?: number;
}

interface Artifact {
    id: string;
    url: string;
    title?: string;
    published_date?: string;
    domain?: string;
    snippet?: string;
}

interface RelatedStory {
    id: string;
    title: string;
    description?: string;
    cover_image?: string;
    created_at?: string;
    match_score: number;
    entities: Array<{name: string; type: string}>;
}

interface EventData {
    id: string;
    slug: string;
    title: string;
    description?: string;
    content?: string;
    cover_image?: string;
    created_at?: string;
    last_updated?: string;
    coherence: number;
    timely: number;
    people: Entity[];
    organizations: Entity[];
    locations: Entity[];
    claims: Claim[];
    artifacts: Artifact[];
    related_stories: RelatedStory[];
    entity_count: number;
    source_count: number;
    claim_count: number;
}

// Helper: Parse overview text with inline entity markers
const parseOverviewWithEntities = (text: string) => {
    // Parse {{Entity Name|type|confidence}} markers
    const parts = [];
    let lastIndex = 0;
    const regex = /\{\{([^|]+)\|([^|]+)\|([^}]+)\}\}/g;
    let match;

    while ((match = regex.exec(text)) !== null) {
        // Add text before match
        if (match.index > lastIndex) {
            parts.push({ type: 'text', content: text.substring(lastIndex, match.index) });
        }

        // Add entity chip
        parts.push({
            type: 'entity',
            name: match[1],
            entityType: match[2],
            confidence: parseFloat(match[3])
        });

        lastIndex = match.index + match[0].length;
    }

    // Add remaining text
    if (lastIndex < text.length) {
        parts.push({ type: 'text', content: text.substring(lastIndex) });
    }

    return parts;
};

// Entity chip component with inline headshot
const EntityChip: React.FC<{ name: string; type: string; confidence: number; event?: any }> = ({ name, type, confidence, event }) => {
    const [showTooltip, setShowTooltip] = useState(false);

    // Find entity in event data to get thumbnail
    const findEntity = () => {
        if (!event) return null;
        const allEntities = [
            ...(event.people || []).map((p: any) => ({ ...p, entityType: 'person' })),
            ...(event.organizations || []).map((o: any) => ({ ...o, entityType: 'org' })),
            ...(event.locations || []).map((l: any) => ({ ...l, entityType: 'loc' }))
        ];
        return allEntities.find((e: any) => e.name === name || e.canonical_name === name);
    };

    const entity = findEntity();
    const thumbnail = entity?.thumbnail || entity?.wikidata_thumbnail;
    const description = entity?.description;

    const getIconAndColor = () => {
        switch (type) {
            case 'person':
                return { icon: <UserCircle size={12} />, bg: 'bg-green-50', text: 'text-green-800', border: 'border-green-200', hoverBg: 'hover:bg-green-100' };
            case 'org':
                return { icon: <Building2 size={12} />, bg: 'bg-blue-50', text: 'text-blue-800', border: 'border-blue-200', hoverBg: 'hover:bg-blue-100' };
            case 'loc':
                return { icon: <MapPin size={12} />, bg: 'bg-red-50', text: 'text-red-800', border: 'border-red-200', hoverBg: 'hover:bg-red-100' };
            default:
                return { icon: null, bg: 'bg-gray-50', text: 'text-gray-800', border: 'border-gray-200', hoverBg: 'hover:bg-gray-100' };
        }
    };

    const { icon, bg, text, border, hoverBg } = getIconAndColor();
    const isHighConfidence = confidence >= 0.85;

    return (
        <span className="relative inline-block">
            <span
                className={`inline-flex items-center gap-1.5 px-2 py-1 rounded-full ${bg} ${text} ${border} ${hoverBg} border text-sm font-medium cursor-pointer transition-all ${!isHighConfidence ? 'border-dashed' : ''}`}
                onMouseEnter={() => setShowTooltip(true)}
                onMouseLeave={() => setShowTooltip(false)}
            >
                {thumbnail ? (
                    <img
                        src={thumbnail}
                        alt={name}
                        className="w-5 h-5 rounded-full object-cover border border-white shadow-sm"
                    />
                ) : (
                    icon
                )}
                <span className="font-semibold">{name}</span>
                {!isHighConfidence && <HelpCircle size={10} className="opacity-50" />}
            </span>
            {showTooltip && (
                <div className="absolute z-50 bottom-full left-0 mb-2 bg-white border border-gray-300 rounded-lg shadow-xl overflow-hidden" style={{ minWidth: '200px', maxWidth: '280px' }}>
                    {thumbnail && (
                        <div className="h-24 bg-gradient-to-br from-gray-100 to-gray-200 flex items-center justify-center">
                            <img
                                src={thumbnail}
                                alt={name}
                                className="h-20 w-20 rounded-full object-cover border-2 border-white shadow-lg"
                            />
                        </div>
                    )}
                    <div className="p-3">
                        <div className="font-bold text-gray-900 mb-1">{name}</div>
                        {description && (
                            <div className="text-xs text-gray-600 mb-2 line-clamp-2">{description}</div>
                        )}
                        <div className="flex items-center justify-between text-xs">
                            <span className="text-gray-500 capitalize">{type}</span>
                            <span className={`font-medium ${confidence >= 0.85 ? 'text-green-600' : confidence >= 0.70 ? 'text-yellow-600' : 'text-orange-600'}`}>
                                {Math.round(confidence * 100)}%
                            </span>
                        </div>
                    </div>
                </div>
            )}
        </span>
    );
};

// Claim component with confidence visualization
const ClaimCard: React.FC<{ claim: any }> = ({ claim }) => {
    const getStatusStyle = () => {
        switch (claim.status) {
            case 'verified':
                return { bg: 'bg-green-50', border: 'border-green-300', icon: <CheckCircle2 size={14} className="text-green-600" />, label: 'Verified', textColor: 'text-green-700' };
            case 'disputed':
                return { bg: 'bg-red-50', border: 'border-red-300', icon: <AlertTriangle size={14} className="text-red-600" />, label: 'Disputed', textColor: 'text-red-700' };
            case 'unverified':
                return { bg: 'bg-yellow-50', border: 'border-yellow-300', icon: <HelpCircle size={14} className="text-yellow-600" />, label: 'Unverified', textColor: 'text-yellow-700' };
            default:
                return { bg: 'bg-gray-50', border: 'border-gray-300', icon: null, label: 'Unknown', textColor: 'text-gray-700' };
        }
    };

    const status = getStatusStyle();
    const confidencePercent = Math.round(claim.confidence * 100);

    return (
        <div className={`${status.bg} ${status.border} border rounded-lg p-3 mb-2`}>
            <div className="flex items-start justify-between gap-3 mb-2">
                <div className="flex-1">
                    <div className="text-gray-900 leading-relaxed text-sm">{claim.text}</div>
                </div>
                <div className="flex items-center gap-1.5">
                    {status.icon}
                    <span className={`text-xs font-medium ${status.textColor}`}>{status.label}</span>
                </div>
            </div>

            <div className="flex items-center gap-4 text-xs text-gray-600">
                <div className="flex items-center gap-1">
                    <FileText size={11} />
                    {claim.sources} source{claim.sources !== 1 ? 's' : ''}
                </div>
                <div className={`flex items-center gap-1 ${confidencePercent >= 85 ? 'text-green-600' : confidencePercent >= 70 ? 'text-yellow-600' : 'text-orange-600'}`}>
                    <TrendingUp size={11} />
                    {confidencePercent}% confidence
                </div>
                {claim.modality && (
                    <span className="px-1.5 py-0.5 bg-gray-200 rounded text-gray-700 text-xs">
                        {claim.modality}
                    </span>
                )}
            </div>

            {claim.dispute_reason && (
                <div className="mt-3 pt-3 border-t border-red-200 text-sm text-red-800">
                    <div className="flex items-start gap-2">
                        <AlertTriangle size={13} className="mt-0.5 flex-shrink-0" />
                        <div>
                            <div className="font-medium">Dispute:</div>
                            <div className="text-red-700">{claim.dispute_reason}</div>
                            <div className="mt-1 flex items-center gap-3 text-xs">
                                <span className="flex items-center gap-1">
                                    <ThumbsUp size={9} /> {claim.votes_for || 0} support
                                </span>
                                <span className="flex items-center gap-1">
                                    <ThumbsDown size={9} /> {claim.votes_against || 0} oppose
                                </span>
                            </div>
                        </div>
                    </div>
                </div>
            )}

            {claim.note && (
                <div className="mt-2 text-xs text-gray-600 italic">
                    Note: {claim.note}
                </div>
            )}
        </div>
    );
};

const EventPage: React.FC = () => {
    const { eventSlug } = useParams<{ eventSlug: string }>();
    const navigate = useNavigate();
    const [event, setEvent] = useState<any | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [activeTab, setActiveTab] = useState<'overview' | 'timeline'>('overview');
    const [expandedSections, setExpandedSections] = useState<Set<string>>(new Set());

    useEffect(() => {
        const fetchEvent = async () => {
            try {
                const response = await fetch(`/api/event/${eventSlug}`);
                if (!response.ok) throw new Error('Event not found');

                const data = await response.json();
                setEvent(data.event);
            } catch (err) {
                setError(err instanceof Error ? err.message : 'Failed to load event');
            } finally {
                setLoading(false);
            }
        };

        fetchEvent();
    }, [eventSlug]);

    if (loading) {
        return (
            <div className="min-h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-gray-900 flex items-center justify-center">
                <div className="animate-spin rounded-full h-16 w-16 border-t-2 border-b-2 border-blue-400"></div>
            </div>
        );
    }

    if (error || !event) {
        return (
            <div className="min-h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-gray-900 flex items-center justify-center text-white">
                <div className="text-center">
                    <h1 className="text-4xl font-bold mb-4">Event Not Found</h1>
                    <p className="text-gray-400 mb-8">{error}</p>
                    <button
                        onClick={() => navigate('/feed')}
                        className="px-6 py-3 bg-blue-600 hover:bg-blue-700 rounded-lg transition"
                    >
                        Return to Feed
                    </button>
                </div>
            </div>
        );
    }

    const formatDate = (dateStr?: string) => {
        if (!dateStr) return 'Unknown date';
        try {
            return new Date(dateStr).toLocaleDateString('en-US', {
                year: 'numeric',
                month: 'long',
                day: 'numeric',
                hour: '2-digit',
                minute: '2-digit'
            });
        } catch {
            return dateStr;
        }
    };

    const allEntities = [
        ...(event.people || []).map((p: any) => ({ ...p, type: 'person' })),
        ...(event.organizations || []).map((o: any) => ({ ...o, type: 'organization' })),
        ...(event.locations || []).map((l: any) => ({ ...l, type: 'location' }))
    ];

    return (
        <div className="min-h-screen bg-gray-50">
            {/* Compact Header */}
            <div className="bg-white border-b border-gray-200">
                <div className="max-w-7xl mx-auto px-6 py-4">
                    {/* Breadcrumb */}
                    <div className="flex items-center gap-2 text-sm text-gray-600 mb-3">
                        <Link to="/feed" className="hover:text-blue-600">Feed</Link>
                        <span>/</span>
                        <span className="text-gray-900 font-medium">{event.slug.replace(/_/g, ' ')}</span>
                    </div>

                    <div className="flex items-start justify-between gap-6">
                        {/* Title & Description */}
                        <div className="flex-1">
                            <h1 className="text-3xl font-bold text-gray-900 mb-2 leading-tight">
                                {event.title}
                            </h1>
                            {event.description && (
                                <p className="text-base text-gray-600 leading-relaxed">
                                    {event.description}
                                </p>
                            )}
                        </div>

                        {/* Compact Metrics - Horizontal */}
                        <div className="flex items-center gap-6 text-sm">
                            <div className="text-center">
                                <div className="flex items-center gap-1 text-blue-600 mb-1">
                                    <TrendingUp size={14} />
                                    <span className="font-medium">Coherence</span>
                                </div>
                                <div className="text-2xl font-bold text-gray-900">{Math.round(event.coherence)}</div>
                            </div>
                            <div className="h-12 w-px bg-gray-200"></div>
                            <div className="text-center">
                                <div className="flex items-center gap-1 text-green-600 mb-1">
                                    <Users size={14} />
                                    <span className="font-medium">Entities</span>
                                </div>
                                <div className="text-2xl font-bold text-gray-900">{event.entity_count}</div>
                            </div>
                            <div className="h-12 w-px bg-gray-200"></div>
                            <div className="text-center">
                                <div className="flex items-center gap-1 text-purple-600 mb-1">
                                    <FileText size={14} />
                                    <span className="font-medium">Claims</span>
                                </div>
                                <div className="text-2xl font-bold text-gray-900">{event.claim_count}</div>
                            </div>
                            <div className="h-12 w-px bg-gray-200"></div>
                            <div className="text-center">
                                <div className="flex items-center gap-1 text-orange-600 mb-1">
                                    <Eye size={14} />
                                    <span className="font-medium">Sources</span>
                                </div>
                                <div className="text-2xl font-bold text-gray-900">{event.source_count}</div>
                            </div>
                        </div>
                    </div>

                    {/* Updated timestamp */}
                    <div className="flex items-center gap-2 text-gray-500 text-xs mt-3">
                        <Clock size={12} />
                        <span>Last updated: {formatDate(event.last_updated)}</span>
                    </div>
                </div>
            </div>

            {/* Navigation Tabs */}
            <div className="sticky top-0 z-10 bg-white border-b border-gray-200 shadow-sm">
                <div className="max-w-7xl mx-auto px-6">
                    <div className="flex gap-8">
                        {[
                            { id: 'overview', label: 'Overview' },
                            { id: 'timeline', label: 'Timeline' }
                        ].map(tab => (
                            <button
                                key={tab.id}
                                onClick={() => setActiveTab(tab.id as any)}
                                className={`py-3 px-1 border-b-2 transition text-sm font-medium ${
                                    activeTab === tab.id
                                        ? 'border-blue-600 text-blue-600'
                                        : 'border-transparent text-gray-600 hover:text-gray-900 hover:border-gray-300'
                                }`}
                            >
                                {tab.label}
                            </button>
                        ))}
                    </div>
                </div>
            </div>

            {/* Content Area */}
            <div className="max-w-7xl mx-auto px-6 py-12">
                {/* OVERVIEW TAB */}
                {activeTab === 'overview' && (
                    <div className="grid grid-cols-1 lg:grid-cols-4 gap-8 max-w-7xl mx-auto">
                        {/* Main Reading Area - 3 columns */}
                        <div className="lg:col-span-3 space-y-6">

                            {/* Overview with floating images */}
                            {event.overview && (
                                <div className="bg-white rounded-lg p-8 border border-gray-200">
                                    <h1 className="text-2xl font-bold text-gray-900 mb-6 leading-tight">Complete Event Overview</h1>

                                    <div className="prose prose-gray max-w-none text-gray-800 leading-relaxed">
                                        {/* Hero Image - Full Width */}
                                        {event.artifacts && event.artifacts.find((a: any) => a.id === 'img-1') && (() => {
                                            const heroImg = event.artifacts.find((a: any) => a.id === 'img-1');
                                            return (
                                                <div className="mb-6 rounded-lg overflow-hidden border border-gray-200">
                                                    <img src={heroImg.url} alt={heroImg.caption} className="w-full h-80 object-cover" />
                                                    <div className="bg-gray-50 px-4 py-2 text-sm text-gray-600 flex items-center justify-between">
                                                        <span>{heroImg.caption}</span>
                                                        <span className="text-xs text-green-600 flex items-center gap-1">
                                                            <CheckCircle2 size={12} />
                                                            Verified
                                                        </span>
                                                    </div>
                                                </div>
                                            );
                                        })()}

                                        {event.overview.split('\n').map((line: string, idx: number) => {
                                            // Inject floating images at strategic points
                                            const renderFloatingImage = (position: string) => {
                                                // Insert gallery image after first paragraph
                                                if (idx === 3 && position === 'after-intro') {
                                                    const galleryImg = event.artifacts?.find((a: any) => a.id === 'img-2');
                                                    if (galleryImg) {
                                                        return (
                                                            <div className="float-right ml-6 mb-4 w-80 rounded-lg overflow-hidden border border-gray-200 shadow-sm">
                                                                <img src={galleryImg.url} alt={galleryImg.caption} className="w-full h-48 object-cover" />
                                                                <div className="bg-gray-50 px-3 py-2 text-xs text-gray-600">
                                                                    <div className="font-medium mb-1">{galleryImg.caption}</div>
                                                                    <div className="flex items-center justify-between">
                                                                        <span className="text-gray-500">{galleryImg.source}</span>
                                                                        <span className="text-green-600 flex items-center gap-1">
                                                                            <CheckCircle2 size={10} />
                                                                            {Math.round(galleryImg.confidence * 100)}%
                                                                        </span>
                                                                    </div>
                                                                </div>
                                                            </div>
                                                        );
                                                    }
                                                }

                                                // Insert missing image placeholder after investigation section
                                                if (idx === 10 && position === 'missing-necklace') {
                                                    const bounty = event.artifacts?.find((a: any) => a.id === 'bounty-1');
                                                    if (bounty) {
                                                        return (
                                                            <div className="float-left mr-6 mb-4 w-72 rounded-lg overflow-hidden border-2 border-dashed border-purple-300 bg-purple-50 shadow-sm cursor-pointer hover:border-purple-400 transition group">
                                                                <div className="aspect-square bg-gradient-to-br from-purple-100 to-purple-50 flex flex-col items-center justify-center p-6">
                                                                    <Target size={48} className="text-purple-400 mb-3 group-hover:scale-110 transition" />
                                                                    <div className="text-center">
                                                                        <div className="text-xs text-purple-600 font-bold mb-1">MISSING IMAGE</div>
                                                                        <div className="text-sm font-semibold text-gray-900 mb-2">{bounty.target}</div>
                                                                        <div className="flex items-center justify-center gap-1 text-purple-700 font-bold text-sm">
                                                                            <Coins size={14} />
                                                                            {bounty.reward} gems
                                                                        </div>
                                                                    </div>
                                                                </div>
                                                                <div className="bg-purple-100 px-3 py-2 text-xs text-purple-800">
                                                                    <div className="flex items-center justify-between">
                                                                        <span>{bounty.contributors} contributors</span>
                                                                        <span className="font-medium">Click to contribute →</span>
                                                                    </div>
                                                                </div>
                                                            </div>
                                                        );
                                                    }
                                                }

                                                // Insert pending verification image
                                                if (idx === 16 && position === 'crown-image') {
                                                    const pendingImg = event.artifacts?.find((a: any) => a.id === 'img-3');
                                                    if (pendingImg) {
                                                        return (
                                                            <div className="float-right ml-6 mb-4 w-80 rounded-lg overflow-hidden border-2 border-yellow-300 bg-yellow-50 shadow-sm">
                                                                <img src={pendingImg.url} alt={pendingImg.caption} className="w-full h-48 object-cover" />
                                                                <div className="bg-yellow-50 px-3 py-2 text-xs">
                                                                    <div className="font-medium text-gray-900 mb-1">{pendingImg.caption}</div>
                                                                    <div className="flex items-center justify-between">
                                                                        <span className="text-gray-600">{pendingImg.source}</span>
                                                                        <span className="text-yellow-700 flex items-center gap-1 font-medium">
                                                                            <HelpCircle size={10} />
                                                                            Pending ({Math.round(pendingImg.confidence * 100)}%)
                                                                        </span>
                                                                    </div>
                                                                    <div className="mt-2 flex items-center gap-2 text-xs">
                                                                        <span className="flex items-center gap-1 text-green-600">
                                                                            <ThumbsUp size={10} /> {pendingImg.votes_for}
                                                                        </span>
                                                                        <span className="flex items-center gap-1 text-red-600">
                                                                            <ThumbsDown size={10} /> {pendingImg.votes_against}
                                                                        </span>
                                                                        <button className="ml-auto text-blue-600 font-medium hover:text-blue-700">
                                                                            Vote
                                                                        </button>
                                                                    </div>
                                                                </div>
                                                            </div>
                                                        );
                                                    }
                                                }

                                                return null;
                                            };

                                            if (line.startsWith('## ')) {
                                                return (
                                                    <div key={idx}>
                                                        <div className="clear-both"></div>
                                                        {(() => {
                                                            const headingText = line.replace('## ', '');
                                                            // Find matching sub-event
                                                            const matchingSubEvent = event.sub_events?.find((se: any) =>
                                                                se.title.toLowerCase().includes(headingText.toLowerCase()) ||
                                                                headingText.toLowerCase().includes(se.title.toLowerCase())
                                                            );

                                                            if (matchingSubEvent) {
                                                                const isExpanded = expandedSections.has(matchingSubEvent.id);
                                                                const unverifiedCount = matchingSubEvent.claim_count - matchingSubEvent.verified_claims;

                                                                return (
                                                                    <div className="mt-8 mb-4">
                                                                        <button
                                                                            onClick={() => {
                                                                                const newExpanded = new Set(expandedSections);
                                                                                if (isExpanded) {
                                                                                    newExpanded.delete(matchingSubEvent.id);
                                                                                } else {
                                                                                    newExpanded.add(matchingSubEvent.id);
                                                                                }
                                                                                setExpandedSections(newExpanded);
                                                                            }}
                                                                            className="w-full flex items-center justify-between border-b border-gray-200 pb-2 hover:border-blue-400 transition group text-left"
                                                                        >
                                                                            <h2 className="text-xl font-bold text-gray-900">{headingText}</h2>
                                                                            <div className="flex items-center gap-3 flex-shrink-0">
                                                                                <div className="flex items-center gap-2 text-xs">
                                                                                    <span className="text-gray-600">{matchingSubEvent.claim_count} claims</span>
                                                                                    {matchingSubEvent.disputed_claims > 0 && (
                                                                                        <span className="px-2 py-0.5 bg-red-50 text-red-700 rounded font-medium">
                                                                                            {matchingSubEvent.disputed_claims} disputed
                                                                                        </span>
                                                                                    )}
                                                                                    {unverifiedCount > matchingSubEvent.disputed_claims && (
                                                                                        <span className="px-2 py-0.5 bg-yellow-50 text-yellow-700 rounded font-medium">
                                                                                            {unverifiedCount - matchingSubEvent.disputed_claims} unverified
                                                                                        </span>
                                                                                    )}
                                                                                </div>
                                                                                {isExpanded ?
                                                                                    <ChevronUp size={18} className="text-gray-600 group-hover:text-blue-600" /> :
                                                                                    <ChevronDown size={18} className="text-gray-600 group-hover:text-blue-600" />
                                                                                }
                                                                            </div>
                                                                        </button>

                                                                        {isExpanded && (
                                                                            <div className="mt-4 space-y-2 pl-4 border-l-2 border-blue-200">
                                                                                {matchingSubEvent.claims.map((claim: any) => (
                                                                                    <ClaimCard key={claim.id} claim={claim} />
                                                                                ))}
                                                                            </div>
                                                                        )}
                                                                    </div>
                                                                );
                                                            } else {
                                                                return <h2 className="text-xl font-bold text-gray-900 mt-8 mb-4 border-b border-gray-200 pb-2">{headingText}</h2>;
                                                            }
                                                        })()}
                                                    </div>
                                                );
                                            } else if (line.startsWith('**')) {
                                                return <p key={idx} className="font-semibold text-blue-700 mb-2 mt-4">{line.replace(/\*\*/g, '')}</p>;
                                            } else if (line.startsWith('- ')) {
                                                return <li key={idx} className="ml-6 mb-2 text-gray-700 leading-relaxed">{line.replace('- ', '')}</li>;
                                            } else if (line.trim() === '') {
                                                return <div key={idx} className="h-2"></div>;
                                            } else {
                                                // Parse entities in line
                                                const parts = parseOverviewWithEntities(line);
                                                return (
                                                    <div key={idx}>
                                                        {renderFloatingImage('after-intro')}
                                                        {renderFloatingImage('missing-necklace')}
                                                        {renderFloatingImage('crown-image')}
                                                        <p className="mb-4 text-base leading-relaxed">
                                                            {parts.map((part: any, pIdx) =>
                                                                part.type === 'entity' ? (
                                                                    <EntityChip
                                                                        key={pIdx}
                                                                        name={part.name || ''}
                                                                        type={part.entityType || 'unknown'}
                                                                        confidence={part.confidence || 0}
                                                                        event={event}
                                                                    />
                                                                ) : (
                                                                    <span key={pIdx}>{part.content}</span>
                                                                )
                                                            )}
                                                        </p>
                                                    </div>
                                                );
                                            }
                                        })}

                                        <div className="clear-both"></div>
                                    </div>
                                </div>
                            )}

                            {/* Artifacts & Bounties */}
                            {event.artifacts && event.artifacts.length > 0 && (
                                <div className="bg-white rounded-lg p-6 border border-gray-200">
                                    <h2 className="text-xl font-bold text-gray-900 mb-4">Visual Evidence</h2>
                                    <div className="grid grid-cols-1 gap-4">
                                        {event.artifacts.map((artifact: any) => {
                                            if (artifact.type === 'image') {
                                                return (
                                                    <div key={artifact.id} className={`rounded-lg overflow-hidden border ${artifact.status === 'verified' ? 'border-green-300' : 'border-yellow-300'}`}>
                                                        <img src={artifact.url} alt={artifact.caption} className="w-full aspect-video object-cover" />
                                                        <div className={`p-3 ${artifact.status === 'verified' ? 'bg-green-50' : 'bg-yellow-50'}`}>
                                                            <div className="flex items-start justify-between gap-3 mb-2">
                                                                <p className="text-gray-900 font-medium flex-1 text-sm">{artifact.caption}</p>
                                                                <div className="flex items-center gap-1">
                                                                    {artifact.status === 'verified' ? (
                                                                        <>
                                                                            <CheckCircle2 size={14} className="text-green-600" />
                                                                            <span className="text-xs text-green-700 font-medium">Verified</span>
                                                                        </>
                                                                    ) : (
                                                                        <>
                                                                            <HelpCircle size={14} className="text-yellow-600" />
                                                                            <span className="text-xs text-yellow-700 font-medium">Pending</span>
                                                                        </>
                                                                    )}
                                                                </div>
                                                            </div>
                                                            <div className="flex items-center gap-4 text-xs text-gray-600">
                                                                <span>Source: {artifact.source}</span>
                                                                <span>{Math.round(artifact.confidence * 100)}% confidence</span>
                                                                {artifact.votes_for && (
                                                                    <span className="flex items-center gap-1">
                                                                        <ThumbsUp size={10} /> {artifact.votes_for} / <ThumbsDown size={10} /> {artifact.votes_against}
                                                                    </span>
                                                                )}
                                                            </div>
                                                        </div>
                                                    </div>
                                                );
                                            } else if (artifact.type === 'image_bounty') {
                                                return (
                                                    <div key={artifact.id} className="border-2 border-dashed border-purple-300 bg-purple-50 rounded-lg p-4">
                                                        <div className="flex items-start gap-4">
                                                            <div className="p-2.5 bg-purple-100 rounded-lg">
                                                                <Target size={28} className="text-purple-600" />
                                                            </div>
                                                            <div className="flex-1">
                                                                <div className="flex items-center gap-2 mb-2">
                                                                    <h4 className="text-base font-semibold text-gray-900">{artifact.target}</h4>
                                                                    <span className="px-2 py-0.5 bg-purple-200 text-purple-700 text-xs font-bold rounded">BOUNTY</span>
                                                                </div>
                                                                <p className="text-gray-700 text-sm mb-3">{artifact.description}</p>
                                                                <div className="flex items-center gap-4 text-sm">
                                                                    <div className="flex items-center gap-1 text-purple-700 font-semibold">
                                                                        <Coins size={14} />
                                                                        {artifact.reward} {artifact.currency}
                                                                    </div>
                                                                    <span className="text-gray-600 text-xs">{artifact.contributors} contributors</span>
                                                                </div>
                                                                <button className="mt-3 px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition text-sm font-medium">
                                                                    Submit Evidence
                                                                </button>
                                                            </div>
                                                        </div>
                                                    </div>
                                                );
                                            }
                                        })}
                                    </div>
                                </div>
                            )}

                            {/* Narratives Section - Community Perspectives */}
                            {event.comments && event.comments.length > 0 && (
                                <div className="bg-white rounded-lg p-6 border border-gray-200">
                                    <div className="mb-6">
                                        <h2 className="text-xl font-bold text-gray-900 mb-2">Narratives & Community Knowledge</h2>
                                        <p className="text-sm text-gray-600">Perspectives, analysis, and contributions from the community</p>
                                    </div>

                                    <div className="space-y-4">
                                        {event.comments.map((comment: any) => {
                                            const typeStyles = {
                                                knowledge_contribution: {
                                                    bg: 'bg-green-50',
                                                    border: 'border-green-200',
                                                    icon: <CheckCircle2 size={14} className="text-green-600" />,
                                                    label: 'Knowledge Contribution',
                                                    color: 'text-green-700'
                                                },
                                                potential_clue: {
                                                    bg: 'bg-blue-50',
                                                    border: 'border-blue-200',
                                                    icon: <AlertTriangle size={14} className="text-blue-600" />,
                                                    label: 'Potential Clue',
                                                    color: 'text-blue-700'
                                                },
                                                dispute_resolution: {
                                                    bg: 'bg-orange-50',
                                                    border: 'border-orange-200',
                                                    icon: <Shield size={14} className="text-orange-600" />,
                                                    label: 'Dispute Resolution',
                                                    color: 'text-orange-700'
                                                },
                                                question: {
                                                    bg: 'bg-purple-50',
                                                    border: 'border-purple-200',
                                                    icon: <HelpCircle size={14} className="text-purple-600" />,
                                                    label: 'Question',
                                                    color: 'text-purple-700'
                                                }
                                            };

                                            const style = typeStyles[comment.type as keyof typeof typeStyles] || typeStyles.knowledge_contribution;

                                            return (
                                                <div key={comment.id} className={`${style.bg} ${style.border} border rounded-lg p-4`}>
                                                    {/* Author Header */}
                                                    <div className="flex items-start justify-between mb-3">
                                                        <div className="flex items-center gap-3">
                                                            <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-500 rounded-full flex items-center justify-center text-white font-bold text-sm">
                                                                {comment.author.name.charAt(0)}
                                                            </div>
                                                            <div>
                                                                <div className="flex items-center gap-2">
                                                                    <span className="text-gray-900 font-semibold">{comment.author.name}</span>
                                                                    {comment.author.badges && comment.author.badges.map((badge: string) => (
                                                                        <span key={badge} className="px-2 py-0.5 bg-blue-100 text-blue-700 text-xs rounded flex items-center gap-1">
                                                                            <Shield size={10} />
                                                                            {badge}
                                                                        </span>
                                                                    ))}
                                                                </div>
                                                                <div className="flex items-center gap-3 text-xs text-gray-600">
                                                                    <span>Reputation: {comment.author.reputation}</span>
                                                                    <span>{new Date(comment.created_at).toLocaleDateString()}</span>
                                                                </div>
                                                            </div>
                                                        </div>
                                                        <div className="flex items-center gap-2">
                                                            {style.icon}
                                                            <span className={`text-xs font-medium ${style.color}`}>{style.label}</span>
                                                        </div>
                                                    </div>

                                                    {/* Comment Text */}
                                                    <p className="text-gray-700 leading-relaxed mb-3">{comment.text}</p>

                                                    {/* Extracted Claims */}
                                                    {comment.extracted_claims && comment.extracted_claims.length > 0 && (
                                                        <div className="mt-3 pt-3 border-t border-green-300">
                                                            <div className="text-sm font-medium text-green-700 mb-2 flex items-center gap-1">
                                                                <CheckCircle2 size={14} />
                                                                Claim Extracted:
                                                            </div>
                                                            {comment.extracted_claims.map((claim: any, idx: number) => (
                                                                <div key={idx} className="bg-green-100 border border-green-300 rounded px-3 py-2 text-sm text-green-900">
                                                                    "{claim.text}"
                                                                    <span className="text-xs text-green-700 ml-2">
                                                                        ({Math.round(claim.confidence * 100)}% confidence • {claim.status.replace(/_/g, ' ')})
                                                                    </span>
                                                                </div>
                                                            ))}
                                                        </div>
                                                    )}

                                                    {/* Moderator Note */}
                                                    {comment.moderator_note && (
                                                        <div className="mt-3 pt-3 border-t border-gray-300">
                                                            <div className="text-xs text-gray-600 italic">
                                                                <span className="font-medium">Moderator:</span> {comment.moderator_note}
                                                            </div>
                                                        </div>
                                                    )}

                                                    {/* Voting & Actions */}
                                                    <div className="flex items-center justify-between mt-4 pt-3 border-t border-gray-200">
                                                        <div className="flex items-center gap-3">
                                                            <button className="flex items-center gap-1 text-gray-600 hover:text-green-600 transition">
                                                                <ThumbsUp size={16} />
                                                                <span className="text-sm font-medium">{comment.votes.for}</span>
                                                            </button>
                                                            <button className="flex items-center gap-1 text-gray-600 hover:text-red-600 transition">
                                                                <ThumbsDown size={16} />
                                                                <span className="text-sm font-medium">{comment.votes.against}</span>
                                                            </button>
                                                        </div>
                                                        <div className="flex items-center gap-2">
                                                            <button className="text-sm text-gray-600 hover:text-gray-900 transition font-medium">Reply</button>
                                                            {comment.type === 'knowledge_contribution' && !comment.extracted_claims?.length && (
                                                                <button className="text-sm text-blue-600 hover:text-blue-700 transition font-medium">Extract Claim</button>
                                                            )}
                                                        </div>
                                                    </div>
                                                </div>
                                            );
                                        })}
                                    </div>

                                    {/* Add Comment */}
                                    <div className="mt-6 bg-gray-50 border border-gray-200 rounded-lg p-4">
                                        <h4 className="text-base font-bold text-gray-900 mb-3">Contribute to this Event</h4>
                                        <textarea
                                            className="w-full bg-white border border-gray-300 rounded-lg p-3 text-gray-900 placeholder-gray-500 focus:outline-none focus:border-blue-500 transition text-sm"
                                            rows={4}
                                            placeholder="Share knowledge, raise questions, or submit clues..."
                                        ></textarea>
                                        <div className="flex items-center justify-between mt-3">
                                            <div className="flex items-center gap-2 text-xs text-gray-600">
                                                <span>💡 Submit a clue</span>
                                                <span>•</span>
                                                <span>❓ Ask a question</span>
                                                <span>•</span>
                                                <span>🚩 Report error</span>
                                            </div>
                                            <button className="px-5 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition font-medium text-sm">
                                                Post Comment
                                            </button>
                                        </div>
                                    </div>
                                </div>
                            )}
                        </div>

                        {/* Assistive Sidebar - 1 column */}
                        <div className="lg:col-span-1 space-y-4">
                            {/* Event Quality & Funding */}
                            <div className="bg-gradient-to-br from-blue-50 to-purple-50 rounded-lg p-4 border border-blue-200 sticky top-24">
                                {/* Coherence Score */}
                                <div className="mb-4">
                                    <div className="flex items-center justify-between mb-2">
                                        <span className="text-xs text-gray-600 font-medium">EVENT COHERENCE</span>
                                        <span className={`text-xs font-bold ${event.coherence >= 80 ? 'text-green-600' : event.coherence >= 60 ? 'text-blue-600' : 'text-yellow-600'}`}>
                                            {event.coherence >= 80 ? 'Excellent' : event.coherence >= 60 ? 'Good' : 'Developing'}
                                        </span>
                                    </div>
                                    <div className="flex items-center gap-3">
                                        <div className="flex-1">
                                            <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                                                <div
                                                    className={`h-full rounded-full transition-all ${event.coherence >= 80 ? 'bg-green-500' : event.coherence >= 60 ? 'bg-blue-500' : 'bg-yellow-500'}`}
                                                    style={{ width: `${event.coherence}%` }}
                                                ></div>
                                            </div>
                                        </div>
                                        <div className="text-2xl font-bold text-gray-900">{Math.round(event.coherence)}</div>
                                    </div>
                                    <div className="text-xs text-gray-500 mt-1">Based on source agreement & verification</div>
                                </div>

                                <div className="border-t border-gray-200 pt-4 mb-4">
                                    <div className="flex items-center gap-2 mb-3">
                                        <Users size={18} className="text-blue-600" />
                                        <h3 className="font-bold text-gray-900 text-sm">Fund Event Contributors</h3>
                                    </div>

                                    <div className="bg-white rounded-lg p-3 border border-gray-200 mb-3">
                                        <div className="flex items-center justify-between mb-2">
                                            <span className="text-xs text-gray-600">Total Funded</span>
                                            <span className="text-lg font-bold text-blue-600">
                                                {event.economics.total_tips} <span className="text-xs text-gray-600">credits</span>
                                            </span>
                                        </div>
                                        <div className="flex items-center justify-between text-xs">
                                            <span className="text-gray-600">Contributors</span>
                                            <span className="font-semibold text-gray-900">{event.economics.contributor_count} people</span>
                                        </div>
                                    </div>

                                    {/* Distribution Breakdown */}
                                    <div className="bg-gray-50 rounded p-2 mb-3">
                                        <div className="text-xs text-gray-500 mb-2 font-medium">Distribution</div>
                                        <div className="space-y-1 text-xs">
                                            {Object.entries(event.economics.funding_breakdown).map(([key, value]: [string, any]) => (
                                                <div key={key} className="flex items-center justify-between">
                                                    <span className="text-gray-600 capitalize">• {key.replace(/_/g, ' ')}</span>
                                                    <span className="font-semibold text-gray-900">{value}¢</span>
                                                </div>
                                            ))}
                                        </div>
                                        <button className="text-xs text-blue-600 hover:text-blue-700 mt-2 font-medium">
                                            View full breakdown →
                                        </button>
                                    </div>

                                    {/* Fund Action */}
                                    <div className="bg-white rounded-lg p-3 border-2 border-blue-400 mb-2">
                                        <div className="text-xs font-medium text-gray-900 mb-2">Support contributors who:</div>
                                        <ul className="text-xs text-gray-600 space-y-1 mb-3">
                                            <li>• Verified {event.claim_count} claims</li>
                                            <li>• Submitted {event.artifacts?.filter((a: any) => a.type === 'image').length} verified images</li>
                                            <li>• Contributed sources & analysis</li>
                                        </ul>
                                        <div className="flex gap-2">
                                            <button className="flex-1 py-2 bg-blue-600 hover:bg-blue-700 text-white font-semibold rounded transition text-xs">
                                                Fund 100¢
                                            </button>
                                            <button className="px-3 py-2 bg-blue-100 hover:bg-blue-200 text-blue-700 font-medium rounded transition text-xs">
                                                Custom
                                            </button>
                                        </div>
                                    </div>

                                    <div className="text-xs text-gray-500 text-center">
                                        1 credit = 1¢ • Distributed to all contributors
                                    </div>
                                </div>
                            </div>

                            {/* Active Bounties */}
                            {event.bounties && event.bounties.length > 0 && (
                                <div className="bg-white rounded-lg p-4 border border-gray-200">
                                    <div className="flex items-center gap-2 mb-3">
                                        <Target size={18} className="text-purple-600" />
                                        <h3 className="font-bold text-gray-900 text-sm">Active Bounties</h3>
                                    </div>
                                    <div className="space-y-2">
                                        {event.bounties.slice(0, 3).map((bounty: any) => (
                                            <div key={bounty.id} className="p-2 bg-purple-50 rounded border border-purple-200">
                                                <div className="text-xs font-medium text-gray-900 mb-1">{bounty.target}</div>
                                                <div className="flex items-center justify-between text-xs">
                                                    <span className="text-purple-700 font-bold">{bounty.reward} gems</span>
                                                    <span className="text-gray-600">{bounty.submissions} submissions</span>
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                    <button className="w-full mt-2 py-1.5 text-xs text-purple-600 hover:text-purple-700 font-medium">
                                        View All Bounties →
                                    </button>
                                </div>
                            )}

                            {/* Quick Stats */}
                            <div className="bg-gray-50 rounded-lg p-4 border border-gray-200">
                                <h3 className="font-bold text-gray-900 text-sm mb-3">Event Stats</h3>
                                <div className="space-y-2 text-xs">
                                    <div className="flex items-center justify-between">
                                        <span className="text-gray-600">Verified Claims</span>
                                        <span className="font-semibold text-green-600">
                                            {event.sub_events?.reduce((acc: number, se: any) => acc + se.verified_claims, 0) || 0}
                                        </span>
                                    </div>
                                    <div className="flex items-center justify-between">
                                        <span className="text-gray-600">Disputed Claims</span>
                                        <span className="font-semibold text-red-600">
                                            {event.sub_events?.reduce((acc: number, se: any) => acc + se.disputed_claims, 0) || 0}
                                        </span>
                                    </div>
                                    <div className="flex items-center justify-between">
                                        <span className="text-gray-600">Images</span>
                                        <span className="font-semibold text-gray-900">
                                            {event.artifacts?.filter((a: any) => a.type === 'image').length || 0} verified
                                        </span>
                                    </div>
                                    <div className="flex items-center justify-between">
                                        <span className="text-gray-600">Missing</span>
                                        <span className="font-semibold text-purple-600">
                                            {event.artifacts?.filter((a: any) => a.type === 'image_bounty').length || 0} bounties
                                        </span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                )}

                {/* TIMELINE TAB */}
                {activeTab === 'timeline' && (
                    <div className="max-w-4xl">
                        <h2 className="text-3xl font-bold text-gray-900 mb-4">Complete Event Timeline</h2>
                        <p className="text-gray-600 mb-8">Chronological sequence of all verified developments in the Louvre Heist 2025</p>

                        {event.timeline && event.timeline.length > 0 ? (
                            <div className="relative">
                                {/* Timeline line */}
                                <div className="absolute left-8 top-0 bottom-0 w-0.5 bg-blue-200"></div>

                                <div className="space-y-6">
                                    {event.timeline.map((item: any) => (
                                        <div key={item.id} className="relative flex gap-6">
                                            {/* Timeline dot - color by severity */}
                                            <div className={`relative z-10 flex items-center justify-center w-16 h-16 rounded-full border-4 border-white shadow-md ${
                                                item.severity === 'critical' ? 'bg-red-500' :
                                                item.severity === 'high' ? 'bg-orange-500' :
                                                item.severity === 'medium' ? 'bg-blue-500' :
                                                'bg-green-500'
                                            }`}>
                                                <CheckCircle2 size={24} className="text-white" />
                                            </div>

                                            {/* Timeline card */}
                                            <div className="flex-1 bg-white rounded-lg p-6 border border-gray-200 shadow-sm">
                                                <div className="flex items-start justify-between mb-3">
                                                    <div className="flex-1">
                                                        <div className="text-sm text-blue-600 mb-2 flex items-center gap-2">
                                                            <Calendar size={14} />
                                                            {formatDate(item.date)}
                                                        </div>
                                                        <h3 className="text-xl font-bold text-gray-900 mb-2">{item.title}</h3>
                                                    </div>
                                                    {item.verified && (
                                                        <div className="flex items-center gap-1 text-green-600 text-sm">
                                                            <CheckCircle2 size={16} />
                                                            <span>Verified</span>
                                                        </div>
                                                    )}
                                                </div>
                                                <p className="text-gray-700 leading-relaxed mb-3">{item.description}</p>
                                                <div className="flex items-center gap-4 text-sm text-gray-600">
                                                    <span className={`px-2 py-1 rounded ${
                                                        item.type === 'event' ? 'bg-red-50 text-red-700' :
                                                        item.type === 'development' ? 'bg-blue-50 text-blue-700' :
                                                        'bg-purple-50 text-purple-700'
                                                    }`}>
                                                        {item.type}
                                                    </span>
                                                    {item.sources && (
                                                        <span className="flex items-center gap-1">
                                                            <FileText size={14} />
                                                            {item.sources} source{item.sources !== 1 ? 's' : ''}
                                                        </span>
                                                    )}
                                                </div>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        ) : (
                            <div className="text-center py-12 text-gray-500">
                                <Clock size={48} className="mx-auto mb-4 opacity-50" />
                                <p>Timeline data will be added as more events are verified</p>
                            </div>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
};

export default EventPage;
