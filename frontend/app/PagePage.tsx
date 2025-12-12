import React, { useState, useEffect, useRef } from 'react';
import { useParams, useLocation, Link } from 'react-router-dom';

interface ClaimEntity {
    id: string;
    canonical_name: string;
    entity_type: string;
    wikidata_qid?: string;
}

interface Claim {
    id: string;
    text: string;
    event_time?: string;
    confidence?: number;
    created_at?: string;
    entities?: ClaimEntity[];
}

interface Entity {
    id: string;
    canonical_name: string;
    entity_type: string;
    wikidata_qid?: string;
    mention_count?: number;
    confidence?: number;
}

interface PageData {
    page: {
        id: string;
        url: string;
        canonical_url?: string;
        title?: string;
        description?: string;
        author?: string;
        thumbnail_url?: string;
        site_name?: string;
        domain?: string;
        language?: string;
        word_count?: number;
        status: string;
        pub_time?: string;
        metadata_confidence?: number;
        semantic_confidence?: number;
        created_at?: string;
        updated_at?: string;
    };
    claims: Claim[];
    claims_count: number;
    entities: Entity[];
    entities_count: number;
}

const PagePage: React.FC = () => {
    const { pageId } = useParams<{ pageId: string }>();
    const location = useLocation();
    const [data, setData] = useState<PageData | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [highlightedClaim, setHighlightedClaim] = useState<string | null>(null);
    const claimRefs = useRef<{ [key: string]: HTMLDivElement | null }>({});

    useEffect(() => {
        loadPage();
    }, [pageId]);

    // Handle anchor scroll when data loads or hash changes
    useEffect(() => {
        if (data && location.hash) {
            const claimId = location.hash.slice(1); // Remove #
            scrollToClaim(claimId);
        }
    }, [data, location.hash]);

    const loadPage = async () => {
        try {
            setLoading(true);
            setError(null);
            const response = await fetch(`/api/page/${pageId}`);
            if (!response.ok) {
                throw new Error('Page not found');
            }
            const result = await response.json();
            setData(result);
        } catch (err: any) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    const scrollToClaim = (claimId: string) => {
        const element = claimRefs.current[claimId];
        if (element) {
            element.scrollIntoView({ behavior: 'smooth', block: 'center' });
            setHighlightedClaim(claimId);
            // Remove highlight after animation
            setTimeout(() => setHighlightedClaim(null), 2000);
        }
    };

    const getStatusColor = (status: string) => {
        switch (status) {
            case 'semantic_complete':
            case 'event_complete':
                return 'bg-green-100 text-green-800';
            case 'extracted':
            case 'knowledge_complete':
                return 'bg-blue-100 text-blue-800';
            case 'preview':
            case 'stub':
                return 'bg-yellow-100 text-yellow-800';
            case 'failed':
            case 'semantic_failed':
                return 'bg-red-100 text-red-800';
            default:
                return 'bg-gray-100 text-gray-800';
        }
    };

    const getStatusLabel = (status: string) => {
        switch (status) {
            case 'semantic_complete': return 'Processed';
            case 'event_complete': return 'Complete';
            case 'extracted': return 'Extracted';
            case 'knowledge_complete': return 'Analyzed';
            case 'preview': return 'Preview';
            case 'stub': return 'Pending';
            case 'failed': return 'Failed';
            case 'semantic_failed': return 'Analysis Failed';
            default: return status;
        }
    };

    const getEntityTypeColor = (type: string) => {
        switch (type) {
            case 'PERSON': return 'bg-purple-100 text-purple-700';
            case 'ORGANIZATION': return 'bg-orange-100 text-orange-700';
            case 'GPE':
            case 'LOC':
            case 'LOCATION': return 'bg-green-100 text-green-700';
            case 'EVENT': return 'bg-pink-100 text-pink-700';
            case 'DATE': return 'bg-cyan-100 text-cyan-700';
            default: return 'bg-gray-100 text-gray-700';
        }
    };

    const getEntityTypeLabel = (type: string) => {
        switch (type) {
            case 'PERSON': return 'Person';
            case 'ORGANIZATION': return 'Org';
            case 'GPE': return 'Place';
            case 'LOC': return 'Location';
            case 'LOCATION': return 'Location';
            case 'EVENT': return 'Event';
            case 'DATE': return 'Date';
            default: return type;
        }
    };

    const copyClaimLink = (claimId: string) => {
        const url = `${window.location.origin}${window.location.pathname}#${claimId}`;
        navigator.clipboard.writeText(url);
    };

    const getFaviconUrl = (domain: string) => {
        return `https://www.google.com/s2/favicons?domain=${domain}&sz=64`;
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
                    <h1 className="text-2xl font-bold text-gray-800 mb-2">Page Not Found</h1>
                    <p className="text-gray-600 mb-4">{error || 'The requested page could not be found.'}</p>
                    <Link to="/" className="text-blue-600 hover:underline">Back to Home</Link>
                </div>
            </div>
        );
    }

    const { page, claims, entities } = data;

    return (
        <div className="min-h-screen bg-gray-50">
            {/* Header with thumbnail and metadata */}
            <div className="bg-white border-b">
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
                    <div className="flex items-start gap-6">
                        {/* Thumbnail */}
                        {page.thumbnail_url ? (
                            <div className="flex-shrink-0">
                                <img
                                    src={page.thumbnail_url}
                                    alt={page.title || 'Page thumbnail'}
                                    className="w-48 h-32 rounded-lg object-cover shadow-md"
                                    onError={(e) => {
                                        (e.target as HTMLImageElement).style.display = 'none';
                                    }}
                                />
                            </div>
                        ) : (
                            <div className="flex-shrink-0 w-48 h-32 rounded-lg bg-gray-200 flex items-center justify-center">
                                <span className="text-4xl text-gray-400">📄</span>
                            </div>
                        )}

                        {/* Page Info */}
                        <div className="flex-1 min-w-0">
                            {/* Source info with favicon */}
                            <div className="flex items-center gap-2 mb-2">
                                {page.domain && (
                                    <img
                                        src={getFaviconUrl(page.domain)}
                                        alt=""
                                        className="w-4 h-4"
                                        onError={(e) => {
                                            (e.target as HTMLImageElement).style.display = 'none';
                                        }}
                                    />
                                )}
                                <span className="text-sm text-gray-500">
                                    {page.site_name || page.domain}
                                </span>
                                <span className={`px-2 py-0.5 rounded text-xs font-medium ${getStatusColor(page.status)}`}>
                                    {getStatusLabel(page.status)}
                                </span>
                            </div>

                            {/* Title */}
                            <h1 className="text-2xl font-bold text-gray-900 mb-2 line-clamp-2">
                                {page.title || 'Untitled Page'}
                            </h1>

                            {/* Description */}
                            {page.description && (
                                <p className="text-gray-600 mb-3 line-clamp-2">
                                    {page.description}
                                </p>
                            )}

                            {/* Meta row: author, date, language */}
                            <div className="flex items-center gap-4 text-sm text-gray-500">
                                {page.author && (
                                    <span>By {page.author}</span>
                                )}
                                {page.pub_time && (
                                    <span>
                                        {new Date(page.pub_time).toLocaleDateString()}
                                    </span>
                                )}
                                {page.language && (
                                    <span className="uppercase">{page.language}</span>
                                )}
                            </div>

                            {/* External link */}
                            <div className="mt-3">
                                <a
                                    href={page.url}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="inline-flex items-center gap-1 text-blue-600 hover:text-blue-800 text-sm"
                                >
                                    View original article
                                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                                    </svg>
                                </a>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {/* Stats */}
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
                <div className="flex gap-4 flex-wrap">
                    <div className="bg-white rounded-lg px-4 py-3 shadow-sm">
                        <div className="text-2xl font-bold text-gray-900">{page.word_count || 0}</div>
                        <div className="text-sm text-gray-500">Words</div>
                    </div>
                    <div className="bg-white rounded-lg px-4 py-3 shadow-sm">
                        <div className="text-2xl font-bold text-gray-900">{data.claims_count}</div>
                        <div className="text-sm text-gray-500">Claims</div>
                    </div>
                    <div className="bg-white rounded-lg px-4 py-3 shadow-sm">
                        <div className="text-2xl font-bold text-gray-900">{data.entities_count}</div>
                        <div className="text-sm text-gray-500">Entities</div>
                    </div>
                    {page.semantic_confidence !== undefined && page.semantic_confidence > 0 && (
                        <div className="bg-white rounded-lg px-4 py-3 shadow-sm">
                            <div className="text-2xl font-bold text-gray-900">
                                {(page.semantic_confidence * 100).toFixed(0)}%
                            </div>
                            <div className="text-sm text-gray-500">Confidence</div>
                        </div>
                    )}
                </div>
            </div>

            {/* Claims Section */}
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
                <h2 className="text-xl font-bold text-gray-900 mb-4">
                    Extracted Claims
                </h2>

                {claims.length === 0 ? (
                    <div className="bg-white rounded-lg p-6 text-center text-gray-500">
                        {page.status === 'semantic_complete'
                            ? 'No claims were extracted from this page.'
                            : 'Claims will appear here once the page is fully processed.'}
                    </div>
                ) : (
                    <div className="space-y-3">
                        {claims.map((claim) => (
                            <div
                                key={claim.id}
                                id={claim.id}
                                ref={(el) => { claimRefs.current[claim.id] = el; }}
                                className={`bg-white rounded-lg p-4 shadow-sm border transition-all duration-300 ${
                                    highlightedClaim === claim.id
                                        ? 'border-blue-500 ring-2 ring-blue-200'
                                        : 'border-gray-100 hover:border-gray-200'
                                }`}
                            >
                                <p className="text-gray-800 mb-3">{claim.text}</p>

                                {/* Inline entities */}
                                {claim.entities && claim.entities.length > 0 && (
                                    <div className="flex flex-wrap gap-1 mb-2">
                                        {claim.entities.map((entity) => (
                                            <Link
                                                key={entity.id}
                                                to={`/entity/${entity.id}`}
                                                className={`px-2 py-0.5 rounded text-xs font-medium hover:opacity-80 transition-opacity ${getEntityTypeColor(entity.entity_type)}`}
                                            >
                                                {entity.canonical_name}
                                            </Link>
                                        ))}
                                    </div>
                                )}

                                {/* Meta row */}
                                <div className="flex items-center justify-between text-sm text-gray-500">
                                    <div className="flex items-center gap-4">
                                        {claim.event_time && (
                                            <span>
                                                {new Date(claim.event_time).toLocaleDateString()}
                                            </span>
                                        )}
                                        {claim.confidence && (
                                            <span>
                                                {(claim.confidence * 100).toFixed(0)}% confidence
                                            </span>
                                        )}
                                    </div>
                                    <button
                                        onClick={() => copyClaimLink(claim.id)}
                                        className="text-gray-400 hover:text-gray-600 transition-colors"
                                        title="Copy link to this claim"
                                    >
                                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" />
                                        </svg>
                                    </button>
                                </div>
                            </div>
                        ))}
                    </div>
                )}
            </div>

            {/* Entities Section */}
            {entities.length > 0 && (
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
                    <h2 className="text-xl font-bold text-gray-900 mb-4">
                        Mentioned Entities
                    </h2>

                    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
                        {entities.map((entity) => (
                            <Link
                                key={entity.id}
                                to={`/entity/${entity.id}`}
                                className="bg-white rounded-lg p-3 shadow-sm border border-gray-100 hover:border-gray-200 hover:shadow transition-all"
                            >
                                <div className="flex items-center gap-2 mb-1">
                                    <span className={`px-1.5 py-0.5 rounded text-xs font-medium ${getEntityTypeColor(entity.entity_type)}`}>
                                        {getEntityTypeLabel(entity.entity_type)}
                                    </span>
                                </div>
                                <div className="font-medium text-gray-900 truncate">
                                    {entity.canonical_name}
                                </div>
                                {entity.mention_count && entity.mention_count > 1 && (
                                    <div className="text-xs text-gray-500 mt-1">
                                        {entity.mention_count} mentions
                                    </div>
                                )}
                            </Link>
                        ))}
                    </div>
                </div>
            )}

            {/* Page ID footer */}
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
                <div className="text-center text-sm text-gray-400">
                    Page ID: {page.id}
                    {page.created_at && (
                        <span className="mx-2">|</span>
                    )}
                    {page.created_at && (
                        <span>Ingested: {new Date(page.created_at).toLocaleDateString()}</span>
                    )}
                </div>
            </div>
        </div>
    );
};

export default PagePage;
