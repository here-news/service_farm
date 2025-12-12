import React, { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';

interface Claim {
    id: string;
    text: string;
    event_time?: string;
    confidence?: number;
    page_id?: string;
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
}

const EntityPage: React.FC = () => {
    const { entityId } = useParams<{ entityId: string }>();
    const [data, setData] = useState<EntityData | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

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

    const { entity, claims } = data;

    return (
        <div className="min-h-screen bg-gray-50">
            {/* Header */}
            <div className="bg-white border-b">
                <div className="max-w-4xl mx-auto px-4 py-8">
                    <div className="flex items-start gap-6">
                        {/* Image */}
                        {entity.image_url ? (
                            <div className="flex-shrink-0">
                                <img
                                    src={entity.image_url}
                                    alt={entity.canonical_name}
                                    className="w-32 h-32 rounded-lg object-cover shadow-md"
                                    onError={(e) => {
                                        (e.target as HTMLImageElement).style.display = 'none';
                                    }}
                                />
                            </div>
                        ) : (
                            <div className="flex-shrink-0 w-32 h-32 rounded-lg bg-gray-200 flex items-center justify-center">
                                <span className="text-4xl text-gray-400">
                                    {entity.entity_type === 'PERSON' ? '👤' :
                                     entity.entity_type === 'ORGANIZATION' ? '🏢' : '📍'}
                                </span>
                            </div>
                        )}

                        {/* Info */}
                        <div className="flex-1">
                            <div className="flex items-center gap-3 mb-2">
                                <span className={`px-2 py-1 rounded text-sm font-medium ${getEntityTypeColor(entity.entity_type)}`}>
                                    {getEntityTypeLabel(entity.entity_type)}
                                </span>
                                {entity.wikidata_qid && (
                                    <a
                                        href={`https://www.wikidata.org/wiki/${entity.wikidata_qid}`}
                                        target="_blank"
                                        rel="noopener noreferrer"
                                        className="px-2 py-1 rounded text-sm font-medium bg-blue-100 text-blue-800 hover:bg-blue-200 transition-colors"
                                    >
                                        {entity.wikidata_qid} ↗
                                    </a>
                                )}
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
                    </div>
                </div>
            </div>

            {/* Stats */}
            <div className="max-w-4xl mx-auto px-4 py-4">
                <div className="flex gap-4">
                    <div className="bg-white rounded-lg px-4 py-3 shadow-sm">
                        <div className="text-2xl font-bold text-gray-900">{entity.mention_count || 0}</div>
                        <div className="text-sm text-gray-500">Mentions</div>
                    </div>
                    <div className="bg-white rounded-lg px-4 py-3 shadow-sm">
                        <div className="text-2xl font-bold text-gray-900">{claims.length}</div>
                        <div className="text-sm text-gray-500">Related Claims</div>
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

            {/* Claims Section */}
            <div className="max-w-4xl mx-auto px-4 py-6">
                <h2 className="text-xl font-bold text-gray-900 mb-4">
                    Related Claims
                </h2>

                {claims.length === 0 ? (
                    <div className="bg-white rounded-lg p-6 text-center text-gray-500">
                        No claims found for this entity.
                    </div>
                ) : (
                    <div className="space-y-3">
                        {claims.map((claim) => (
                            <div
                                key={claim.id}
                                className="bg-white rounded-lg p-4 shadow-sm border border-gray-100 hover:border-gray-200 transition-colors"
                            >
                                <p className="text-gray-800">{claim.text}</p>
                                <div className="flex items-center gap-4 mt-2 text-sm text-gray-500">
                                    {claim.event_time && (
                                        <span>
                                            {new Date(claim.event_time).toLocaleDateString()}
                                        </span>
                                    )}
                                    {claim.confidence && (
                                        <span>
                                            Confidence: {(claim.confidence * 100).toFixed(0)}%
                                        </span>
                                    )}
                                    <span className="text-gray-400">{claim.id}</span>
                                </div>
                            </div>
                        ))}
                    </div>
                )}
            </div>

            {/* Wikidata Link */}
            {entity.wikidata_qid && (
                <div className="max-w-4xl mx-auto px-4 py-6">
                    <div className="bg-blue-50 rounded-lg p-4 border border-blue-100">
                        <div className="flex items-center gap-3">
                            <span className="text-2xl">🔗</span>
                            <div>
                                <h3 className="font-medium text-gray-900">Wikidata Entry</h3>
                                <a
                                    href={`https://www.wikidata.org/wiki/${entity.wikidata_qid}`}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="text-blue-600 hover:underline"
                                >
                                    View {entity.canonical_name} on Wikidata ({entity.wikidata_qid})
                                </a>
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

export default EntityPage;
