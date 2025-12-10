import { LitElement, html, css, nothing } from 'lit';
import { customElement, state } from 'lit/decorators.js';
import { unsafeHTML } from 'lit/directives/unsafe-html.js';
import './entity-tag';
import './entity-link';
import './story-reference';
import './story-chat-button';
import './story-chat-sidebar';
import { parseStoryUrl } from '../utils/storyUrl';

interface Entity {
    id: string;
    canonical_id?: string;
    name: string;
    wikidata_qid?: string;
    wikidata_thumbnail?: string;
    wikidata_description?: string;
    description?: string;
}

interface Claim {
    id: string;
    text: string;
    confidence?: number;
}

interface Artifact {
    url: string;
    title: string;
    domain: string;
    published_at?: string;
}

interface RelatedStory {
    id: string;
    title: string;
    shared_entities: number;
}

interface StoryData {
    id: string;
    title: string;
    description: string;
    content: string;
    created_at: string;
    health_indicator?: string;
    tcf_score: number;
    coherence: number;
    timely: number;
    funding: number;
    explanation: string;
    claim_count: number;
    artifact_count: number;
    coherence_breakdown?: any;
    entities: {
        people: Entity[];
        organizations: Entity[];
        locations: Entity[];
    };
    claims: Claim[];
    artifacts: Artifact[];
    related_stories: RelatedStory[];
}

@customElement('story-detail')
export class StoryDetail extends LitElement {
    @state() story: StoryData | null = null;
    @state() loading = true;
    @state() error: string | null = null;
    @state() storyId: string = '';
    @state() userCredits = 0;
    @state() isChatUnlocked = false;
    @state() isChatOpen = false;

    static styles = css`
        :host {
            display: block;
            min-height: 100vh;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
        }

        .story-container {
            background: white;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            overflow: hidden;
        }

        .story-header {
            padding: 2rem;
            border-bottom: 1px solid #e5e7eb;
        }

        .story-title {
            font-size: 2rem;
            font-weight: 700;
            color: #111827;
            margin-bottom: 1rem;
            line-height: 1.3;
        }

        .story-meta {
            display: flex;
            gap: 1rem;
            flex-wrap: wrap;
            align-items: center;
            color: #6b7280;
            font-size: 0.875rem;
        }

        .health-badge {
            padding: 0.25rem 0.75rem;
            border-radius: 12px;
            font-size: 0.75rem;
            font-weight: 600;
        }

        .health-healthy {
            background: #d1fae5;
            color: #065f46;
        }

        .health-growing {
            background: #dbeafe;
            color: #1e40af;
        }

        .health-stale {
            background: #fee2e2;
            color: #991b1b;
        }

        .tcf-section {
            padding: 1.5rem 2rem;
            background: #f9fafb;
            border-bottom: 1px solid #e5e7eb;
        }

        .tcf-score {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 1.25rem;
            font-weight: 700;
            color: #667eea;
            margin-bottom: 1rem;
        }

        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 1rem;
        }

        .metric {
            text-align: center;
        }

        .metric-value {
            font-size: 1.5rem;
            font-weight: 700;
            color: #111827;
        }

        .metric-label {
            font-size: 0.75rem;
            color: #6b7280;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-top: 0.25rem;
        }

        .explanation {
            margin-top: 1rem;
            padding: 1rem;
            background: white;
            border-radius: 8px;
            font-size: 0.875rem;
            color: #374151;
            border-left: 3px solid #667eea;
        }

        .story-content {
            padding: 2rem;
        }

        .content-text {
            font-size: 1.125rem;
            line-height: 1.8;
            color: #374151;
            margin-bottom: 2rem;
        }

        .content-text .entity {
            background: linear-gradient(120deg, #fef3c7 0%, #fde68a 100%);
            padding: 0.125rem 0.25rem;
            border-radius: 3px;
            cursor: pointer;
            font-weight: 500;
            transition: all 0.2s;
        }

        .content-text .entity:hover {
            background: linear-gradient(120deg, #fde68a 0%, #fcd34d 100%);
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }

        .section {
            margin-bottom: 2rem;
        }

        .section-title {
            font-size: 1.25rem;
            font-weight: 700;
            color: #111827;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .entity-section {
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }

        .entity-group {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }

        .entity-group-title {
            font-size: 0.875rem;
            font-weight: 600;
            color: #6b7280;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .entity-list {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
        }

        .claim-list {
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }

        .claim-item {
            padding: 1rem;
            background: #f9fafb;
            border-radius: 8px;
            border-left: 3px solid #667eea;
        }

        .claim-text {
            font-size: 0.9375rem;
            color: #374151;
            line-height: 1.6;
        }

        .claim-confidence {
            font-size: 0.75rem;
            color: #6b7280;
            margin-top: 0.5rem;
        }

        .artifact-list {
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
        }

        .artifact-item {
            display: flex;
            flex-direction: column;
            gap: 0.25rem;
            padding: 1rem;
            background: #f9fafb;
            border-radius: 8px;
            text-decoration: none;
            color: inherit;
            transition: all 0.2s;
        }

        .artifact-item:hover {
            background: #f3f4f6;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        }

        .artifact-title {
            font-weight: 600;
            color: #667eea;
            font-size: 0.9375rem;
        }

        .artifact-meta {
            font-size: 0.8125rem;
            color: #6b7280;
        }

        .related-stories {
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
        }

        .related-story {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1rem;
            background: #f9fafb;
            border-radius: 8px;
            cursor: pointer;
            text-decoration: none;
            color: inherit;
            transition: all 0.2s;
        }

        .related-story:hover {
            background: #f3f4f6;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        }

        .related-title {
            font-weight: 600;
            color: #374151;
            font-size: 0.9375rem;
        }

        .related-badge {
            background: #dbeafe;
            color: #1e40af;
            padding: 0.25rem 0.75rem;
            border-radius: 12px;
            font-size: 0.75rem;
            font-weight: 600;
        }

        .loading {
            text-align: center;
            padding: 4rem;
            color: white;
            font-size: 1.125rem;
        }

        .error {
            background: #fee;
            color: #c33;
            padding: 1rem;
            border-radius: 8px;
            margin: 2rem;
        }

        @media (max-width: 768px) {
            .container {
                padding: 1rem;
            }

            .story-title {
                font-size: 1.5rem;
            }

            .story-header,
            .tcf-section,
            .story-content {
                padding: 1rem;
            }

            .content-text {
                font-size: 1rem;
            }
        }
    `;

    connectedCallback() {
        super.connectedCallback();
        // Extract story ID from URL (supports both /story/:id and /story/:id/:slug)
        const storyId = parseStoryUrl(window.location.pathname);
        if (storyId) {
            this.storyId = storyId;
            this.loadStory();
            this.loadUserCredits();
            this.checkChatUnlocked();
        } else {
            this.error = 'Invalid story URL';
        }
    }

    async loadStory() {
        try {
            this.loading = true;
            const response = await fetch(`/api/stories/${this.storyId}`);
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            const data = await response.json();
            this.story = data.story;
            this.loading = false;
        } catch (err) {
            this.error = err instanceof Error ? err.message : 'Failed to load story';
            this.loading = false;
        }
    }

    async loadUserCredits() {
        try {
            const response = await fetch('/api/chat/credits', { credentials: 'include' });
            if (response.ok) {
                const data = await response.json();
                this.userCredits = data.credits;
            }
        } catch (error) {
            console.error('Failed to load user credits:', error);
        }
    }

    async checkChatUnlocked() {
        try {
            const response = await fetch(`/api/chat/session/${this.storyId}`, { credentials: 'include' });
            if (response.ok) {
                const session = await response.json();
                this.isChatUnlocked = session !== null;
            }
        } catch (error) {
            console.error('Failed to check chat status:', error);
        }
    }

    handleOpenChat() {
        this.isChatOpen = true;
    }

    handleCloseChat() {
        this.isChatOpen = false;
    }

    handleChatUnlocked() {
        this.isChatUnlocked = true;
        this.loadUserCredits(); // Refresh credits
    }

    buildEntityLookup(): Map<string, Entity> {
        const lookup = new Map<string, Entity>();

        if (!this.story) return lookup;

        // Build lookup by both ID and name
        const allEntities = [
            ...this.story.entities.people.map(e => ({ ...e, type: 'person' })),
            ...this.story.entities.organizations.map(e => ({ ...e, type: 'organization' })),
            ...this.story.entities.locations.map(e => ({ ...e, type: 'location' }))
        ];

        allEntities.forEach(entity => {
            if (entity.id) lookup.set(entity.id, entity);
            if (entity.canonical_id) lookup.set(entity.canonical_id, entity);
            if (entity.name) lookup.set(entity.name, entity);
        });

        return lookup;
    }

    parseEntityMarkup(text: string): string {
        if (!text) return '';

        const entityLookup = this.buildEntityLookup();

        // First, strip out page citations {{cite:uuid,uuid,...}} since we don't have page metadata
        // These will be replaced with superscript citation numbers in a future update
        let result = text.replace(/\{\{cite:[^}]+\}\}/g, '');

        // Parse story citations: {{story:story_id}} or {{story:story_id:reference_text}}
        result = result.replace(/\{\{story:([^}]+)\}\}/g, (_match, content) => {
            const parts = content.split(':').map((s: string) => s.trim());
            const storyId = parts[0];
            const referenceText = parts.length > 1 ? parts.slice(1).join(':') : 'related story';

            return `<story-reference storyid="${storyId}" referencetext="${referenceText}"></story-reference>`;
        });

        // Then parse [[Entity Name|canonical_id]] or [[Entity Name]] markup
        result = result.replace(/\[\[([^\]]+)\]\]/g, (_match, content) => {
            // Parse new format: [[Entity Name|canonical_id]]
            let entityName = content;
            let entityId = '';
            let entityType = 'unknown';

            if (content.includes('|')) {
                const parts = content.split('|').map((s: string) => s.trim());
                entityName = parts[0];
                entityId = parts[1];
            } else {
                // Legacy format: [[Entity Name]]
                entityName = content;
                entityId = content; // Try name as ID
            }

            // Look up entity metadata
            const entity = entityLookup.get(entityId) || entityLookup.get(entityName);
            if (entity) {
                entityId = entity.id || entity.canonical_id || entityId;
                entityType = (entity as any).type || 'unknown';
            }

            return `<entity-link entityid="${entityId}" entityname="${entityName}" entitytype="${entityType}"></entity-link>`;
        });

        return result;
    }

    formatTimeAgo(dateStr: string): string {
        const date = new Date(dateStr);
        const now = new Date();
        const diffMs = now.getTime() - date.getTime();
        const diffMins = Math.floor(diffMs / 60000);
        const diffHours = Math.floor(diffMs / 3600000);
        const diffDays = Math.floor(diffMs / 86400000);

        if (diffMins < 60) return `${diffMins} min ago`;
        if (diffHours < 24) return `${diffHours} hour${diffHours > 1 ? 's' : ''} ago`;
        if (diffDays < 7) return `${diffDays} day${diffDays > 1 ? 's' : ''} ago`;
        return date.toLocaleDateString();
    }

    render() {
        if (this.loading) {
            return html`<div class="loading">Loading story...</div>`;
        }

        if (this.error) {
            return html`<div class="error">Error: ${this.error}</div>`;
        }

        if (!this.story) {
            return html`<div class="error">Story not found</div>`;
        }

        const healthClass = this.story.health_indicator === 'healthy' ? 'health-healthy' :
                           this.story.health_indicator === 'growing' ? 'health-growing' :
                           'health-stale';

        const healthText = this.story.health_indicator === 'healthy' ? '✓ Healthy' :
                          this.story.health_indicator === 'growing' ? '📈 Growing' :
                          '⏸️ Stale';

        return html`
            <div class="container">
                <div class="story-container">
                    <div class="story-header">
                        <h1 class="story-title">${this.story.title}</h1>
                        <div class="story-meta">
                            ${this.story.health_indicator ? html`
                                <span class="health-badge ${healthClass}">${healthText}</span>
                            ` : nothing}
                            ${this.story.created_at && this.formatTimeAgo(this.story.created_at) !== 'Invalid Date' ? html`
                                <span>${this.formatTimeAgo(this.story.created_at)}</span>
                            ` : nothing}
                            ${this.story.claim_count && this.story.claim_count > 0 ? html`
                                <span>${this.story.claim_count} claim${this.story.claim_count !== 1 ? 's' : ''}</span>
                            ` : nothing}
                            ${this.story.artifact_count && this.story.artifact_count > 0 ? html`
                                <span>${this.story.artifact_count} source${this.story.artifact_count !== 1 ? 's' : ''}</span>
                            ` : nothing}
                        </div>
                    </div>

                    ${this.story.tcf_score && this.story.coherence ? html`
                        <div class="tcf-section">
                            <div class="tcf-score">
                                TCF Score: ${Math.round(this.story.tcf_score)}
                            </div>
                            <div class="metrics-grid">
                                <div class="metric">
                                    <div class="metric-value">${Math.round(this.story.coherence)}</div>
                                    <div class="metric-label">Coherence</div>
                                </div>
                                ${this.story.timely ? html`
                                    <div class="metric">
                                        <div class="metric-value">${Math.round(this.story.timely)}</div>
                                        <div class="metric-label">Timely</div>
                                    </div>
                                ` : nothing}
                                ${this.story.funding !== undefined ? html`
                                    <div class="metric">
                                        <div class="metric-value">${Math.round(this.story.funding)}</div>
                                        <div class="metric-label">Funding</div>
                                    </div>
                                ` : nothing}
                            </div>
                            ${this.story.explanation ? html`
                                <div class="explanation">
                                    <strong>Why this ranks:</strong> ${this.story.explanation}
                                </div>
                            ` : nothing}
                        </div>
                    ` : nothing}

                    <div class="story-content">
                        ${this.story.description ? html`
                            <div class="content-text">
                                ${unsafeHTML(this.parseEntityMarkup(this.story.description))}
                            </div>
                        ` : nothing}

                        ${this.story.content ? html`
                            <div class="content-text">
                                ${unsafeHTML(this.parseEntityMarkup(this.story.content))}
                            </div>
                        ` : nothing}

                        ${this.renderEntities()}
                        ${this.renderClaims()}
                        ${this.renderArtifacts()}
                        ${this.renderRelatedStories()}

                        <!-- Discussion Section -->
                        <comment-thread .storyId=${this.storyId}></comment-thread>
                    </div>
                </div>
            </div>

            <!-- Premium Chat Components -->
            <story-chat-button
                .storyId=${this.storyId}
                .userCredits=${this.userCredits}
                .isUnlocked=${this.isChatUnlocked}
                @open-chat=${this.handleOpenChat}
                @chat-unlocked=${this.handleChatUnlocked}
            ></story-chat-button>

            <story-chat-sidebar
                .storyId=${this.storyId}
                ?isopen=${this.isChatOpen}
                @close-chat=${this.handleCloseChat}
            ></story-chat-sidebar>
        `;
    }

    renderEntities() {
        if (!this.story) return nothing;

        const { people, organizations, locations } = this.story.entities;
        const hasEntities = people.length > 0 || organizations.length > 0 || locations.length > 0;

        if (!hasEntities) return nothing;

        return html`
            <div class="section">
                <h2 class="section-title">🏷️ Entities</h2>
                <div class="entity-section">
                    ${people.length > 0 ? html`
                        <div class="entity-group">
                            <div class="entity-group-title">People</div>
                            <div class="entity-list">
                                ${people.map(p => html`
                                    <entity-tag .name=${p.name} type="person"></entity-tag>
                                `)}
                            </div>
                        </div>
                    ` : nothing}

                    ${organizations.length > 0 ? html`
                        <div class="entity-group">
                            <div class="entity-group-title">Organizations</div>
                            <div class="entity-list">
                                ${organizations.map(o => html`
                                    <entity-tag .name=${o.name} type="organization"></entity-tag>
                                `)}
                            </div>
                        </div>
                    ` : nothing}

                    ${locations.length > 0 ? html`
                        <div class="entity-group">
                            <div class="entity-group-title">Locations</div>
                            <div class="entity-list">
                                ${locations.map(l => html`
                                    <entity-tag .name=${l.name} type="location"></entity-tag>
                                `)}
                            </div>
                        </div>
                    ` : nothing}
                </div>
            </div>
        `;
    }

    renderClaims() {
        if (!this.story || this.story.claims.length === 0) return nothing;

        return html`
            <div class="section">
                <h2 class="section-title">📋 Claims (${this.story.claims.length})</h2>
                <div class="claim-list">
                    ${this.story.claims.map(claim => html`
                        <div class="claim-item">
                            <div class="claim-text">${claim.text}</div>
                            ${claim.confidence ? html`
                                <div class="claim-confidence">
                                    Confidence: ${(claim.confidence * 100).toFixed(0)}%
                                </div>
                            ` : nothing}
                        </div>
                    `)}
                </div>
            </div>
        `;
    }

    renderArtifacts() {
        if (!this.story || this.story.artifacts.length === 0) return nothing;

        return html`
            <div class="section">
                <h2 class="section-title">📰 Sources (${this.story.artifacts.length})</h2>
                <div class="artifact-list">
                    ${this.story.artifacts.map(artifact => html`
                        <a href="${artifact.url}" target="_blank" rel="noopener noreferrer" class="artifact-item">
                            <div class="artifact-title">${artifact.title}</div>
                            <div class="artifact-meta">
                                ${artifact.domain}
                                ${artifact.published_at ? ` · ${this.formatTimeAgo(artifact.published_at)}` : ''}
                            </div>
                        </a>
                    `)}
                </div>
            </div>
        `;
    }

    renderRelatedStories() {
        if (!this.story || this.story.related_stories.length === 0) return nothing;

        return html`
            <div class="section">
                <h2 class="section-title">🔗 Related Stories (${this.story.related_stories.length})</h2>
                <div class="related-stories">
                    ${this.story.related_stories.map(related => html`
                        <a href="/story/${related.id}" class="related-story">
                            <div class="related-title">${related.title}</div>
                            <div class="related-badge">
                                ${related.shared_entities} shared entities
                            </div>
                        </a>
                    `)}
                </div>
            </div>
        `;
    }
}
