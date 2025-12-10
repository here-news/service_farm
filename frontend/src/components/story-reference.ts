import { LitElement, html, css } from 'lit';
import { customElement, property, state } from 'lit/decorators.js';

interface StoryMetadata {
    id: string;
    title: string;
    description?: string;
    created_at?: string;
    claim_count?: number;
    coherence?: number;
    health_indicator?: string;
}

@customElement('story-reference')
export class StoryReference extends LitElement {
    @property({ type: String }) storyId = '';
    @property({ type: String }) referenceText = '';

    @state() showTooltip = false;
    @state() metadata: StoryMetadata | null = null;
    @state() loading = false;

    static styles = css`
        :host {
            display: inline;
            position: relative;
        }

        .story-ref {
            color: #3182ce;
            text-decoration: underline;
            text-decoration-style: dotted;
            cursor: pointer;
            font-weight: 500;
            transition: all 0.2s;
        }

        .story-ref:hover {
            color: #2563eb;
            text-decoration-style: solid;
        }

        .tooltip {
            position: absolute;
            bottom: 100%;
            left: 50%;
            transform: translateX(-50%) translateY(-8px);
            background: white;
            color: #374151;
            padding: 1rem;
            border-radius: 8px;
            width: 320px;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.15);
            z-index: 1000;
            opacity: 0;
            pointer-events: none;
            transition: opacity 0.2s;
            border: 1px solid #e5e7eb;
        }

        .tooltip.show {
            opacity: 1;
        }

        .tooltip::after {
            content: '';
            position: absolute;
            top: 100%;
            left: 50%;
            transform: translateX(-50%);
            border: 6px solid transparent;
            border-top-color: white;
        }

        .tooltip::before {
            content: '';
            position: absolute;
            top: 100%;
            left: 50%;
            transform: translateX(-50%);
            border: 7px solid transparent;
            border-top-color: #e5e7eb;
            margin-top: 1px;
        }

        .tooltip-title {
            font-weight: 600;
            font-size: 0.9375rem;
            color: #111827;
            margin-bottom: 0.5rem;
            line-height: 1.3;
        }

        .tooltip-desc {
            font-size: 0.8125rem;
            color: #6b7280;
            line-height: 1.5;
            margin-bottom: 0.75rem;
        }

        .tooltip-stats {
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
        }

        .stat-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.25rem;
            padding: 0.25rem 0.5rem;
            background: #f9fafb;
            border: 1px solid #e5e7eb;
            border-radius: 4px;
            font-size: 0.75rem;
            color: #6b7280;
            font-weight: 500;
        }

        .stat-badge.coherence-high {
            background: #d1fae5;
            border-color: #a7f3d0;
            color: #065f46;
        }

        .stat-badge.coherence-medium {
            background: #fef3c7;
            border-color: #fde68a;
            color: #92400e;
        }

        .stat-badge.coherence-low {
            background: #fee2e2;
            border-color: #fecaca;
            color: #991b1b;
        }

        .loading {
            font-size: 0.8125rem;
            color: #9ca3af;
            font-style: italic;
        }

        .click-hint {
            margin-top: 0.5rem;
            padding-top: 0.5rem;
            border-top: 1px solid #e5e7eb;
            font-size: 0.75rem;
            color: #9ca3af;
            text-align: center;
        }
    `;

    connectedCallback() {
        super.connectedCallback();
        this.addEventListener('mouseenter', this.handleMouseEnter);
        this.addEventListener('mouseleave', this.handleMouseLeave);
    }

    disconnectedCallback() {
        super.disconnectedCallback();
        this.removeEventListener('mouseenter', this.handleMouseEnter);
        this.removeEventListener('mouseleave', this.handleMouseLeave);
    }

    async handleMouseEnter() {
        this.showTooltip = true;

        // Load metadata if not already loaded
        if (!this.metadata && !this.loading && this.storyId) {
            await this.loadMetadata();
        }
    }

    handleMouseLeave() {
        this.showTooltip = false;
    }

    async loadMetadata() {
        this.loading = true;
        try {
            const response = await fetch(`/api/stories/${this.storyId}`);
            if (response.ok) {
                const data = await response.json();
                const story = data.story;
                this.metadata = {
                    id: story.id,
                    title: story.title,
                    description: story.description,
                    created_at: story.created_at,
                    claim_count: story.claim_count,
                    coherence: story.coherence,
                    health_indicator: story.health_indicator
                };
            }
        } catch (error) {
            console.error('Failed to load story metadata:', error);
        } finally {
            this.loading = false;
        }
    }

    handleClick(e: Event) {
        // Navigate to story page
        e.preventDefault();
        window.location.href = `/story/${this.storyId}`;
    }

    getCoherenceClass(coherence?: number): string {
        if (!coherence) return 'coherence-low';
        if (coherence >= 60) return 'coherence-high';
        if (coherence >= 30) return 'coherence-medium';
        return 'coherence-low';
    }

    formatTimeAgo(dateStr?: string): string {
        if (!dateStr) return '';
        const date = new Date(dateStr);
        const now = new Date();
        const diffMs = now.getTime() - date.getTime();
        const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
        const diffDays = Math.floor(diffHours / 24);

        if (diffHours < 1) return 'Just now';
        if (diffHours < 24) return `${diffHours}h ago`;
        if (diffDays < 7) return `${diffDays}d ago`;
        return date.toLocaleDateString();
    }

    render() {
        const coherenceClass = this.getCoherenceClass(this.metadata?.coherence);

        return html`
            <span class="story-ref" @click=${this.handleClick}>
                ${this.referenceText}
            </span>

            ${this.showTooltip ? html`
                <div class="tooltip ${this.showTooltip ? 'show' : ''}">
                    ${this.loading ? html`
                        <div class="loading">Loading story preview...</div>
                    ` : this.metadata ? html`
                        <div class="tooltip-title">${this.metadata.title}</div>

                        ${this.metadata.description ? html`
                            <div class="tooltip-desc">
                                ${this.metadata.description.substring(0, 120)}${this.metadata.description.length > 120 ? '...' : ''}
                            </div>
                        ` : ''}

                        <div class="tooltip-stats">
                            ${this.metadata.claim_count !== undefined && this.metadata.claim_count > 0 ? html`
                                <span class="stat-badge">📋 ${this.metadata.claim_count} claims</span>
                            ` : ''}

                            ${this.metadata.coherence !== undefined ? html`
                                <span class="stat-badge ${coherenceClass}">
                                    💡 ${Math.round(this.metadata.coherence)}%
                                </span>
                            ` : ''}

                            ${this.metadata.created_at ? html`
                                <span class="stat-badge">🕐 ${this.formatTimeAgo(this.metadata.created_at)}</span>
                            ` : ''}
                        </div>

                        <div class="click-hint">Click to read full story</div>
                    ` : html`
                        <div class="loading">Story not found</div>
                    `}
                </div>
            ` : ''}
        `;
    }
}
