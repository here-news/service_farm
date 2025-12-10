import { LitElement, html, css } from 'lit';
import { customElement, property, state } from 'lit/decorators.js';

interface EntityMetadata {
    id: string;
    canonical_id?: string;
    name: string;
    entity_type: string;
    wikidata_qid?: string;
    wikidata_thumbnail?: string;
    wikidata_description?: string;
    description?: string;
    story_count?: number;
}

@customElement('entity-link')
export class EntityLink extends LitElement {
    @property({ type: String }) entityId = '';
    @property({ type: String }) entityName = '';
    @property({ type: String }) entityType = 'unknown';

    @state() showTooltip = false;
    @state() showHeadshot = false;
    @state() metadata: EntityMetadata | null = null;
    @state() loading = false;
    @state() hasLoadedHeadshot = false;
    @state() imageError = false;

    static styles = css`
        :host {
            display: inline;
            position: relative;
        }

        .entity-link {
            position: relative;
            display: inline;
            color: inherit;
            text-decoration: none;
            background: linear-gradient(120deg, #fef3c7 0%, #fde68a 100%);
            padding: 0.125rem 0.25rem;
            border-radius: 3px;
            border-bottom: 2px solid;
            cursor: pointer;
            font-weight: 500;
            transition: all 0.2s;
        }

        .entity-link.person {
            border-bottom-color: #805ad5;
        }

        .entity-link.organization {
            border-bottom-color: #3182ce;
        }

        .entity-link.location {
            border-bottom-color: #38a169;
        }

        .entity-link.unknown {
            border-bottom-color: #a0aec0;
        }

        .entity-link:hover {
            background: linear-gradient(120deg, #fde68a 0%, #fcd34d 100%);
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }

        .entity-link.person:hover {
            border-bottom-color: #9f7aea;
        }

        .entity-link.organization:hover {
            border-bottom-color: #4299e1;
        }

        .entity-link.location:hover {
            border-bottom-color: #48bb78;
        }

        .tooltip {
            position: absolute;
            bottom: 100%;
            left: 50%;
            transform: translateX(-50%) translateY(-8px);
            background: #2d3748;
            color: white;
            padding: 0.75rem;
            border-radius: 8px;
            font-size: 0.875rem;
            white-space: nowrap;
            max-width: 300px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
            z-index: 1000;
            opacity: 0;
            pointer-events: none;
            transition: opacity 0.2s;
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
            border-top-color: #2d3748;
        }

        .tooltip-content {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }

        .tooltip-name {
            font-weight: 600;
            font-size: 1rem;
        }

        .tooltip-badge {
            display: inline-block;
            padding: 0.125rem 0.5rem;
            border-radius: 12px;
            font-size: 0.75rem;
            font-weight: 500;
        }

        .tooltip-badge.person {
            background: #805ad5;
        }

        .tooltip-badge.organization {
            background: #3182ce;
        }

        .tooltip-badge.location {
            background: #38a169;
        }

        .tooltip-desc {
            font-size: 0.8125rem;
            color: #e2e8f0;
            white-space: normal;
            max-width: 280px;
        }

        .tooltip-qid {
            font-size: 0.75rem;
            color: #a0aec0;
            font-family: monospace;
        }

        .headshot {
            position: absolute;
            bottom: 100%;
            left: -48px;
            width: 48px;
            height: 48px;
            border-radius: 50%;
            object-fit: cover;
            border: 3px solid;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
            opacity: 0;
            transform: scale(0.8) translateY(8px);
            transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
            z-index: 999;
            pointer-events: none;
        }

        .headshot.show {
            opacity: 1;
            transform: scale(1) translateY(-8px);
        }

        .headshot.person {
            border-color: #805ad5;
        }

        .headshot.organization {
            border-color: #3182ce;
        }

        .headshot.location {
            border-color: #38a169;
        }
    `;

    connectedCallback() {
        super.connectedCallback();
        // Lazy load metadata on hover
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
        if (!this.metadata && !this.loading && this.entityId) {
            await this.loadMetadata();
        }

        // Show headshot with delay
        if (this.metadata?.wikidata_thumbnail && !this.hasLoadedHeadshot) {
            this.hasLoadedHeadshot = true;
            setTimeout(() => {
                this.showHeadshot = true;
            }, 100);
        }
    }

    handleMouseLeave() {
        this.showTooltip = false;
        this.showHeadshot = false;
    }

    async loadMetadata() {
        this.loading = true;
        try {
            const response = await fetch(`/api/stories/entity/${this.entityId}`);
            if (response.ok) {
                const data = await response.json();
                this.metadata = data.entity;
                this.entityType = data.entity.entity_type || this.entityType;
            }
        } catch (error) {
            console.error('Failed to load entity metadata:', error);
        } finally {
            this.loading = false;
        }
    }

    handleImageError() {
        this.imageError = true;
        this.showHeadshot = false;
    }

    handleClick(e: Event) {
        // Navigate to entity page (TODO: implement entity page route)
        e.preventDefault();
        console.log('Navigate to entity:', this.entityId);
    }

    render() {
        const typeClass = this.entityType.toLowerCase();

        return html`
            <span class="entity-link ${typeClass}" @click=${this.handleClick}>
                ${this.entityName}
            </span>

            ${this.showTooltip && this.metadata ? html`
                <div class="tooltip ${this.showTooltip ? 'show' : ''}">
                    <div class="tooltip-content">
                        <div class="tooltip-name">${this.metadata.name}</div>
                        <span class="tooltip-badge ${typeClass}">${this.metadata.entity_type}</span>
                        ${this.metadata.wikidata_qid ? html`
                            <div class="tooltip-qid">${this.metadata.wikidata_qid}</div>
                        ` : ''}
                        ${this.metadata.description ? html`
                            <div class="tooltip-desc">${this.metadata.description}</div>
                        ` : ''}
                    </div>
                </div>
            ` : ''}

            ${this.showHeadshot && this.metadata?.wikidata_thumbnail && !this.imageError ? html`
                <img
                    class="headshot ${typeClass} ${this.showHeadshot ? 'show' : ''}"
                    src="${this.metadata.wikidata_thumbnail}"
                    alt="${this.metadata.name}"
                    @error=${this.handleImageError}
                />
            ` : ''}
        `;
    }
}
