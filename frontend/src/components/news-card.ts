import { LitElement, html, css } from 'lit';
import { customElement, property } from 'lit/decorators.js';

interface Person {
    id: string;
    name: string;
    image: string;
}

interface Story {
    story_id: string;
    title: string;
    description?: string;
    created_at?: string;
    last_updated?: string;
    people?: Person[];
    cover_image?: string;
    claim_count?: number;
    coherence?: number;
    health_indicator?: string;
}

@customElement('news-card')
export class NewsCard extends LitElement {
    @property({ type: Object }) story!: Story;

    static styles = css`
        :host {
            display: block;
        }

        .card {
            position: relative;
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            padding: 1.25rem;
            transition: all 0.2s;
            cursor: pointer;
            overflow: hidden;
            background: linear-gradient(to bottom right, #eff6ff 0%, #ffffff 50%, #faf5ff 100%);
        }

        .card-background {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            z-index: 0;
        }

        .card-gradient-overlay {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(to bottom right, rgba(239, 246, 255, 0.85) 0%, rgba(255, 255, 255, 0.85) 50%, rgba(250, 245, 255, 0.85) 100%);
            z-index: 1;
        }

        /* Blur effect layer - on top of gradient */
        .card-blur-overlay {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            z-index: 2;
            pointer-events: none;
        }

        /* Blur from top-left to bottom-right (image clearer on bottom-right) */
        .card-blur-overlay.blur-tl-br {
            backdrop-filter: blur(0px);
            -webkit-backdrop-filter: blur(0px);
            mask-image: linear-gradient(135deg,
                rgba(255, 255, 255, 1) 0%,
                rgba(255, 255, 255, 0.7) 40%,
                rgba(255, 255, 255, 0.3) 70%,
                rgba(255, 255, 255, 0) 100%
            );
            -webkit-mask-image: linear-gradient(135deg,
                rgba(255, 255, 255, 1) 0%,
                rgba(255, 255, 255, 0.7) 40%,
                rgba(255, 255, 255, 0.3) 70%,
                rgba(255, 255, 255, 0) 100%
            );
            background: linear-gradient(135deg,
                rgba(239, 246, 255, 0.6) 0%,
                rgba(255, 255, 255, 0.4) 40%,
                rgba(255, 255, 255, 0.2) 70%,
                transparent 100%
            );
        }

        /* Blur from bottom-right to top-left (image clearer on top-left) */
        .card-blur-overlay.blur-br-tl {
            backdrop-filter: blur(0px);
            -webkit-backdrop-filter: blur(0px);
            mask-image: linear-gradient(315deg,
                rgba(255, 255, 255, 1) 0%,
                rgba(255, 255, 255, 0.7) 40%,
                rgba(255, 255, 255, 0.3) 70%,
                rgba(255, 255, 255, 0) 100%
            );
            -webkit-mask-image: linear-gradient(315deg,
                rgba(255, 255, 255, 1) 0%,
                rgba(255, 255, 255, 0.7) 40%,
                rgba(255, 255, 255, 0.3) 70%,
                rgba(255, 255, 255, 0) 100%
            );
            background: linear-gradient(315deg,
                rgba(239, 246, 255, 0.6) 0%,
                rgba(255, 255, 255, 0.4) 40%,
                rgba(255, 255, 255, 0.2) 70%,
                transparent 100%
            );
        }

        .card-content-wrapper {
            position: relative;
            z-index: 10;
        }

        .card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.08);
            border-color: #667eea;
        }

        .card-header {
            display: flex;
            align-items: flex-start;
            margin-bottom: 0.75rem;
        }

        .card-content {
            flex: 1;
            min-width: 0;
        }

        .title {
            font-size: 1.125rem;
            font-weight: 600;
            color: #111827;
            line-height: 1.4;
            margin-bottom: 0.5rem;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
            overflow: hidden;
        }

        .description {
            font-size: 0.875rem;
            color: #6b7280;
            line-height: 1.5;
            margin-bottom: 0.75rem;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
            overflow: hidden;
        }

        .meta-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.75rem;
            flex-wrap: wrap;
        }

        .entity-chips {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            flex-wrap: wrap;
        }

        .entity-chip {
            display: flex;
            align-items: center;
            gap: 0.375rem;
            padding: 0.25rem 0.625rem;
            background: #f3f4f6;
            border: 1px solid #e5e7eb;
            border-radius: 6px;
            font-size: 0.75rem;
            color: #374151;
            font-weight: 500;
        }

        .entity-avatar {
            width: 20px;
            height: 20px;
            border-radius: 50%;
            object-fit: cover;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.625rem;
            font-weight: 600;
        }

        .entity-more {
            padding: 0.25rem 0.625rem;
            background: #e5e7eb;
            border-radius: 6px;
            font-size: 0.75rem;
            color: #6b7280;
            font-weight: 500;
        }

        .time-ago {
            font-size: 0.75rem;
            color: #9ca3af;
            font-style: italic;
            white-space: nowrap;
        }

        .stats-row {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin-top: 0.5rem;
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

        .health-badge {
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
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

        @media (max-width: 640px) {
            .title {
                font-size: 1rem;
            }
        }
    `;

    getTimeAgo(dateString?: string): string {
        if (!dateString) return '';

        const date = new Date(dateString);
        const now = new Date();
        const diffMs = now.getTime() - date.getTime();
        const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
        const diffDays = Math.floor(diffHours / 24);

        if (diffHours < 1) return 'Just now';
        if (diffHours < 24) return `${diffHours}h ago`;
        if (diffDays < 7) return `${diffDays}d ago`;
        return date.toLocaleDateString();
    }

    getInitials(name: string): string {
        return name
            .split(' ')
            .map(n => n[0])
            .join('')
            .toUpperCase()
            .slice(0, 2);
    }

    handleClick() {
        this.dispatchEvent(new CustomEvent('story-click', {
            detail: {
                storyId: this.story.story_id,
                title: this.story.title
            },
            bubbles: true,
            composed: true
        }));
    }

    getCoherenceClass(coherence?: number): string {
        if (!coherence) return 'coherence-low';
        if (coherence >= 60) return 'coherence-high';
        if (coherence >= 30) return 'coherence-medium';
        return 'coherence-low';
    }

    getRandomBlurClass(): string {
        // Randomly choose between the two blur directions
        return Math.random() > 0.5 ? 'blur-tl-br' : 'blur-br-tl';
    }

    render() {
        const timeAgo = this.getTimeAgo(this.story.last_updated || this.story.created_at);
        const people = this.story.people || [];
        const displayPeople = people.slice(0, 2);
        const remainingCount = people.length - 2;

        const healthClass = this.story.health_indicator === 'healthy' ? 'health-healthy' :
                           this.story.health_indicator === 'growing' ? 'health-growing' :
                           'health-stale';

        const healthText = this.story.health_indicator === 'healthy' ? '✓ Healthy' :
                          this.story.health_indicator === 'growing' ? '📈 Growing' :
                          '⏸️ Stale';

        const coherence = this.story.coherence;
        const coherenceClass = this.getCoherenceClass(coherence);

        const hasCoverImage = !!(this.story.cover_image && this.story.cover_image.trim().length > 0);
        const blurClass = this.getRandomBlurClass();

        return html`
            <div class="card ${hasCoverImage ? 'has-image' : ''}" @click=${this.handleClick}>
                ${hasCoverImage ? html`
                    <div class="card-background" style="background-image: url('${this.story.cover_image}')"></div>
                    <div class="card-gradient-overlay"></div>
                    <div class="card-blur-overlay ${blurClass}"></div>
                ` : ''}

                <div class="card-content-wrapper">
                    <div class="card-header">
                        <div class="card-content">
                            <h3 class="title">${this.story.title}</h3>
                        </div>
                    </div>

                ${this.story.description ? html`
                    <p class="description">${this.story.description}</p>
                ` : ''}

                <div class="meta-row">
                    ${displayPeople.length > 0 ? html`
                        <div class="entity-chips">
                            ${displayPeople.map(person => html`
                                <div class="entity-chip" title="${person.name}">
                                    ${person.image ? html`
                                        <img src="${person.image}" alt="${person.name}" class="entity-avatar" />
                                    ` : html`
                                        <div class="entity-avatar">${this.getInitials(person.name)}</div>
                                    `}
                                    <span>${person.name.split(' ')[0]}</span>
                                </div>
                            `)}
                            ${remainingCount > 0 ? html`
                                <div class="entity-more">+${remainingCount}</div>
                            ` : ''}
                        </div>
                    ` : html`<div></div>`}

                    ${timeAgo ? html`
                        <span class="time-ago">🕐 ${timeAgo}</span>
                    ` : ''}
                </div>

                <div class="stats-row">
                    ${this.story.claim_count !== undefined && this.story.claim_count > 0 ? html`
                        <span class="stat-badge">📋 ${this.story.claim_count} claim${this.story.claim_count !== 1 ? 's' : ''}</span>
                    ` : ''}

                    ${coherence !== undefined ? html`
                        <span class="stat-badge ${coherenceClass}" title="Coherence score: ${Math.round(coherence)}/100">
                            💡 ${Math.round(coherence)}% coherence
                        </span>
                    ` : ''}

                    ${this.story.health_indicator ? html`
                        <span class="health-badge ${healthClass}">${healthText}</span>
                    ` : ''}
                </div>
                </div>
            </div>
        `;
    }
}
