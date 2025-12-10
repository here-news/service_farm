import { LitElement, html, css } from 'lit';
import { customElement, state } from 'lit/decorators.js';
import './components/app-header';
import './components/story-card-skeleton';
import './components/news-card';
import './components/story-detail';
import './components/comment-thread';
import { getStoryUrl } from './utils/storyUrl';

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
    tcf_score: number;
    timely: number;
    funding: number;
    explanation: string;
}

interface FeedResponse {
    status: string;
    count: number;
    stories: Story[];
}

@customElement('app-root')
class AppRoot extends LitElement {
    @state() stories: Story[] = [];
    @state() loading = true;
    @state() loadingMore = false;
    @state() error: string | null = null;
    @state() hasMore = true;
    @state() newStoriesCount = 0;
    @state() minCoherence = 0.0;
    @state() debouncedCoherence = 0.0;

    private refreshInterval?: number;
    private scrollThreshold = 1000; // Start loading when 1000px from bottom
    private pageSize = 12; // Load 12 stories at a time (classic_app pattern)
    private coherenceDebounceTimer?: number;

    static styles = css`
        :host {
            display: block;
            min-height: 100vh;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 2rem;
        }

        .feed-section {
            background: white;
            border-radius: 12px;
            padding: 2rem;
            margin-top: 2rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }

        .loading {
            text-align: center;
            padding: 4rem;
            font-size: 1.125rem;
            color: #6b7280;
        }

        .error {
            background: #fee;
            color: #c33;
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1rem;
        }

        .feed {
            display: grid;
            gap: 1.5rem;
        }


        .loading-more {
            text-align: center;
            padding: 2rem;
            color: #6b7280;
            font-size: 0.875rem;
        }

        .no-more {
            text-align: center;
            padding: 2rem;
            color: #9ca3af;
            font-size: 0.875rem;
            font-style: italic;
        }

        .new-stories-banner {
            position: fixed;
            top: 80px;
            left: 50%;
            transform: translateX(-50%);
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 0.75rem 1.5rem;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
            cursor: pointer;
            z-index: 100;
            animation: slideDown 0.3s ease-out;
            font-size: 0.875rem;
            font-weight: 500;
        }

        @keyframes slideDown {
            from {
                opacity: 0;
                transform: translateX(-50%) translateY(-10px);
            }
            to {
                opacity: 1;
                transform: translateX(-50%) translateY(0);
            }
        }

        .skeleton-grid {
            display: grid;
            gap: 1.5rem;
        }

        @media (max-width: 768px) {
            .container {
                padding: 1rem;
            }

            .feed-section {
                padding: 1rem;
            }

            .new-stories-banner {
                top: 70px;
                padding: 0.5rem 1rem;
                font-size: 0.8125rem;
            }
        }
    `;

    async connectedCallback() {
        super.connectedCallback();
        this.loadPreferences();
        await this.loadFeed(true);
        this.startBackgroundRefresh();
        window.addEventListener('scroll', this.handleScroll);
    }

    disconnectedCallback() {
        super.disconnectedCallback();
        this.stopBackgroundRefresh();
        window.removeEventListener('scroll', this.handleScroll);
    }

    loadPreferences() {
        // Classic_app pattern: Load filter preferences from localStorage
        const savedCoherence = localStorage.getItem('story_min_coherence');
        if (savedCoherence) {
            const coherence = parseFloat(savedCoherence);
            this.minCoherence = coherence;
            this.debouncedCoherence = coherence;
        }
    }

    savePreferences() {
        // Classic_app pattern: Save filter preferences to localStorage
        localStorage.setItem('story_min_coherence', this.debouncedCoherence.toString());
    }

    handleCoherenceChange(e: Event) {
        const target = e.target as HTMLInputElement;
        this.minCoherence = parseFloat(target.value);

        // Debounce filter updates (classic_app pattern: 500ms)
        if (this.coherenceDebounceTimer) {
            clearTimeout(this.coherenceDebounceTimer);
        }

        this.coherenceDebounceTimer = window.setTimeout(() => {
            this.debouncedCoherence = this.minCoherence;
            this.savePreferences();
            this.loadFeed(true);
        }, 500);
    }

    async loadFeed(initial = false) {
        try {
            if (initial) {
                this.loading = true;
                this.stories = [];
            }

            const params = new URLSearchParams({
                limit: this.pageSize.toString(),
                min_coherence: this.debouncedCoherence.toString()
            });

            const response = await fetch(`/api/coherence/feed?${params}`);
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            const data: FeedResponse = await response.json();
            this.stories = data.stories || [];
            this.hasMore = data.stories.length === this.pageSize;
            this.loading = false;
        } catch (err) {
            this.error = err instanceof Error ? err.message : 'Failed to load feed';
            this.loading = false;
        }
    }

    async loadMore() {
        if (this.loadingMore || !this.hasMore) return;

        this.loadingMore = true;
        try {
            const params = new URLSearchParams({
                limit: this.pageSize.toString(),
                offset: this.stories.length.toString(),
                min_coherence: this.debouncedCoherence.toString()
            });

            const response = await fetch(`/api/coherence/feed?${params}`);
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            const data: FeedResponse = await response.json();
            const newStories = data.stories || [];
            this.stories = [...this.stories, ...newStories];
            this.hasMore = newStories.length === this.pageSize;
        } catch (err) {
            this.error = err instanceof Error ? err.message : 'Failed to load more stories';
        } finally {
            this.loadingMore = false;
        }
    }

    handleScroll = () => {
        const scrollHeight = document.documentElement.scrollHeight;
        const scrollTop = document.documentElement.scrollTop;
        const clientHeight = document.documentElement.clientHeight;
        const distanceFromBottom = scrollHeight - scrollTop - clientHeight;

        // Trigger load more when within threshold
        if (distanceFromBottom < this.scrollThreshold && !this.loadingMore && this.hasMore) {
            this.loadMore();
        }
    };

    startBackgroundRefresh() {
        // Refresh every 30 seconds (classic_app pattern)
        this.refreshInterval = window.setInterval(() => {
            this.checkNewStories();
        }, 30000);
    }

    stopBackgroundRefresh() {
        if (this.refreshInterval) {
            clearInterval(this.refreshInterval);
        }
    }

    async checkNewStories() {
        try {
            const params = new URLSearchParams({
                limit: this.pageSize.toString(),
                min_coherence: this.debouncedCoherence.toString()
            });

            const response = await fetch(`/api/coherence/feed?${params}`);
            if (!response.ok) return;

            const data: FeedResponse = await response.json();
            const newStories = data.stories || [];

            // Check if there are new stories by comparing first story ID
            if (newStories.length > 0 && this.stories.length > 0) {
                const firstNewId = newStories[0].story_id;
                const hasNew = !this.stories.some(s => s.story_id === firstNewId);

                if (hasNew) {
                    // Count how many new stories there are
                    const newCount = newStories.findIndex(s =>
                        this.stories.some(existing => existing.story_id === s.story_id)
                    );
                    this.newStoriesCount = newCount === -1 ? newStories.length : newCount;

                    // Auto-hide banner after 5 seconds
                    setTimeout(() => {
                        this.newStoriesCount = 0;
                    }, 5000);
                }
            }
        } catch (err) {
            // Silent fail for background refresh
            console.error('Background refresh failed:', err);
        }
    }

    async refreshFeed() {
        this.newStoriesCount = 0;
        await this.loadFeed(true);
    }

    render() {
        return html`
            <app-header></app-header>

            ${this.newStoriesCount > 0 ? html`
                <div class="new-stories-banner" @click=${this.refreshFeed}>
                    ${this.newStoriesCount} new ${this.newStoriesCount === 1 ? 'story' : 'stories'} available - Click to refresh
                </div>
            ` : ''}

            <div class="container">
                <div class="feed-section">
                    ${this.error ? html`<div class="error">Error: ${this.error}</div>` : ''}

                    ${this.loading ? html`
                        <div class="skeleton-grid">
                            ${Array(this.pageSize).fill(0).map(() => html`<story-card-skeleton></story-card-skeleton>`)}
                        </div>
                    ` : html`
                        <div class="feed">
                            ${this.stories.length === 0 ? html`
                                <div class="loading">No stories found</div>
                            ` : this.stories.map((story) => this.renderStory(story))}
                        </div>

                        ${this.loadingMore ? html`
                            <div class="loading-more">
                                <div class="skeleton-grid">
                                    ${Array(3).fill(0).map(() => html`<story-card-skeleton></story-card-skeleton>`)}
                                </div>
                            </div>
                        ` : ''}

                        ${!this.hasMore && this.stories.length > 0 ? html`
                            <div class="no-more">No more stories</div>
                        ` : ''}
                    `}
                </div>
            </div>
        `;
    }

    renderStory(story: Story) {
        return html`
            <news-card
                .story=${story}
                @story-click=${(e: CustomEvent) => this.handleStoryClick(e.detail)}
            ></news-card>
        `;
    }

    handleStoryClick(detail: { storyId: string; title: string }) {
        // Generate SEO-friendly URL with slug
        window.location.href = getStoryUrl(detail.storyId, detail.title);
    }
}

// Mount the app with routing
const app = document.getElementById('app');
if (app) {
    // Check if we're on a story detail page
    const path = window.location.pathname;
    if (path.startsWith('/story/')) {
        app.innerHTML = `
            <app-header></app-header>
            <story-detail></story-detail>
        `;
    } else {
        // Default to feed page
        app.innerHTML = '<app-root></app-root>';
    }
}
