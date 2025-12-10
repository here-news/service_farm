import { LitElement, html, css } from 'lit';
import { customElement, state, query } from 'lit/decorators.js';

interface SearchResult {
    id: string;
    title: string;
    description?: string;
    category?: string;
    artifact_count?: number;
}

@customElement('search-bar')
export class SearchBar extends LitElement {
    @state() query = '';
    @state() results: SearchResult[] = [];
    @state() isSearching = false;
    @state() showResults = false;

    @query('input') input!: HTMLInputElement;

    private searchTimeout: number | null = null;

    static styles = css`
        :host {
            display: block;
            position: relative;
        }

        .search-container {
            position: relative;
        }

        .search-input-wrapper {
            position: relative;
            display: flex;
            align-items: center;
        }

        .search-icon {
            position: absolute;
            left: 12px;
            width: 20px;
            height: 20px;
            color: #9ca3af;
            pointer-events: none;
        }

        input {
            width: 100%;
            padding: 0.625rem 3rem 0.625rem 2.5rem;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            font-size: 0.875rem;
            outline: none;
            transition: all 0.2s;
        }

        input:focus {
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }

        .shortcut {
            position: absolute;
            right: 12px;
            padding: 0.25rem 0.5rem;
            background: #f3f4f6;
            border: 1px solid #e5e7eb;
            border-radius: 4px;
            font-size: 0.75rem;
            color: #6b7280;
            font-weight: 600;
            pointer-events: none;
        }

        .spinner {
            position: absolute;
            right: 12px;
            width: 16px;
            height: 16px;
            border: 2px solid #667eea;
            border-top-color: transparent;
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        .results-dropdown {
            position: absolute;
            top: calc(100% + 0.5rem);
            left: 0;
            right: 0;
            background: white;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
            max-height: 400px;
            overflow-y: auto;
            z-index: 50;
        }

        .result-item {
            padding: 0.75rem 1rem;
            border-bottom: 1px solid #f3f4f6;
            cursor: pointer;
            transition: background 0.15s;
        }

        .result-item:last-child {
            border-bottom: none;
        }

        .result-item:hover {
            background: #f9fafb;
        }

        .result-title {
            font-size: 0.875rem;
            font-weight: 500;
            color: #111827;
            margin-bottom: 0.25rem;
        }

        .result-description {
            font-size: 0.75rem;
            color: #6b7280;
            line-height: 1.4;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
            overflow: hidden;
        }

        .result-meta {
            display: flex;
            gap: 0.5rem;
            margin-top: 0.25rem;
            font-size: 0.75rem;
        }

        .result-category {
            background: #f3f4f6;
            padding: 0.125rem 0.5rem;
            border-radius: 4px;
            color: #6b7280;
        }

        .no-results {
            padding: 2rem 1rem;
            text-align: center;
            color: #6b7280;
            font-size: 0.875rem;
        }
    `;

    connectedCallback() {
        super.connectedCallback();
        window.addEventListener('keydown', this.handleGlobalKeydown);
        window.addEventListener('click', this.handleClickOutside);
    }

    disconnectedCallback() {
        super.disconnectedCallback();
        window.removeEventListener('keydown', this.handleGlobalKeydown);
        window.removeEventListener('click', this.handleClickOutside);
    }

    handleGlobalKeydown = (e: KeyboardEvent) => {
        // Ignore keyboard shortcuts during IME composition
        if (e.isComposing) {
            return;
        }
        // Focus search on "/" key
        if (e.key === '/' && document.activeElement !== this.input) {
            e.preventDefault();
            this.input?.focus();
        }
        // Close on Escape
        if (e.key === 'Escape') {
            this.showResults = false;
            this.input?.blur();
        }
    };

    handleClickOutside = (e: MouseEvent) => {
        if (!this.contains(e.target as Node)) {
            this.showResults = false;
        }
    };

    handleInput(e: Event) {
        const target = e.target as HTMLInputElement;
        this.query = target.value;

        if (this.searchTimeout) {
            clearTimeout(this.searchTimeout);
        }

        if (this.query.trim().length < 2) {
            this.results = [];
            this.showResults = false;
            return;
        }

        this.isSearching = true;
        // Classic_app pattern: 500ms debounce for user-driven filters
        this.searchTimeout = window.setTimeout(() => this.performSearch(), 500);
    }

    async performSearch() {
        try {
            const response = await fetch('/api/coherence/search', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: this.query.trim(), limit: 5 })
            });

            if (response.ok) {
                const data = await response.json();
                this.results = data.matches || [];
                this.showResults = true;
            }
        } catch (err) {
            console.error('Search error:', err);
            this.results = [];
        } finally {
            this.isSearching = false;
        }
    }

    handleResultClick(storyId: string) {
        // Navigate to story detail page
        window.location.href = `/story/${storyId}`;
    }

    render() {
        return html`
            <div class="search-container">
                <div class="search-input-wrapper">
                    <svg class="search-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                    </svg>
                    <input
                        type="text"
                        @input=${this.handleInput}
                        placeholder="Search stories..."
                        @focus=${() => { if (this.results.length > 0) this.showResults = true; }}
                    />
                    ${this.isSearching ? html`
                        <div class="spinner"></div>
                    ` : html`
                        <kbd class="shortcut">/</kbd>
                    `}
                </div>

                ${this.showResults && this.query.trim().length >= 2 ? html`
                    <div class="results-dropdown">
                        ${this.results.length > 0 ? this.results.map(result => html`
                            <div class="result-item" @click=${() => this.handleResultClick(result.id)}>
                                <div class="result-title">${result.title}</div>
                                ${result.description ? html`
                                    <div class="result-description">${result.description}</div>
                                ` : ''}
                                <div class="result-meta">
                                    ${result.category ? html`
                                        <span class="result-category">${result.category}</span>
                                    ` : ''}
                                    ${result.artifact_count !== undefined ? html`
                                        <span>${result.artifact_count} sources</span>
                                    ` : ''}
                                </div>
                            </div>
                        `) : html`
                            <div class="no-results">
                                No stories found for "${this.query}"
                            </div>
                        `}
                    </div>
                ` : ''}
            </div>
        `;
    }
}
