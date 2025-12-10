import { LitElement, html, css } from 'lit';
import { customElement } from 'lit/decorators.js';

@customElement('story-card-skeleton')
export class StoryCardSkeleton extends LitElement {
    static styles = css`
        :host {
            display: block;
        }

        .skeleton-card {
            background: white;
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            padding: 1.5rem;
            animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
        }

        @keyframes pulse {
            0%, 100% {
                opacity: 1;
            }
            50% {
                opacity: 0.5;
            }
        }

        .skeleton-title {
            height: 1.5rem;
            background: #e5e7eb;
            border-radius: 6px;
            margin-bottom: 1rem;
            width: 75%;
        }

        .skeleton-scores {
            display: flex;
            gap: 0.75rem;
            margin-bottom: 1rem;
            flex-wrap: wrap;
        }

        .skeleton-score {
            height: 2rem;
            width: 5rem;
            background: #f3f4f6;
            border-radius: 6px;
        }

        .skeleton-score:first-child {
            background: linear-gradient(135deg, #e5e7eb 0%, #d1d5db 100%);
        }

        .skeleton-explanation {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }

        .skeleton-line {
            height: 0.875rem;
            background: #f3f4f6;
            border-radius: 4px;
        }

        .skeleton-line:nth-child(1) {
            width: 100%;
        }

        .skeleton-line:nth-child(2) {
            width: 90%;
        }

        .skeleton-line:nth-child(3) {
            width: 60%;
        }
    `;

    render() {
        return html`
            <div class="skeleton-card">
                <div class="skeleton-title"></div>
                <div class="skeleton-scores">
                    <div class="skeleton-score"></div>
                    <div class="skeleton-score"></div>
                    <div class="skeleton-score"></div>
                    <div class="skeleton-score"></div>
                </div>
                <div class="skeleton-explanation">
                    <div class="skeleton-line"></div>
                    <div class="skeleton-line"></div>
                    <div class="skeleton-line"></div>
                </div>
            </div>
        `;
    }
}
