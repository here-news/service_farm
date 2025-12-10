import { LitElement, html, css } from 'lit';
import { customElement, property } from 'lit/decorators.js';

@customElement('entity-tag')
export class EntityTag extends LitElement {
    @property({ type: String }) name = '';
    @property({ type: String }) type: 'person' | 'organization' | 'location' | 'event' = 'person';

    static styles = css`
        :host {
            display: inline-block;
        }

        .tag {
            display: inline-flex;
            align-items: center;
            gap: 0.25rem;
            padding: 0.375rem 0.75rem;
            border-radius: 6px;
            font-size: 0.8125rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s;
            border: 1px solid transparent;
        }

        .tag:hover {
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            transform: translateY(-1px);
        }

        .tag-person {
            background: linear-gradient(120deg, #dbeafe 0%, #bfdbfe 100%);
            color: #1e40af;
        }

        .tag-person:hover {
            border-color: #3b82f6;
        }

        .tag-organization {
            background: linear-gradient(120deg, #f3e8ff 0%, #e9d5ff 100%);
            color: #6b21a8;
        }

        .tag-organization:hover {
            border-color: #9333ea;
        }

        .tag-location {
            background: linear-gradient(120deg, #d1fae5 0%, #a7f3d0 100%);
            color: #065f46;
        }

        .tag-location:hover {
            border-color: #10b981;
        }

        .tag-event {
            background: linear-gradient(120deg, #fef3c7 0%, #fde68a 100%);
            color: #92400e;
        }

        .tag-event:hover {
            border-color: #f59e0b;
        }

        .icon {
            font-size: 0.875rem;
        }
    `;

    getIcon() {
        switch (this.type) {
            case 'person':
                return '👤';
            case 'organization':
                return '🏢';
            case 'location':
                return '📍';
            case 'event':
                return '📅';
            default:
                return '🏷️';
        }
    }

    render() {
        return html`
            <div class="tag tag-${this.type}">
                <span class="icon">${this.getIcon()}</span>
                <span>${this.name}</span>
            </div>
        `;
    }
}
