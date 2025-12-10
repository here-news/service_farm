import { LitElement, html, css } from 'lit';
import { customElement, property, state } from 'lit/decorators.js';

@customElement('story-chat-button')
export class StoryChatButton extends LitElement {
    @property({ type: String }) storyId = '';
    @property({ type: Number }) userCredits = 0;
    @property({ type: Boolean }) isUnlocked = false;

    @state() showUnlockModal = false;
    @state() isUnlocking = false;

    static styles = css`
        :host {
            position: fixed;
            bottom: 2rem;
            right: 2rem;
            z-index: 100;
        }

        .chat-button {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1rem 1.5rem;
            border-radius: 50px;
            border: none;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
            transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
        }

        .chat-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 16px rgba(102, 126, 234, 0.6);
        }

        .chat-button.unlocked {
            background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
        }

        .button-icon {
            font-size: 1.25rem;
        }

        .credit-badge {
            background: rgba(255, 255, 255, 0.2);
            padding: 0.25rem 0.5rem;
            border-radius: 12px;
            font-size: 0.875rem;
        }

        .modal-overlay {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.5);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 1000;
        }

        .modal {
            background: white;
            border-radius: 16px;
            padding: 2rem;
            max-width: 400px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
        }

        .modal-header {
            font-size: 1.5rem;
            font-weight: 700;
            color: #111827;
            margin-bottom: 1rem;
        }

        .modal-body {
            color: #6b7280;
            margin-bottom: 1.5rem;
            line-height: 1.6;
        }

        .cost-display {
            background: #f9fafb;
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1.5rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .cost-label {
            font-weight: 600;
            color: #374151;
        }

        .cost-value {
            font-size: 1.25rem;
            font-weight: 700;
            color: #667eea;
        }

        .balance-display {
            font-size: 0.875rem;
            color: #6b7280;
            margin-top: 0.5rem;
        }

        .modal-actions {
            display: flex;
            gap: 1rem;
        }

        .modal-button {
            flex: 1;
            padding: 0.75rem;
            border-radius: 8px;
            border: none;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
        }

        .modal-button.primary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }

        .modal-button.primary:hover {
            opacity: 0.9;
        }

        .modal-button.primary:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        .modal-button.secondary {
            background: #f3f4f6;
            color: #374151;
        }

        .modal-button.secondary:hover {
            background: #e5e7eb;
        }

        .error-message {
            background: #fee2e2;
            color: #991b1b;
            padding: 0.75rem;
            border-radius: 8px;
            margin-bottom: 1rem;
            font-size: 0.875rem;
        }
    `;

    handleButtonClick() {
        if (this.isUnlocked) {
            this.dispatchEvent(new CustomEvent('open-chat', { bubbles: true, composed: true }));
        } else {
            this.showUnlockModal = true;
        }
    }

    async handleUnlock() {
        this.isUnlocking = true;

        try {
            const response = await fetch('/api/chat/unlock', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                credentials: 'include',
                body: JSON.stringify({ story_id: this.storyId })
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Failed to unlock chat');
            }

            this.isUnlocked = true;
            this.showUnlockModal = false;
            this.dispatchEvent(new CustomEvent('chat-unlocked', { bubbles: true, composed: true }));
            this.dispatchEvent(new CustomEvent('open-chat', { bubbles: true, composed: true }));

        } catch (error) {
            console.error('Error unlocking chat:', error);
            alert(error instanceof Error ? error.message : 'Failed to unlock chat');
        } finally {
            this.isUnlocking = false;
        }
    }

    handleCloseModal() {
        this.showUnlockModal = false;
    }

    render() {
        const UNLOCK_COST = 10;
        const hasEnoughCredits = this.userCredits >= UNLOCK_COST;

        return html`
            <button class="chat-button ${this.isUnlocked ? 'unlocked' : ''}" @click=${this.handleButtonClick}>
                <span class="button-icon">${this.isUnlocked ? '💬' : '🔓'}</span>
                <span>${this.isUnlocked ? 'Chat' : 'Start Chat'}</span>
                ${!this.isUnlocked ? html`
                    <span class="credit-badge">10 credits</span>
                ` : ''}
            </button>

            ${this.showUnlockModal ? html`
                <div class="modal-overlay" @click=${this.handleCloseModal}>
                    <div class="modal" @click=${(e: Event) => e.stopPropagation()}>
                        <div class="modal-header">Unlock Premium Chat</div>
                        <div class="modal-body">
                            Get AI-powered insights about this story. Ask questions, explore connections,
                            and dive deeper into the facts.
                        </div>

                        ${!hasEnoughCredits ? html`
                            <div class="error-message">
                                Insufficient credits. You have ${this.userCredits} credits, but need ${UNLOCK_COST}.
                            </div>
                        ` : ''}

                        <div class="cost-display">
                            <div class="cost-label">Cost to unlock:</div>
                            <div class="cost-value">${UNLOCK_COST} credits</div>
                        </div>

                        <div class="balance-display">
                            Your balance: ${this.userCredits} credits
                            ${hasEnoughCredits ? html`
                                → ${this.userCredits - UNLOCK_COST} after unlock
                            ` : ''}
                        </div>

                        <div class="modal-actions">
                            <button class="modal-button secondary" @click=${this.handleCloseModal}>
                                Cancel
                            </button>
                            <button
                                class="modal-button primary"
                                @click=${this.handleUnlock}
                                ?disabled=${!hasEnoughCredits || this.isUnlocking}
                            >
                                ${this.isUnlocking ? 'Unlocking...' : 'Unlock Chat'}
                            </button>
                        </div>
                    </div>
                </div>
            ` : ''}
        `;
    }
}
