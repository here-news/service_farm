import { LitElement, html, css } from 'lit';
import { customElement, property, state } from 'lit/decorators.js';
import { unsafeHTML } from 'lit/directives/unsafe-html.js';

interface ChatMessage {
    role: 'user' | 'assistant';
    content: string;
}

interface ChatSession {
    id: string;
    message_count: number;
    status: string;
}

@customElement('story-chat-sidebar')
export class StoryChatSidebar extends LitElement {
    @property({ type: String }) storyId = '';
    @property({ type: Boolean }) isOpen = false;

    @state() messages: ChatMessage[] = [];
    @state() isSending = false;
    @state() session: ChatSession | null = null;
    @state() remainingMessages = 100;

    private textareaRef?: HTMLTextAreaElement;

    // CRITICAL FIX: Disable Shadow DOM to fix IME (Chinese/Japanese/Korean) input
    // Shadow DOM blocks composition events needed for IME input
    createRenderRoot() {
        return this;
    }

    static styles = css`
        :host {
            position: fixed;
            top: 0;
            right: 0;
            bottom: 0;
            width: 400px;
            background: white;
            box-shadow: -4px 0 12px rgba(0, 0, 0, 0.1);
            transform: translateX(100%);
            transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            z-index: 200;
            display: flex;
            flex-direction: column;
        }

        :host([isopen]) {
            transform: translateX(0);
        }

        .sidebar-header {
            padding: 1.5rem;
            border-bottom: 1px solid #e5e7eb;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }

        .header-top {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.5rem;
        }

        .header-title {
            font-size: 1.25rem;
            font-weight: 700;
        }

        .close-button {
            background: rgba(255, 255, 255, 0.2);
            border: none;
            color: white;
            width: 32px;
            height: 32px;
            border-radius: 50%;
            cursor: pointer;
            font-size: 1.25rem;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: background 0.2s;
        }

        .close-button:hover {
            background: rgba(255, 255, 255, 0.3);
        }

        .message-counter {
            font-size: 0.875rem;
            opacity: 0.9;
        }

        .messages-container {
            flex: 1;
            overflow-y: auto;
            padding: 1.5rem;
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }

        .message {
            display: flex;
            gap: 0.5rem;
        }

        .message.user {
            justify-content: flex-end;
        }

        .message.assistant {
            justify-content: flex-start;
        }

        .phi-avatar {
            width: 2rem;
            height: 2rem;
            border-radius: 50%;
            background: linear-gradient(to bottom right, #6366f1, #a855f7, #ec4899);
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: 700;
            font-size: 1.125rem;
            flex-shrink: 0;
            box-shadow: 0 2px 8px rgba(99, 102, 241, 0.3);
        }

        .message-bubble {
            max-width: 80%;
            padding: 0.75rem 1rem;
            border-radius: 12px;
            line-height: 1.5;
            word-wrap: break-word;
        }

        .message.user .message-bubble {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }

        .message.assistant .message-bubble {
            background: #f3f4f6;
            color: #374151;
        }

        .message-bubble entity-link {
            color: inherit;
        }

        .thinking-indicator {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .thinking-dots {
            display: flex;
            gap: 0.25rem;
        }

        .thinking-dot {
            width: 0.5rem;
            height: 0.5rem;
            background: #9ca3af;
            border-radius: 50%;
            animation: bounce 1.4s infinite ease-in-out;
        }

        .thinking-dot:nth-child(1) {
            animation-delay: 0ms;
        }

        .thinking-dot:nth-child(2) {
            animation-delay: 150ms;
        }

        .thinking-dot:nth-child(3) {
            animation-delay: 300ms;
        }

        .thinking-text {
            font-size: 0.75rem;
            color: #6b7280;
        }

        @keyframes bounce {
            0%, 80%, 100% {
                transform: translateY(0);
            }
            40% {
                transform: translateY(-0.5rem);
            }
        }

        .input-container {
            padding: 1rem;
            border-top: 1px solid #e5e7eb;
            background: #f9fafb;
        }

        .input-box {
            display: flex;
            gap: 0.5rem;
            align-items: center;
        }

        .message-input {
            flex: 1;
            padding: 0.75rem;
            border: 1px solid #d1d5db;
            border-radius: 8px;
            font-size: 0.875rem;
            outline: none;
            transition: border-color 0.2s;
            resize: none;
            min-height: 40px;
            max-height: 120px;
            font-family: inherit;
            line-height: 1.5;
        }

        .message-input:focus {
            border-color: #667eea;
        }

        .send-button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 0.75rem 1.25rem;
            border-radius: 8px;
            font-size: 0.875rem;
            font-weight: 600;
            cursor: pointer;
            transition: opacity 0.2s;
        }

        .send-button:hover {
            opacity: 0.9;
        }

        .send-button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        .limit-warning {
            background: #fef3c7;
            color: #92400e;
            padding: 0.75rem;
            border-radius: 8px;
            font-size: 0.875rem;
            margin-top: 0.5rem;
        }

        .exhausted-notice {
            background: #fee2e2;
            color: #991b1b;
            padding: 1rem;
            border-radius: 8px;
            font-size: 0.875rem;
            margin: 1rem;
        }

        .empty-state {
            flex: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            color: #9ca3af;
            gap: 1rem;
            padding: 2rem;
        }

        .empty-icon {
            font-size: 3rem;
        }

        .empty-text {
            text-align: center;
            line-height: 1.6;
        }
    `;

    connectedCallback() {
        super.connectedCallback();
        this.loadConversationHistory();
    }

    firstUpdated() {
        // Capture textarea reference for value access (using Light DOM after createRenderRoot fix)
        const textarea = this.querySelector('.message-input') as HTMLTextAreaElement;
        if (textarea) {
            this.textareaRef = textarea;
        }
    }

    updated(changedProperties: Map<string, any>) {
        if (changedProperties.has('storyId') && this.storyId) {
            this.loadConversationHistory();
            this.loadSession();
        }
        // Update textarea ref when sidebar opens (using Light DOM after createRenderRoot fix)
        if (changedProperties.has('isOpen') && this.isOpen) {
            const textarea = this.querySelector('.message-input') as HTMLTextAreaElement;
            if (textarea) {
                this.textareaRef = textarea;
            }
        }
    }

    getTextareaValue(): string {
        return this.textareaRef?.value || '';
    }

    clearTextarea() {
        if (this.textareaRef) {
            this.textareaRef.value = '';
        }
    }

    loadConversationHistory() {
        try {
            const stored = localStorage.getItem(`chat_${this.storyId}`);
            if (stored) {
                this.messages = JSON.parse(stored);
            }
        } catch (error) {
            console.error('Failed to load conversation history:', error);
        }
    }

    saveConversationHistory() {
        try {
            localStorage.setItem(`chat_${this.storyId}`, JSON.stringify(this.messages));
        } catch (error) {
            console.error('Failed to save conversation history:', error);
        }
    }

    async loadSession() {
        try {
            const response = await fetch(`/api/chat/session/${this.storyId}`, {
                credentials: 'include'
            });

            if (response.ok) {
                this.session = await response.json();
                if (this.session) {
                    this.remainingMessages = Math.max(0, 100 - this.session.message_count);
                }
            }
        } catch (error) {
            console.error('Failed to load session:', error);
        }
    }

    handleClose() {
        this.dispatchEvent(new CustomEvent('close-chat', { bubbles: true, composed: true }));
    }

    async handleSendMessage() {
        const userMessage = this.getTextareaValue().trim();
        if (!userMessage || this.isSending) return;

        this.clearTextarea();

        // Add user message to UI
        this.messages = [
            ...this.messages,
            { role: 'user', content: userMessage }
        ];

        this.isSending = true;

        try {
            const response = await fetch('/api/chat/message', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                credentials: 'include',
                body: JSON.stringify({
                    story_id: this.storyId,
                    message: userMessage,
                    conversation_history: this.messages.slice(-10) // Last 10 messages for context
                })
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Failed to send message');
            }

            const data = await response.json();

            // Add AI response
            this.messages = [
                ...this.messages,
                { role: 'assistant', content: data.message }
            ];

            this.remainingMessages = data.remaining_messages;

            if (data.session_status === 'exhausted') {
                this.session = { ...this.session!, status: 'exhausted' };
            }

            this.saveConversationHistory();

            // Scroll to bottom
            this.scrollToBottom();

        } catch (error) {
            console.error('Error sending message:', error);
            alert(error instanceof Error ? error.message : 'Failed to send message');
        } finally {
            this.isSending = false;
        }
    }

    scrollToBottom() {
        setTimeout(() => {
            // Using Light DOM after createRenderRoot fix
            const container = this.querySelector('.messages-container');
            if (container) {
                container.scrollTop = container.scrollHeight;
            }
        }, 100);
    }

    parseEntityMarkup(text: string): string {
        // Parse [[Entity Name]] markup
        return text.replace(/\[\[([^\]|]+)(?:\|([^\]]+))?\]\]/g, (_, name, id) => {
            return `<entity-link entityid="${id || name}" entityname="${name}" entitytype="unknown"></entity-link>`;
        });
    }

    render() {
        const isExhausted = this.session?.status === 'exhausted';
        const showWarning = this.remainingMessages <= 10 && this.remainingMessages > 0;

        return html`
            <div class="sidebar-header">
                <div class="header-top">
                    <div class="header-title">Story Chat</div>
                    <button class="close-button" @click=${this.handleClose}>×</button>
                </div>
                <div class="message-counter">
                    ${this.remainingMessages} / 100 messages remaining
                </div>
            </div>

            <div class="messages-container">
                ${this.messages.length === 0 ? html`
                    <div class="empty-state">
                        <div class="empty-icon">💬</div>
                        <div class="empty-text">
                            Start a conversation about this story. Ask questions, explore connections,
                            or get AI-powered insights.
                        </div>
                    </div>
                ` : ''}

                ${this.messages.map(msg => html`
                    <div class="message ${msg.role}">
                        ${msg.role === 'assistant' ? html`
                            <div class="phi-avatar">φ</div>
                        ` : ''}
                        <div class="message-bubble">
                            ${msg.role === 'assistant'
                                ? unsafeHTML(this.parseEntityMarkup(msg.content))
                                : msg.content
                            }
                        </div>
                    </div>
                `)}

                ${this.isSending ? html`
                    <div class="message assistant">
                        <div class="phi-avatar">φ</div>
                        <div class="message-bubble">
                            <div class="thinking-indicator">
                                <div class="thinking-dots">
                                    <div class="thinking-dot"></div>
                                    <div class="thinking-dot"></div>
                                    <div class="thinking-dot"></div>
                                </div>
                                <span class="thinking-text">Thinking...</span>
                            </div>
                        </div>
                    </div>
                ` : ''}
            </div>

            ${isExhausted ? html`
                <div class="exhausted-notice">
                    You've reached the 100-message limit for this story. Start a new conversation
                    by visiting another story.
                </div>
            ` : html`
                <div class="input-container">
                    ${showWarning ? html`
                        <div class="limit-warning">
                            Only ${this.remainingMessages} messages remaining
                        </div>
                    ` : ''}
                    <div class="input-box">
                        <textarea
                            class="message-input"
                            placeholder="Ask a question..."
                            rows="1"
                        ></textarea>
                        <button
                            class="send-button"
                            type="button"
                            ?disabled=${this.isSending}
                            @click=${this.handleSendMessage}
                        >
                            ${this.isSending ? 'Sending...' : 'Send'}
                        </button>
                    </div>
                </div>
            `}
        `;
    }
}
