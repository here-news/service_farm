import { LitElement, html, css, TemplateResult } from 'lit';
import { customElement, property, state } from 'lit/decorators.js';

interface Comment {
    id: string;
    story_id: string;
    user_id: string;
    user_name: string;
    user_picture?: string;
    user_email: string;
    text: string;
    parent_comment_id?: string;
    reaction_type?: 'support' | 'refute' | 'question' | 'comment';
    created_at: string;
    updated_at?: string;
}

@customElement('comment-thread')
export class CommentThread extends LitElement {
    @property({ type: String }) storyId!: string;
    @state() comments: Comment[] = [];
    @state() loading = false;
    @state() error = '';
    @state() replyTo: string | null = null;
    @state() newCommentReaction: string | null = null;
    @state() isAuthenticated = false;
    @state() currentUser: any = null;

    private commentTextareaRef?: HTMLTextAreaElement;

    // CRITICAL FIX: Disable Shadow DOM to fix IME (Chinese/Japanese/Korean) input
    // Shadow DOM blocks composition events needed for IME input
    createRenderRoot() {
        return this;
    }

    static styles = css`
        :host {
            display: block;
            margin-top: 3rem;
        }

        .comments-section {
            max-width: 800px;
            margin: 0 auto;
        }

        .section-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1.5rem;
            padding-bottom: 1rem;
            border-bottom: 2px solid #e5e7eb;
        }

        .section-title {
            font-size: 1.5rem;
            font-weight: 700;
            color: #111827;
        }

        .comment-count {
            font-size: 0.875rem;
            color: #6b7280;
        }

        .comment-form {
            background: #f9fafb;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 2rem;
        }

        .comment-form.reply {
            margin-left: 3rem;
            margin-bottom: 1rem;
            background: #eff6ff;
            border-color: #3b82f6;
        }

        textarea {
            width: 100%;
            min-height: 80px;
            padding: 0.75rem;
            border: 1px solid #d1d5db;
            border-radius: 6px;
            font-family: inherit;
            font-size: 0.875rem;
            resize: vertical;
            outline: none;
        }

        textarea:focus {
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }

        .form-actions {
            display: flex;
            gap: 0.5rem;
            margin-top: 0.75rem;
        }

        .reaction-buttons {
            display: flex;
            gap: 0.5rem;
            flex: 1;
        }

        .reaction-btn {
            padding: 0.375rem 0.75rem;
            border: 1px solid #d1d5db;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 500;
            cursor: pointer;
            background: white;
            transition: all 0.2s;
        }

        .reaction-btn:hover {
            background: #f3f4f6;
        }

        .reaction-btn.active {
            border-color: #667eea;
            background: #667eea;
            color: white;
        }

        .reaction-btn.support.active { background: #10b981; border-color: #10b981; }
        .reaction-btn.refute.active { background: #ef4444; border-color: #ef4444; }
        .reaction-btn.question.active { background: #f59e0b; border-color: #f59e0b; }

        .submit-btn {
            padding: 0.375rem 1rem;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 6px;
            font-size: 0.875rem;
            font-weight: 500;
            cursor: pointer;
        }

        .submit-btn:hover {
            opacity: 0.9;
        }

        .submit-btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        .cancel-btn {
            padding: 0.375rem 1rem;
            background: white;
            color: #6b7280;
            border: 1px solid #d1d5db;
            border-radius: 6px;
            font-size: 0.875rem;
            cursor: pointer;
        }

        .comment-tree {
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }

        .comment {
            background: white;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 1rem;
        }

        .comment.reply {
            margin-left: 3rem;
            border-left: 3px solid #667eea;
        }

        .comment-header {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin-bottom: 0.75rem;
        }

        .user-avatar {
            width: 36px;
            height: 36px;
            border-radius: 50%;
            object-fit: cover;
        }

        .user-avatar.placeholder {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 600;
            font-size: 0.875rem;
        }

        .user-info {
            flex: 1;
        }

        .user-name {
            font-weight: 600;
            color: #111827;
            font-size: 0.875rem;
        }

        .comment-time {
            font-size: 0.75rem;
            color: #9ca3af;
        }

        .reaction-badge {
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 500;
        }

        .reaction-badge.support { background: #d1fae5; color: #065f46; }
        .reaction-badge.refute { background: #fee2e2; color: #991b1b; }
        .reaction-badge.question { background: #fef3c7; color: #92400e; }
        .reaction-badge.comment { background: #e0e7ff; color: #3730a3; }

        .comment-text {
            color: #374151;
            line-height: 1.6;
            font-size: 0.875rem;
            margin-bottom: 0.75rem;
            white-space: pre-wrap;
        }

        .comment-actions {
            display: flex;
            gap: 1rem;
            padding-top: 0.5rem;
            border-top: 1px solid #f3f4f6;
        }

        .action-btn {
            font-size: 0.75rem;
            color: #6b7280;
            cursor: pointer;
            border: none;
            background: none;
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
        }

        .action-btn:hover {
            color: #667eea;
            background: #f3f4f6;
        }

        .loading {
            text-align: center;
            padding: 2rem;
            color: #6b7280;
        }

        .error {
            background: #fee;
            color: #c33;
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1rem;
        }

        .empty-state {
            text-align: center;
            padding: 3rem 1rem;
            color: #9ca3af;
        }

        .empty-state-icon {
            font-size: 3rem;
            margin-bottom: 1rem;
        }

        .auth-required {
            background: #eff6ff;
            border: 1px solid #3b82f6;
            border-radius: 8px;
            padding: 1rem;
            text-align: center;
            margin-bottom: 2rem;
        }

        .auth-btn {
            margin-top: 0.75rem;
            padding: 0.5rem 1rem;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 6px;
            font-size: 0.875rem;
            cursor: pointer;
        }
    `;

    async connectedCallback() {
        super.connectedCallback();
        await this.checkAuthStatus();
        await this.loadComments();
    }

    firstUpdated() {
        this.attachTextareaRef();
    }

    updated(changedProperties: Map<string, any>) {
        // Reattach ref when reply mode changes
        if (changedProperties.has('replyTo')) {
            this.attachTextareaRef();
        }
    }

    private attachTextareaRef() {
        const textarea = this.shadowRoot?.querySelector('.comment-textarea') as HTMLTextAreaElement;
        if (textarea) {
            this.commentTextareaRef = textarea;
        }
    }

    async checkAuthStatus() {
        try {
            const response = await fetch('/api/auth/status', {
                credentials: 'include'
            });
            if (response.ok) {
                const data = await response.json();
                this.isAuthenticated = data.authenticated;
                this.currentUser = data.user;
            }
        } catch (err) {
            console.error('Failed to check auth status:', err);
        }
    }


    async loadComments() {
        this.loading = true;
        this.error = '';

        try {
            const response = await fetch(`/api/comments/story/${this.storyId}`);
            if (!response.ok) throw new Error('Failed to load comments');

            this.comments = await response.json();
        } catch (err) {
            this.error = err instanceof Error ? err.message : 'Failed to load comments';
        } finally {
            this.loading = false;
        }
    }

    getCommentTextareaValue(): string {
        return this.commentTextareaRef?.value || '';
    }

    clearCommentTextarea() {
        if (this.commentTextareaRef) {
            this.commentTextareaRef.value = '';
        }
    }

    async handleButtonClick() {
        const commentText = this.getCommentTextareaValue().trim();
        if (!commentText) return;

        try {
            const response = await fetch('/api/comments/', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                credentials: 'include',
                body: JSON.stringify({
                    story_id: this.storyId,
                    text: commentText,
                    parent_comment_id: this.replyTo,
                    reaction_type: this.newCommentReaction
                })
            });

            if (response.status === 401) {
                window.location.href = '/api/auth/login';
                return;
            }

            if (!response.ok) {
                const data = await response.json();
                throw new Error(data.detail || 'Failed to post comment');
            }

            // Success - clear form
            this.clearCommentTextarea();
            this.newCommentReaction = null;
            this.replyTo = null;

            // Reload comments
            await this.loadComments();

        } catch (err) {
            this.error = err instanceof Error ? err.message : 'Failed to post comment';
        }
    }

    handleLogin() {
        window.location.href = '/api/auth/login';
    }

    handleReply(commentId: string) {
        if (!this.isAuthenticated) {
            this.handleLogin();
            return;
        }
        this.replyTo = commentId;
        this.clearCommentTextarea();
        this.newCommentReaction = null;
    }

    handleCancelReply() {
        this.replyTo = null;
        this.clearCommentTextarea();
        this.newCommentReaction = null;
    }

    setReaction(type: string) {
        this.newCommentReaction = this.newCommentReaction === type ? null : type;
    }

    getTimeAgo(dateString: string): string {
        const date = new Date(dateString);
        const now = new Date();
        const diffMs = now.getTime() - date.getTime();
        const diffMins = Math.floor(diffMs / 60000);
        const diffHours = Math.floor(diffMins / 60);
        const diffDays = Math.floor(diffHours / 24);

        if (diffMins < 1) return 'Just now';
        if (diffMins < 60) return `${diffMins}m ago`;
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

    render() {
        const topLevelComments = this.comments.filter(c => !c.parent_comment_id);

        return html`
            <div class="comments-section">
                <div class="section-header">
                    <h2 class="section-title">Discussion</h2>
                    <span class="comment-count">${this.comments.length} comments</span>
                </div>

                ${this.error ? html`<div class="error">${this.error}</div>` : ''}

                ${this.isAuthenticated && !this.replyTo ? this.renderCommentForm() : !this.isAuthenticated ? html`
                    <div class="auth-required">
                        <p>Sign in to join the discussion</p>
                        <button class="auth-btn" @click=${this.handleLogin}>
                            Sign in with Google
                        </button>
                    </div>
                ` : ''}

                ${this.loading ? html`
                    <div class="loading">Loading comments...</div>
                ` : topLevelComments.length === 0 ? html`
                    <div class="empty-state">
                        <div class="empty-state-icon">💬</div>
                        <p>No comments yet. Be the first to share your thoughts!</p>
                    </div>
                ` : html`
                    <div class="comment-tree">
                        ${topLevelComments.map(comment => this.renderComment(comment))}
                    </div>
                `}
            </div>
        `;
    }

    renderCommentForm(isReply = false): TemplateResult {
        return html`
            <div class="comment-form ${isReply ? 'reply' : ''}">
                <textarea
                    placeholder="${isReply ? 'Write your reply...' : 'Share your thoughts...'}"
                    class="comment-textarea"
                ></textarea>

                <div class="form-actions">
                    <div class="reaction-buttons">
                        <button type="button" class="reaction-btn support ${this.newCommentReaction === 'support' ? 'active' : ''}"
                                @click=${() => this.setReaction('support')}>
                            👍 Support
                        </button>
                        <button type="button" class="reaction-btn refute ${this.newCommentReaction === 'refute' ? 'active' : ''}"
                                @click=${() => this.setReaction('refute')}>
                            ⚠️ Refute
                        </button>
                        <button type="button" class="reaction-btn question ${this.newCommentReaction === 'question' ? 'active' : ''}"
                                @click=${() => this.setReaction('question')}>
                            ❓ Question
                        </button>
                    </div>

                    ${isReply ? html`
                        <button type="button" class="cancel-btn" @click=${this.handleCancelReply}>Cancel</button>
                    ` : ''}

                    <button type="button" class="submit-btn" @click=${this.handleButtonClick}>
                        ${isReply ? 'Reply' : 'Comment'}
                    </button>
                </div>
            </div>
        `;
    }

    renderComment(comment: Comment): TemplateResult {
        const replies = this.comments.filter(c => c.parent_comment_id === comment.id);
        const isReply = !!comment.parent_comment_id;

        return html`
            <div class="comment ${isReply ? 'reply' : ''}">
                <div class="comment-header">
                    ${comment.user_picture ? html`
                        <img src="${comment.user_picture}" alt="${comment.user_name}" class="user-avatar" />
                    ` : html`
                        <div class="user-avatar placeholder">${this.getInitials(comment.user_name)}</div>
                    `}

                    <div class="user-info">
                        <div class="user-name">${comment.user_name}</div>
                        <div class="comment-time">${this.getTimeAgo(comment.created_at)}</div>
                    </div>

                    ${comment.reaction_type ? html`
                        <span class="reaction-badge ${comment.reaction_type}">
                            ${comment.reaction_type === 'support' ? '👍 Support' : ''}
                            ${comment.reaction_type === 'refute' ? '⚠️ Refute' : ''}
                            ${comment.reaction_type === 'question' ? '❓ Question' : ''}
                            ${comment.reaction_type === 'comment' ? '💬 Comment' : ''}
                        </span>
                    ` : ''}
                </div>

                <div class="comment-text">${comment.text}</div>

                <div class="comment-actions">
                    <button class="action-btn" @click=${() => this.handleReply(comment.id)}>
                        Reply
                    </button>
                </div>

                ${this.replyTo === comment.id ? this.renderCommentForm(true) : ''}

                ${replies.map(reply => this.renderComment(reply))}
            </div>
        `;
    }
}
