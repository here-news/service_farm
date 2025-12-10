import { LitElement, html, css } from 'lit';
import { customElement, state } from 'lit/decorators.js';
// TEMPORARILY DISABLED TO ISOLATE IME ISSUE
// import './search-bar';
import './user-profile';

interface User {
    id: string;
    name: string;
    email: string;
    picture?: string;
    credits: number;
}

@customElement('app-header')
export class AppHeader extends LitElement {
    @state() user: User | null = null;
    @state() loading = true;

    static styles = css`
        :host {
            display: block;
            background: white;
            border-bottom: 1px solid #e5e7eb;
            position: sticky;
            top: 0;
            z-index: 50;
        }

        .header-container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 0.75rem 1.5rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .left-section {
            display: flex;
            align-items: center;
            gap: 1.5rem;
            flex: 1;
        }

        .logo {
            font-size: 1.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            text-decoration: none;
            cursor: pointer;
            flex-shrink: 0;
        }

        .logo:hover {
            opacity: 0.8;
        }

        .center {
            flex: 1;
            max-width: 500px;
        }

        .actions {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            flex-shrink: 0;
        }

        .icon-btn {
            background: none;
            border: none;
            font-size: 1.5rem;
            cursor: pointer;
            padding: 0.375rem;
            border-radius: 8px;
            transition: all 0.2s;
            position: relative;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .icon-btn:hover {
            background: #f3f4f6;
        }

        .notification-badge {
            position: absolute;
            top: 0;
            right: 0;
            background: #ef4444;
            color: white;
            font-size: 0.625rem;
            border-radius: 50%;
            width: 18px;
            height: 18px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 600;
        }

        .login-btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 0.5rem 1.5rem;
            border-radius: 6px;
            font-size: 0.875rem;
            font-weight: 500;
            cursor: pointer;
            transition: opacity 0.2s;
        }

        .login-btn:hover {
            opacity: 0.9;
        }

        @media (max-width: 768px) {
            .header-container {
                padding: 0.75rem 1rem;
            }

            .left-section {
                gap: 1rem;
            }

            .center {
                display: none;
            }

            .actions {
                gap: 0.5rem;
            }
        }
    `;

    async connectedCallback() {
        super.connectedCallback();
        await this.checkAuth();
    }

    async checkAuth() {
        try {
            const response = await fetch('/api/auth/status');
            if (response.ok) {
                const data = await response.json();
                if (data.authenticated) {
                    this.user = data.user;
                }
            }
        } catch (err) {
            console.error('Auth check failed:', err);
        } finally {
            this.loading = false;
        }
    }

    handleLogin() {
        window.location.href = '/api/auth/login';
    }

    handleNotifications() {
        alert('Notifications coming soon!');
    }

    handleCreate() {
        alert('Create story/quest coming soon!');
    }

    render() {
        return html`
            <div class="header-container">
                <div class="left-section">
                    <a href="/app" class="logo">φ HERE</a>
                    <div class="center">
                        <!-- TEMPORARILY DISABLED TO ISOLATE IME ISSUE -->
                        <!-- <search-bar></search-bar> -->
                    </div>
                </div>

                <div class="actions">
                    ${this.loading ? html`
                        <div style="font-size: 0.875rem; color: #6b7280;">Loading...</div>
                    ` : this.user ? html`
                        <button class="icon-btn" @click=${this.handleCreate} title="Create new content">
                            ➕
                        </button>
                        <button class="icon-btn" @click=${this.handleNotifications} title="Notifications">
                            🔔
                            <span class="notification-badge">3</span>
                        </button>
                        <user-profile .user=${this.user}></user-profile>
                    ` : html`
                        <button class="login-btn" @click=${this.handleLogin}>
                            Sign in with Google
                        </button>
                    `}
                </div>
            </div>
        `;
    }
}
