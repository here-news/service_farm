import { LitElement, html, css } from 'lit';
import { customElement, property, state } from 'lit/decorators.js';

interface User {
    id: string;
    name: string;
    email: string;
    picture?: string;
    credits: number;
}

@customElement('user-profile')
export class UserProfile extends LitElement {
    @property({ type: Object }) user!: User;
    @state() showDropdown = false;

    static styles = css`
        :host {
            display: block;
            position: relative;
        }

        .profile-button {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem 0.75rem;
            border: none;
            background: #f3f4f6;
            cursor: pointer;
            border-radius: 8px;
            transition: background 0.2s;
        }

        .profile-button:hover {
            background: #e5e7eb;
        }

        .user-info {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .username {
            font-size: 0.875rem;
            font-weight: 500;
            color: #374151;
        }

        .credits {
            font-size: 0.75rem;
            color: #6b7280;
            font-weight: 400;
        }

        .avatar {
            width: 32px;
            height: 32px;
            border-radius: 50%;
            object-fit: cover;
            border: 2px solid #e5e7eb;
        }

        .avatar-placeholder {
            width: 32px;
            height: 32px;
            border-radius: 50%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: 600;
            font-size: 0.875rem;
        }

        @media (max-width: 640px) {
            .credits {
                display: none;
            }
        }

        .dropdown {
            position: absolute;
            top: calc(100% + 0.5rem);
            right: 0;
            background: white;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
            min-width: 220px;
            z-index: 50;
        }

        .dropdown-header {
            padding: 1rem;
            border-bottom: 1px solid #e5e7eb;
        }

        .dropdown-name {
            font-weight: 600;
            color: #111827;
            margin-bottom: 0.25rem;
        }

        .dropdown-email {
            font-size: 0.75rem;
            color: #6b7280;
        }

        .dropdown-credits {
            font-size: 0.75rem;
            color: #667eea;
            margin-top: 0.5rem;
            font-weight: 500;
        }

        .dropdown-menu {
            padding: 0.5rem 0;
        }

        .menu-item {
            display: block;
            width: 100%;
            padding: 0.625rem 1rem;
            text-align: left;
            border: none;
            background: none;
            color: #374151;
            font-size: 0.875rem;
            cursor: pointer;
            transition: background 0.15s;
        }

        .menu-item:hover {
            background: #f9fafb;
        }

        .menu-item.danger {
            color: #dc2626;
        }
    `;

    connectedCallback() {
        super.connectedCallback();
        window.addEventListener('click', this.handleClickOutside);
    }

    disconnectedCallback() {
        super.disconnectedCallback();
        window.removeEventListener('click', this.handleClickOutside);
    }

    handleClickOutside = (e: MouseEvent) => {
        if (!this.contains(e.target as Node)) {
            this.showDropdown = false;
        }
    };

    toggleDropdown(e: Event) {
        e.stopPropagation();
        this.showDropdown = !this.showDropdown;
    }

    handleLogout() {
        window.location.href = '/api/auth/logout';
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
        const displayName = this.user.name.split(' ')[0] || this.user.name.split('@')[0];

        return html`
            <button class="profile-button" @click=${this.toggleDropdown}>
                <div class="user-info">
                    <span class="username">${displayName}</span>
                    <span class="credits">(${this.user.credits}C)</span>
                </div>
                ${this.user.picture ? html`
                    <img src=${this.user.picture} alt=${this.user.name} class="avatar" />
                ` : html`
                    <div class="avatar-placeholder">
                        ${this.getInitials(this.user.name)}
                    </div>
                `}
            </button>

            ${this.showDropdown ? html`
                <div class="dropdown">
                    <div class="dropdown-header">
                        <div class="dropdown-name">${this.user.name}</div>
                        <div class="dropdown-email">${this.user.email}</div>
                        <div class="dropdown-credits">💎 ${this.user.credits} credits</div>
                    </div>
                    <div class="dropdown-menu">
                        <button class="menu-item" @click=${() => alert('Profile coming soon!')}>
                            Profile
                        </button>
                        <button class="menu-item" @click=${() => alert('Settings coming soon!')}>
                            Settings
                        </button>
                        <button class="menu-item danger" @click=${this.handleLogout}>
                            Sign out
                        </button>
                    </div>
                </div>
            ` : ''}
        `;
    }
}
