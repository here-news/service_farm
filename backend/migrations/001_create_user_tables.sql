-- Migration: Create user, comment, and chat_session tables
-- This adds community features to the service_farm database

-- ============================================================================
-- USERS TABLE
-- ============================================================================
CREATE TABLE IF NOT EXISTS users (
    user_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    google_id VARCHAR(255) UNIQUE NOT NULL,

    -- Profile
    name VARCHAR(255),
    picture_url TEXT,

    -- Credits and reputation
    credits_balance INTEGER DEFAULT 1000 NOT NULL,
    reputation INTEGER DEFAULT 0 NOT NULL,

    -- Subscription
    subscription_tier VARCHAR(50),  -- 'free', 'premium', etc.
    is_active BOOLEAN DEFAULT true NOT NULL,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    last_login TIMESTAMP WITH TIME ZONE
);

-- Indexes for users table
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_google_id ON users(google_id);
CREATE INDEX IF NOT EXISTS idx_users_created_at ON users(created_at);


-- ============================================================================
-- COMMENTS TABLE
-- ============================================================================
CREATE TABLE IF NOT EXISTS comments (
    id VARCHAR(11) PRIMARY KEY,  -- cm_xxxxxxxx format
    user_id UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,

    -- Parent entity (at least one must be set)
    event_id VARCHAR(11),  -- ev_xxxxxxxx (references events, but no FK for flexibility)
    page_id VARCHAR(11),   -- pg_xxxxxxxx (references pages, but no FK for flexibility)

    -- Content
    text TEXT NOT NULL,

    -- Threading
    parent_comment_id VARCHAR(11) REFERENCES comments(id) ON DELETE CASCADE,

    -- Reaction type (optional classification)
    reaction_type VARCHAR(20),  -- 'support', 'refute', 'question', 'comment'

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,

    -- Ensure at least one parent is set
    CONSTRAINT check_parent_entity CHECK (
        event_id IS NOT NULL OR page_id IS NOT NULL
    )
);

-- Indexes for comments table
CREATE INDEX IF NOT EXISTS idx_comments_user_id ON comments(user_id);
CREATE INDEX IF NOT EXISTS idx_comments_event_id ON comments(event_id);
CREATE INDEX IF NOT EXISTS idx_comments_page_id ON comments(page_id);
CREATE INDEX IF NOT EXISTS idx_comments_parent ON comments(parent_comment_id);
CREATE INDEX IF NOT EXISTS idx_comments_created_at ON comments(created_at);


-- ============================================================================
-- CHAT_SESSIONS TABLE
-- ============================================================================
CREATE TABLE IF NOT EXISTS chat_sessions (
    id VARCHAR(11) PRIMARY KEY,  -- cs_xxxxxxxx format
    event_id VARCHAR(11) NOT NULL,  -- ev_xxxxxxxx
    user_id UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,

    -- Message tracking
    message_count INTEGER DEFAULT 0 NOT NULL,

    -- Credits
    cost INTEGER DEFAULT 10 NOT NULL,  -- Credits spent to unlock

    -- Status
    status VARCHAR(20) DEFAULT 'active' NOT NULL,  -- 'active', 'exhausted'

    -- Timestamps
    unlocked_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    last_message_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,

    -- Unique constraint: one session per event per user
    CONSTRAINT unique_event_user UNIQUE (event_id, user_id)
);

-- Indexes for chat_sessions table
CREATE INDEX IF NOT EXISTS idx_chat_sessions_user_id ON chat_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_chat_sessions_event_id ON chat_sessions(event_id);
CREATE INDEX IF NOT EXISTS idx_chat_sessions_created_at ON chat_sessions(created_at);
CREATE INDEX IF NOT EXISTS idx_chat_sessions_status ON chat_sessions(status);


-- ============================================================================
-- GRANT PERMISSIONS (if needed)
-- ============================================================================
-- Adjust based on your database user
-- GRANT ALL PRIVILEGES ON TABLE users TO herenews_user;
-- GRANT ALL PRIVILEGES ON TABLE comments TO herenews_user;
-- GRANT ALL PRIVILEGES ON TABLE chat_sessions TO herenews_user;
