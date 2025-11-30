-- Migration: Add rogue extraction queue for browser extension
-- Handles URLs that block automated scrapers (401/403)
-- Date: 2025-11-30

-- Create rogue extraction tasks table
CREATE TABLE IF NOT EXISTS core.rogue_extraction_tasks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    page_id UUID NOT NULL REFERENCES core.pages(id) ON DELETE CASCADE,
    url TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending', -- pending, processing, completed, failed

    -- Extracted metadata (filled by browser extension)
    metadata JSONB,

    -- Timing
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    processing_started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,

    -- Error tracking
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,

    -- Index for polling
    CONSTRAINT valid_status CHECK (status IN ('pending', 'processing', 'completed', 'failed'))
);

-- Index for fast polling by browser extension
CREATE INDEX IF NOT EXISTS idx_rogue_tasks_pending
ON core.rogue_extraction_tasks(status, created_at)
WHERE status = 'pending';

-- Index for lookup by page_id
CREATE INDEX IF NOT EXISTS idx_rogue_tasks_page_id
ON core.rogue_extraction_tasks(page_id);

-- Comments
COMMENT ON TABLE core.rogue_extraction_tasks IS 'Queue for browser extension to extract from URLs that block scrapers';
COMMENT ON COLUMN core.rogue_extraction_tasks.metadata IS 'JSON with title, description, thumbnail, content_text extracted by browser';
COMMENT ON COLUMN core.rogue_extraction_tasks.status IS 'pending: awaiting pickup, processing: extension working, completed: done, failed: gave up';
