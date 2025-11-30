-- Migration: Add iframely metadata fields
-- Preserves rich preview metadata from iframely API
-- Date: 2025-11-30

-- Add author field (separate from byline for clarity)
ALTER TABLE core.pages ADD COLUMN IF NOT EXISTS author TEXT;

-- Add thumbnail URL from iframely
ALTER TABLE core.pages ADD COLUMN IF NOT EXISTS thumbnail_url TEXT;

-- Add description for preview (separate from full content)
ALTER TABLE core.pages ADD COLUMN IF NOT EXISTS description TEXT;

-- Comments
COMMENT ON COLUMN core.pages.author IS 'Author name from iframely metadata';
COMMENT ON COLUMN core.pages.thumbnail_url IS 'Preview thumbnail URL from iframely';
COMMENT ON COLUMN core.pages.description IS 'Short description/excerpt from iframely';
