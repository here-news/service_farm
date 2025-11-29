-- Demo Database Schema
-- Simplified version of full architecture for quick demo

-- Pages table (artifacts + content combined for simplicity)
CREATE TABLE IF NOT EXISTS pages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    url TEXT NOT NULL,
    canonical_url TEXT UNIQUE NOT NULL,
    title TEXT,
    content_text TEXT,
    language VARCHAR(10) DEFAULT 'en',
    language_confidence FLOAT DEFAULT 0.0,
    word_count INT DEFAULT 0,
    status VARCHAR(50) DEFAULT 'stub',  -- stub, extracting, extracted, entities_extracted, complete, failed
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_pages_canonical ON pages(canonical_url);
CREATE INDEX idx_pages_status ON pages(status);
CREATE INDEX idx_pages_language ON pages(language);

-- Entities table
CREATE TABLE IF NOT EXISTS entities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    canonical_name TEXT NOT NULL,
    entity_type VARCHAR(50) NOT NULL,  -- PERSON, ORG, GPE, LOC
    language VARCHAR(10) DEFAULT 'en',
    confidence FLOAT DEFAULT 0.5,
    profile_summary TEXT,  -- Enriched profile summary
    profile_roles JSONB,   -- Array of roles/titles
    profile_affiliations JSONB,  -- Array of organizations/affiliations
    profile_key_facts JSONB,  -- Array of key facts
    profile_locations JSONB,  -- Array of associated locations
    mention_count INT DEFAULT 0,
    last_enriched_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(canonical_name, entity_type, language)
);

CREATE INDEX idx_entities_name ON entities(canonical_name);
CREATE INDEX idx_entities_type ON entities(entity_type);

-- Page-Entity relationships
CREATE TABLE IF NOT EXISTS page_entities (
    page_id UUID REFERENCES pages(id) ON DELETE CASCADE,
    entity_id UUID REFERENCES entities(id) ON DELETE CASCADE,
    mention_count INT DEFAULT 1,
    PRIMARY KEY (page_id, entity_id)
);

-- Claims table
CREATE TABLE IF NOT EXISTS claims (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    page_id UUID REFERENCES pages(id) ON DELETE CASCADE,
    text TEXT NOT NULL,
    confidence FLOAT DEFAULT 0.5,
    event_time TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_claims_page ON claims(page_id);

-- Events table
CREATE TABLE IF NOT EXISTS events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title TEXT NOT NULL,
    summary TEXT,
    event_type VARCHAR(50),  -- legal_proceeding, policy_announcement, incident, investigation, etc.
    location TEXT,
    event_start TIMESTAMP WITH TIME ZONE,
    event_end TIMESTAMP WITH TIME ZONE,
    confidence FLOAT DEFAULT 0.5,
    parent_event_id UUID REFERENCES events(id),  -- For event hierarchies
    event_scale VARCHAR(20) DEFAULT 'meso',  -- micro | meso | macro
    relationship_type VARCHAR(50),  -- PHASE_OF | CAUSED_BY | FOLLOWS
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Page-Event relationships
CREATE TABLE IF NOT EXISTS page_events (
    page_id UUID REFERENCES pages(id) ON DELETE CASCADE,
    event_id UUID REFERENCES events(id) ON DELETE CASCADE,
    PRIMARY KEY (page_id, event_id)
);

-- Event-Entity relationships
CREATE TABLE IF NOT EXISTS event_entities (
    event_id UUID REFERENCES events(id) ON DELETE CASCADE,
    entity_id UUID REFERENCES entities(id) ON DELETE CASCADE,
    PRIMARY KEY (event_id, entity_id)
);

-- Helper function to get page status summary
CREATE OR REPLACE FUNCTION get_page_summary(p_page_id UUID)
RETURNS JSON AS $$
DECLARE
    result JSON;
BEGIN
    SELECT json_build_object(
        'page_id', p.id,
        'title', p.title,
        'status', p.status,
        'entities_count', COUNT(DISTINCT pe.entity_id),
        'events_count', COUNT(DISTINCT pev.event_id)
    ) INTO result
    FROM pages p
    LEFT JOIN page_entities pe ON p.id = pe.page_id
    LEFT JOIN page_events pev ON p.id = pev.page_id
    WHERE p.id = p_page_id
    GROUP BY p.id, p.title, p.status;

    RETURN result;
END;
$$ LANGUAGE plpgsql;
