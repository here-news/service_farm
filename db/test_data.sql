-- Test Data for Gen2 Schema
-- Hong Kong Apartment Fire case (from IMPLEMENTATION_BASIS.md)

-- =============================================================================
-- PAGES: News articles about Hong Kong fire
-- =============================================================================

INSERT INTO core.pages (id, url, canonical_url, title, content_text, language, status, word_count, pub_time) VALUES
('11111111-1111-1111-1111-111111111111',
 'https://www.bbc.com/news/world-asia-63762450',
 'https://www.bbc.com/news/world-asia-63762450',
 'Hong Kong fire: 44 dead in apartment block blaze',
 'At least 44 people have died in a fire at an apartment building in Hong Kong...',
 'en',
 'extracted',
 450,
 '2025-11-26 08:30:00+00'),

('22222222-2222-2222-2222-222222222222',
 'https://www.scmp.com/news/hong-kong/society/article/3199951',
 'https://www.scmp.com/news/hong-kong/society/article/3199951',
 '香港火災：44人死亡',
 '香港一座公寓樓發生火災，造成至少44人死亡...',
 'zh',
 'extracted',
 380,
 '2025-11-26 09:15:00+00'),

('33333333-3333-3333-3333-333333333333',
 'https://www.reuters.com/world/china/hong-kong-fire-2025-11-26',
 'https://www.reuters.com/world/china/hong-kong-fire-2025-11-26',
 'Hong Kong apartment fire kills 44, investigation launched',
 'Authorities in Hong Kong have launched an investigation into a deadly fire...',
 'en',
 'extracted',
 520,
 '2025-11-26 10:00:00+00');

-- =============================================================================
-- ENTITIES: People, organizations, locations
-- =============================================================================

-- Location: Hong Kong
INSERT INTO core.entities (id, canonical_name, entity_type, wikidata_qid, names_by_language, semantic_confidence, structural_confidence) VALUES
('e1111111-1111-1111-1111-111111111111',
 'Hong Kong',
 'LOCATION',
 'Q8646',
 '{"en": ["Hong Kong", "HK", "HKSAR"], "zh": ["香港", "香港特別行政區"], "fr": ["Hong Kong"]}'::jsonb,
 0.95,
 0.85);

-- Organization: Hong Kong Fire Services Department
INSERT INTO core.entities (id, canonical_name, entity_type, wikidata_qid, names_by_language, semantic_confidence, structural_confidence) VALUES
('e2222222-2222-2222-2222-222222222222',
 'Hong Kong Fire Services Department',
 'ORGANIZATION',
 'Q5895123',
 '{"en": ["Fire Services Department", "HKFSD"], "zh": ["香港消防處"]}'::jsonb,
 0.90,
 0.75);

-- Person: John Lee (Hong Kong Chief Executive)
INSERT INTO core.entities (id, canonical_name, entity_type, wikidata_qid, names_by_language, semantic_confidence, structural_confidence) VALUES
('e3333333-3333-3333-3333-333333333333',
 'John Lee Ka-chiu',
 'PERSON',
 'Q28839658',
 '{"en": ["John Lee", "Lee Ka-chiu"], "zh": ["李家超"]}'::jsonb,
 0.92,
 0.80);

-- Location: Jordan (Hong Kong neighborhood)
INSERT INTO core.entities (id, canonical_name, entity_type, wikidata_qid, names_by_language, semantic_confidence, structural_confidence) VALUES
('e4444444-4444-4444-4444-444444444444',
 'Jordan, Hong Kong',
 'LOCATION',
 'Q702984',
 '{"en": ["Jordan", "Jordan District"], "zh": ["佐敦"]}'::jsonb,
 0.88,
 0.70);

-- =============================================================================
-- EDGES: Relationships between entities
-- =============================================================================

-- Jordan is LOCATED_IN Hong Kong
INSERT INTO core.edges (from_entity_id, to_entity_id, edge_type, confidence, wikidata_properties, source) VALUES
('e4444444-4444-4444-4444-444444444444',
 'e1111111-1111-1111-1111-111111111111',
 'LOCATED_IN',
 0.95,
 '{"P276": 0.95}'::jsonb,
 'wikidata');

-- John Lee is EMPLOYED_BY Hong Kong government (simplified)
INSERT INTO core.edges (from_entity_id, to_entity_id, edge_type, confidence, source) VALUES
('e3333333-3333-3333-3333-333333333333',
 'e1111111-1111-1111-1111-111111111111',
 'LOCATED_IN',
 0.90,
 'extracted');

-- =============================================================================
-- PAGE_ENTITIES: Which pages mention which entities
-- =============================================================================

-- BBC article mentions Hong Kong, Fire Dept, John Lee, Jordan
INSERT INTO core.page_entities (page_id, entity_id, mention_count) VALUES
('11111111-1111-1111-1111-111111111111', 'e1111111-1111-1111-1111-111111111111', 5),
('11111111-1111-1111-1111-111111111111', 'e2222222-2222-2222-2222-222222222222', 3),
('11111111-1111-1111-1111-111111111111', 'e3333333-3333-3333-3333-333333333333', 2),
('11111111-1111-1111-1111-111111111111', 'e4444444-4444-4444-4444-444444444444', 4);

-- SCMP article (Chinese) mentions same entities
INSERT INTO core.page_entities (page_id, entity_id, mention_count) VALUES
('22222222-2222-2222-2222-222222222222', 'e1111111-1111-1111-1111-111111111111', 6),
('22222222-2222-2222-2222-222222222222', 'e2222222-2222-2222-2222-222222222222', 4),
('22222222-2222-2222-2222-222222222222', 'e4444444-4444-4444-4444-444444444444', 3);

-- Reuters article mentions Hong Kong, Jordan
INSERT INTO core.page_entities (page_id, entity_id, mention_count) VALUES
('33333333-3333-3333-3333-333333333333', 'e1111111-1111-1111-1111-111111111111', 4),
('33333333-3333-3333-3333-333333333333', 'e4444444-4444-4444-4444-444444444444', 5);

-- =============================================================================
-- CLAIMS: Atomic facts extracted from pages
-- =============================================================================

-- Claims from BBC article
INSERT INTO core.claims (id, page_id, text, event_time, confidence, modality) VALUES
('c1111111-1111-1111-1111-111111111111',
 '11111111-1111-1111-1111-111111111111',
 '44 people died in a fire at an apartment building in Hong Kong',
 '2025-11-26 03:00:00+00',
 0.95,
 'observation'),

('c2222222-2222-2222-2222-222222222222',
 '11111111-1111-1111-1111-111111111111',
 'The fire occurred in the Jordan neighborhood',
 '2025-11-26 03:00:00+00',
 0.90,
 'observation'),

('c3333333-3333-3333-3333-333333333333',
 '11111111-1111-1111-1111-111111111111',
 'Chief Executive John Lee expressed condolences to the victims',
 '2025-11-26 10:00:00+00',
 0.88,
 'action');

-- Claims from SCMP article
INSERT INTO core.claims (id, page_id, text, event_time, confidence, modality) VALUES
('c4444444-4444-4444-4444-444444444444',
 '22222222-2222-2222-2222-222222222222',
 '消防處正在調查火災原因',  -- Fire dept investigating cause
 '2025-11-26 05:00:00+00',
 0.85,
 'observation');

-- Claims from Reuters article
INSERT INTO core.claims (id, page_id, text, event_time, confidence, modality) VALUES
('c5555555-5555-5555-5555-555555555555',
 '33333333-3333-3333-3333-333333333333',
 'Authorities launched an investigation into the deadly fire',
 '2025-11-26 06:00:00+00',
 0.92,
 'action');

-- =============================================================================
-- CLAIM_ENTITIES: Which entities are mentioned in which claims
-- =============================================================================

INSERT INTO core.claim_entities (claim_id, entity_id, mention_context) VALUES
('c1111111-1111-1111-1111-111111111111', 'e1111111-1111-1111-1111-111111111111', '...apartment building in Hong Kong'),
('c2222222-2222-2222-2222-222222222222', 'e4444444-4444-4444-4444-444444444444', '...in the Jordan neighborhood'),
('c3333333-3333-3333-3333-333333333333', 'e3333333-3333-3333-3333-333333333333', 'Chief Executive John Lee...'),
('c4444444-4444-4444-4444-444444444444', 'e2222222-2222-2222-2222-222222222222', '...Fire Services Department investigating...'),
('c5555555-5555-5555-5555-555555555555', 'e1111111-1111-1111-1111-111111111111', 'Authorities in Hong Kong...');

-- =============================================================================
-- EVENTS: Event formed from claim clustering
-- =============================================================================

INSERT INTO core.events (id, title, summary, event_type, location, event_start, event_end, confidence, coherence_score, languages, claims_count) VALUES
('00000001-1111-1111-1111-111111111111',
 'Hong Kong Apartment Fire',
 'Deadly fire at apartment building in Jordan, Hong Kong, killed 44 people. Authorities launched investigation.',
 'incident',
 'Jordan, Hong Kong',
 '2025-11-26 03:00:00+00',
 '2025-11-26 06:00:00+00',
 0.92,
 0.88,
 '["en", "zh"]'::jsonb,
 5);

-- =============================================================================
-- PAGE_EVENTS: Link pages to events
-- =============================================================================

INSERT INTO core.page_events (page_id, event_id) VALUES
('11111111-1111-1111-1111-111111111111', '00000001-1111-1111-1111-111111111111'),
('22222222-2222-2222-2222-222222222222', '00000001-1111-1111-1111-111111111111'),
('33333333-3333-3333-3333-333333333333', '00000001-1111-1111-1111-111111111111');

-- =============================================================================
-- EVENT_ENTITIES: Key entities in event
-- =============================================================================

INSERT INTO core.event_entities (event_id, entity_id, role) VALUES
('00000001-1111-1111-1111-111111111111', 'e1111111-1111-1111-1111-111111111111', 'location'),
('00000001-1111-1111-1111-111111111111', 'e4444444-4444-4444-4444-444444444444', 'location'),
('00000001-1111-1111-1111-111111111111', 'e2222222-2222-2222-2222-222222222222', 'investigator'),
('00000001-1111-1111-1111-111111111111', 'e3333333-3333-3333-3333-333333333333', 'official');

-- =============================================================================
-- BRIDGE: User submission metadata
-- =============================================================================

INSERT INTO bridge.artifact_metadata (artifact_id, submitted_by_id, submission_source, submitted_at) VALUES
('11111111-1111-1111-1111-111111111111', 'aaaaaaaa-0000-0000-0000-000000000123', 'browser_extension', '2025-11-26 08:35:00+00'),
('22222222-2222-2222-2222-222222222222', 'aaaaaaaa-0000-0000-0000-000000000123', 'browser_extension', '2025-11-26 09:20:00+00'),
('33333333-3333-3333-3333-333333333333', 'bbbbbbbb-0000-0000-0000-000000000456', 'api', '2025-11-26 10:05:00+00');
