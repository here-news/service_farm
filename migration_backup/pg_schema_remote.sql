--
-- PostgreSQL database dump
--

\restrict Vqms62Tp7tTLTxayeQ9aEY0zOqs6werINSBy0ZauPsfvw0mTWvDlT7ivPq7SiUf

-- Dumped from database version 16.11 (Debian 16.11-1.pgdg12+1)
-- Dumped by pg_dump version 16.11 (Debian 16.11-1.pgdg12+1)

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- Name: content; Type: SCHEMA; Schema: -; Owner: phi_user
--

CREATE SCHEMA content;


ALTER SCHEMA content OWNER TO phi_user;

--
-- Name: SCHEMA content; Type: COMMENT; Schema: -; Owner: phi_user
--

COMMENT ON SCHEMA content IS 'Content storage - page text and embeddings only';


--
-- Name: core; Type: SCHEMA; Schema: -; Owner: phi_user
--

CREATE SCHEMA core;


ALTER SCHEMA core OWNER TO phi_user;

--
-- Name: SCHEMA core; Type: COMMENT; Schema: -; Owner: phi_user
--

COMMENT ON SCHEMA core IS 'Core knowledge graph: pages, entities, events, claims, edges';


--
-- Name: edge_type; Type: TYPE; Schema: core; Owner: phi_user
--

CREATE TYPE core.edge_type AS ENUM (
    'LOCATED_IN',
    'BORDERS',
    'EMPLOYED_BY',
    'MEMBER_OF',
    'PART_OF',
    'OWNS',
    'PARTICIPATED_IN',
    'CAUSED_BY',
    'FOLLOWED_BY',
    'RELATED_TO',
    'MENTIONED_IN',
    'CO_OCCURS_WITH',
    'CONTRADICTS',
    'SUPPORTS'
);


ALTER TYPE core.edge_type OWNER TO phi_user;

--
-- Name: entity_type; Type: TYPE; Schema: core; Owner: phi_user
--

CREATE TYPE core.entity_type AS ENUM (
    'PERSON',
    'ORGANIZATION',
    'LOCATION',
    'EVENT',
    'CONCEPT',
    'PRODUCT',
    'WORK_OF_ART',
    'LAW',
    'OTHER'
);


ALTER TYPE core.entity_type OWNER TO phi_user;

--
-- Name: find_entity_network(uuid, integer, double precision); Type: FUNCTION; Schema: core; Owner: phi_user
--

CREATE FUNCTION core.find_entity_network(p_entity_id uuid, p_max_depth integer DEFAULT 2, p_min_confidence double precision DEFAULT 0.7) RETURNS TABLE(entity_id uuid, canonical_name text, depth integer, path_confidence double precision)
    LANGUAGE plpgsql
    AS $$
BEGIN
    RETURN QUERY
    WITH RECURSIVE entity_network AS (
        -- Base case
        SELECT
            e.id AS entity_id,
            e.canonical_name,
            0 AS depth,
            1.0::float AS path_confidence
        FROM core.entities e
        WHERE e.id = p_entity_id

        UNION

        -- Recursive case
        SELECT
            e.id AS entity_id,
            e.canonical_name,
            en.depth + 1 AS depth,
            (en.path_confidence * eg.confidence)::float AS path_confidence
        FROM core.entities e
        JOIN core.edges eg ON eg.to_entity_id = e.id
        JOIN entity_network en ON eg.from_entity_id = en.entity_id
        WHERE en.depth < p_max_depth
          AND eg.confidence >= p_min_confidence
    )
    SELECT DISTINCT
        entity_network.entity_id,
        entity_network.canonical_name,
        entity_network.depth,
        entity_network.path_confidence
    FROM entity_network
    ORDER BY depth, path_confidence DESC;
END;
$$;


ALTER FUNCTION core.find_entity_network(p_entity_id uuid, p_max_depth integer, p_min_confidence double precision) OWNER TO phi_user;

--
-- Name: get_page_summary(uuid); Type: FUNCTION; Schema: core; Owner: phi_user
--

CREATE FUNCTION core.get_page_summary(p_page_id uuid) RETURNS json
    LANGUAGE plpgsql
    AS $$
DECLARE
    result JSON;
BEGIN
    SELECT json_build_object(
        'page_id', p.id,
        'title', p.title,
        'status', p.status,
        'entities_count', COUNT(DISTINCT pe.entity_id),
        'events_count', COUNT(DISTINCT pev.event_id),
        'claims_count', COUNT(DISTINCT c.id)
    ) INTO result
    FROM core.pages p
    LEFT JOIN core.page_entities pe ON p.id = pe.page_id
    LEFT JOIN core.page_events pev ON p.id = pev.page_id
    LEFT JOIN core.claims c ON p.id = c.page_id
    WHERE p.id = p_page_id
    GROUP BY p.id, p.title, p.status;

    RETURN result;
END;
$$;


ALTER FUNCTION core.get_page_summary(p_page_id uuid) OWNER TO phi_user;

--
-- Name: update_updated_at_column(); Type: FUNCTION; Schema: core; Owner: phi_user
--

CREATE FUNCTION core.update_updated_at_column() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;


ALTER FUNCTION core.update_updated_at_column() OWNER TO phi_user;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: event_embeddings; Type: TABLE; Schema: content; Owner: phi_user
--

CREATE TABLE content.event_embeddings (
    event_id text NOT NULL,
    embedding public.vector(1536),
    created_at timestamp with time zone DEFAULT now()
);


ALTER TABLE content.event_embeddings OWNER TO phi_user;

--
-- Name: TABLE event_embeddings; Type: COMMENT; Schema: content; Owner: phi_user
--

COMMENT ON TABLE content.event_embeddings IS 'Event embeddings for similarity search.';


--
-- Name: claim_embeddings; Type: TABLE; Schema: core; Owner: phi_user
--

CREATE TABLE core.claim_embeddings (
    claim_id text NOT NULL,
    embedding public.vector(1536) NOT NULL,
    model text DEFAULT 'text-embedding-3-small'::text,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE core.claim_embeddings OWNER TO phi_user;

--
-- Name: event_embeddings; Type: TABLE; Schema: core; Owner: phi_user
--

CREATE TABLE core.event_embeddings (
    event_id text NOT NULL,
    embedding public.vector(1536) NOT NULL,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE core.event_embeddings OWNER TO phi_user;

--
-- Name: pages; Type: TABLE; Schema: core; Owner: phi_user
--

CREATE TABLE core.pages (
    id text DEFAULT gen_random_uuid() NOT NULL,
    url text NOT NULL,
    canonical_url text NOT NULL,
    title text,
    description text,
    content_text text,
    byline text,
    author text,
    thumbnail_url text,
    site_name text,
    domain text,
    language character varying(10) DEFAULT 'en'::character varying,
    language_confidence double precision DEFAULT 0.0,
    metadata_confidence double precision DEFAULT 0.0,
    word_count integer DEFAULT 0,
    pub_time timestamp with time zone,
    status character varying(50) DEFAULT 'stub'::character varying,
    current_stage character varying(100),
    error_message text,
    is_healthy boolean DEFAULT true,
    has_screenshot boolean DEFAULT false,
    has_html boolean DEFAULT false,
    has_metadata boolean DEFAULT false,
    metadata jsonb DEFAULT '{}'::jsonb,
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now(),
    completed_at timestamp with time zone,
    embedding public.vector(1536),
    id_new text
);


ALTER TABLE core.pages OWNER TO phi_user;

--
-- Name: TABLE pages; Type: COMMENT; Schema: core; Owner: phi_user
--

COMMENT ON TABLE core.pages IS 'Artifacts + content (merged from Gen1 extraction_tasks + Neo4j Page)';


--
-- Name: rogue_extraction_tasks; Type: TABLE; Schema: core; Owner: phi_user
--

CREATE TABLE core.rogue_extraction_tasks (
    id text DEFAULT (gen_random_uuid())::text NOT NULL,
    page_id text NOT NULL,
    url text NOT NULL,
    status text DEFAULT 'pending'::text NOT NULL,
    metadata jsonb,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    processing_started_at timestamp with time zone,
    completed_at timestamp with time zone,
    error_message text,
    retry_count integer DEFAULT 0,
    CONSTRAINT valid_rogue_status CHECK ((status = ANY (ARRAY['pending'::text, 'processing'::text, 'completed'::text, 'failed'::text])))
);


ALTER TABLE core.rogue_extraction_tasks OWNER TO phi_user;

--
-- Name: event_embeddings event_embeddings_pkey; Type: CONSTRAINT; Schema: content; Owner: phi_user
--

ALTER TABLE ONLY content.event_embeddings
    ADD CONSTRAINT event_embeddings_pkey PRIMARY KEY (event_id);


--
-- Name: claim_embeddings claim_embeddings_pkey; Type: CONSTRAINT; Schema: core; Owner: phi_user
--

ALTER TABLE ONLY core.claim_embeddings
    ADD CONSTRAINT claim_embeddings_pkey PRIMARY KEY (claim_id);


--
-- Name: event_embeddings event_embeddings_pkey; Type: CONSTRAINT; Schema: core; Owner: phi_user
--

ALTER TABLE ONLY core.event_embeddings
    ADD CONSTRAINT event_embeddings_pkey PRIMARY KEY (event_id);


--
-- Name: pages pages_canonical_url_key; Type: CONSTRAINT; Schema: core; Owner: phi_user
--

ALTER TABLE ONLY core.pages
    ADD CONSTRAINT pages_canonical_url_key UNIQUE (canonical_url);


--
-- Name: pages pages_pkey; Type: CONSTRAINT; Schema: core; Owner: phi_user
--

ALTER TABLE ONLY core.pages
    ADD CONSTRAINT pages_pkey PRIMARY KEY (id);


--
-- Name: rogue_extraction_tasks rogue_extraction_tasks_pkey; Type: CONSTRAINT; Schema: core; Owner: phi_user
--

ALTER TABLE ONLY core.rogue_extraction_tasks
    ADD CONSTRAINT rogue_extraction_tasks_pkey PRIMARY KEY (id);


--
-- Name: idx_event_embeddings; Type: INDEX; Schema: content; Owner: phi_user
--

CREATE INDEX idx_event_embeddings ON content.event_embeddings USING ivfflat (embedding public.vector_cosine_ops);


--
-- Name: idx_pages_canonical_url; Type: INDEX; Schema: core; Owner: phi_user
--

CREATE INDEX idx_pages_canonical_url ON core.pages USING btree (canonical_url);


--
-- Name: idx_pages_created_at; Type: INDEX; Schema: core; Owner: phi_user
--

CREATE INDEX idx_pages_created_at ON core.pages USING btree (created_at DESC);


--
-- Name: idx_pages_domain; Type: INDEX; Schema: core; Owner: phi_user
--

CREATE INDEX idx_pages_domain ON core.pages USING btree (domain);


--
-- Name: idx_pages_embedding; Type: INDEX; Schema: core; Owner: phi_user
--

CREATE INDEX idx_pages_embedding ON core.pages USING ivfflat (embedding public.vector_cosine_ops);


--
-- Name: idx_pages_language; Type: INDEX; Schema: core; Owner: phi_user
--

CREATE INDEX idx_pages_language ON core.pages USING btree (language);


--
-- Name: idx_pages_metadata_gin; Type: INDEX; Schema: core; Owner: phi_user
--

CREATE INDEX idx_pages_metadata_gin ON core.pages USING gin (metadata);


--
-- Name: idx_pages_pub_time; Type: INDEX; Schema: core; Owner: phi_user
--

CREATE INDEX idx_pages_pub_time ON core.pages USING btree (pub_time DESC);


--
-- Name: idx_pages_status; Type: INDEX; Schema: core; Owner: phi_user
--

CREATE INDEX idx_pages_status ON core.pages USING btree (status);


--
-- Name: idx_pages_url; Type: INDEX; Schema: core; Owner: phi_user
--

CREATE INDEX idx_pages_url ON core.pages USING btree (url);


--
-- Name: idx_rogue_tasks_page_id; Type: INDEX; Schema: core; Owner: phi_user
--

CREATE INDEX idx_rogue_tasks_page_id ON core.rogue_extraction_tasks USING btree (page_id);


--
-- Name: idx_rogue_tasks_pending; Type: INDEX; Schema: core; Owner: phi_user
--

CREATE INDEX idx_rogue_tasks_pending ON core.rogue_extraction_tasks USING btree (status, created_at) WHERE (status = 'pending'::text);


--
-- Name: pages update_pages_updated_at; Type: TRIGGER; Schema: core; Owner: phi_user
--

CREATE TRIGGER update_pages_updated_at BEFORE UPDATE ON core.pages FOR EACH ROW EXECUTE FUNCTION core.update_updated_at_column();


--
-- PostgreSQL database dump complete
--

\unrestrict Vqms62Tp7tTLTxayeQ9aEY0zOqs6werINSBy0ZauPsfvw0mTWvDlT7ivPq7SiUf

