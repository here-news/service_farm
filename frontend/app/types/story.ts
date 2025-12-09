export interface Person {
  id: string
  name: string
  image?: string
  wikidata_thumbnail?: string
  wikidata_description?: string
  description?: string
}

export interface Entity {
  id: string
  canonical_id?: string
  name: string
  wikidata_qid?: string
  wikidata_thumbnail?: string
  wikidata_description?: string
  description?: string
}

export interface Claim {
  id: string
  text: string
  confidence?: number
}

export interface Artifact {
  url: string
  title: string
  domain: string
  published_at?: string
}

export interface RelatedStory {
  id: string
  title: string
  shared_entities: number
}

export interface CoherenceBreakdown {
  score: number
  entity_overlap: number
  centrality: number
  claim_density: number
  breakdown: {
    shared_entities: number
    total_entities: number
    avg_entity_stories: number
    claim_count: number
  }
}

export interface Story {
  story_id?: string  // Legacy support
  event_id?: string  // New field
  id?: string
  title: string
  summary?: string
  description?: string
  content?: string
  created_at?: string
  last_updated?: string
  people?: Person[]
  cover_image?: string
  claim_count?: number
  coherence?: number
  health_indicator?: string
  tcf_score?: number
  timely?: number
  funding?: number
  explanation?: string
  event_type?: string
  status?: string
  confidence?: number

  // Enriched data
  entities?: {
    people: Entity[]
    organizations: Entity[]
    locations: Entity[]
  }
  claims?: Claim[]
  artifacts?: Artifact[]
  artifact_count?: number
  related_stories?: RelatedStory[]
  coherence_breakdown?: CoherenceBreakdown
}

export interface FeedResponse {
  status: string
  count: number
  events?: Story[]  // Events use Story type for now
  stories?: Story[] // Legacy support
}
