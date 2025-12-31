/**
 * Inquiry API Types - matches backend reee/inquiry/types.py
 */

// Enums
export type InquiryStatus = 'open' | 'resolved' | 'stale' | 'closed'
export type RigorLevel = 'A' | 'B' | 'C'
export type ContributionType = 'evidence' | 'refutation' | 'attribution' | 'scope_correction' | 'disambiguation'
export type TaskType = 'need_primary_source' | 'unresolved_conflict' | 'single_source_only' | 'high_entropy' | 'stale'
export type ObservationKind = 'point' | 'lower_bound' | 'upper_bound' | 'interval' | 'approximate' | 'none'
export type SchemaType = 'monotone_count' | 'categorical' | 'boolean' | 'report_truth' | 'quote_authenticity' | 'custom' | 'forecast'

// Core Types
export interface InquirySchema {
  schema_type: SchemaType
  categories?: string[]
  count_scale?: 'small' | 'medium' | 'large'
  count_max?: number
  count_monotone?: boolean
  hypotheses?: string[]
  rigor: RigorLevel
}

export interface InquirySummary {
  id: string
  title: string
  status: InquiryStatus
  rigor: RigorLevel
  schema_type: SchemaType
  posterior_map: any
  posterior_probability: number
  entropy_bits: number
  stake: number  // "bounty" in UI
  contributions: number
  open_tasks: number
  resolvable: boolean
  resolved_ago?: string
  scope_entities?: string[]
  cover_image?: string
  credible_interval?: [number, number]
}

export interface InquiryDetail extends InquirySummary {
  description?: string
  scope_keywords?: string[]
  scope_time_start?: string
  scope_time_end?: string
  normalized_entropy: number
  created_at: string
  updated_at: string
  created_by?: string
  blocking_tasks: string[]
}

// Observation (from TypedBeliefState)
export interface Observation {
  kind: ObservationKind | string  // Allow string for simulated data
  value_distribution: Record<string | number, number>
  bound_value?: number
  interval_low?: number
  interval_high?: number
  source: string
  claim_id?: string
  timestamp?: number
  extraction_confidence?: number  // Optional for simulated
  signals?: Record<string, any>
}

// Contribution (user submission)
export interface Contribution {
  id: string
  inquiry_id?: string  // Optional for simulated
  user_id?: string
  user_name?: string
  type: ContributionType | string  // Allow string for simulated
  text: string
  source_url?: string
  source_name?: string
  source?: string  // Alternative to source_name
  timestamp?: string
  extracted_value?: any
  observation_kind?: ObservationKind | string
  created_at: string
  processed?: boolean  // Optional for simulated
  claim_ids?: string[]
  posterior_impact?: number  // Optional for simulated
  impact?: number  // Alternative field name
}

// Task (from meta-claims)
export interface InquiryTask {
  id: string
  inquiry_id: string
  type: TaskType
  description: string
  bounty: number
  created_at: string
  claimed_by?: string
  completed: boolean
  completed_at?: string
  meta_claim_id?: string
}

// Surface (L2 cluster)
export interface Surface {
  id: string
  name: string
  claim_count: number
  sources: string[]
  entities?: string[]
  in_scope: boolean
  entropy?: number
  canonical_value?: any
  relations: SurfaceRelation[]
}

export interface SurfaceRelation {
  type: 'CONFIRMS' | 'SUPERSEDES' | 'CONFLICTS' | 'REFINES'
  target: string
  confidence?: number
}

// Belief State (posterior summary)
export interface BeliefState {
  map: any
  map_probability: number
  entropy_bits: number
  normalized_entropy: number
  observation_count: number
  log_scores?: number[]
  total_log_score: number
}

// Resolution status
export interface Resolution {
  status: InquiryStatus
  resolvable: boolean
  stable_since?: string
  blocking_tasks: string[]
  hours_stable?: number
}

// Full trace response
export interface InquiryTrace {
  inquiry: { id: string; title: string } | InquirySummary
  belief_state: BeliefState
  observations: Observation[]
  contributions: Contribution[]
  surfaces?: Surface[]
  tasks: InquiryTask[]
  resolution: Resolution
  posterior_top_10?: Array<{ value: any; probability: number }>
  claims?: Array<{
    id: string
    icon: string
    source: string
    text: string
    extracted_value: any
    observation_kind: string
  }>
}

// Replay state (for animation)
export interface ReplayState {
  currentIndex: number
  isPlaying: boolean
  speed: number // 1x, 2x, etc.
  snapshots: BeliefSnapshot[]
}

export interface BeliefSnapshot {
  index: number
  contribution: Contribution
  belief_state: BeliefState
  posterior_top_10: Array<{ value: any; probability: number }>
  entropy_delta: number
  probability_delta: number
}

// API Input types
export interface ContributionInput {
  type: ContributionType
  text: string
  source_url?: string
  source_name?: string
  extracted_value?: any
  observation_kind?: ObservationKind
  // For attribution
  attributed_to?: string
  original_source?: string
  // For scope correction
  correct_scope?: string
  out_of_scope_reason?: string
}

export interface StakeInput {
  amount: number
}

export interface CreateInquiryInput {
  title: string
  description?: string
  schema: InquirySchema
  scope_entities?: string[]
  scope_keywords?: string[]
  scope_time_start?: string
  scope_time_end?: string
  initial_stake?: number
}

// Alias for convenience
export type Task = InquiryTask
