export { default as InquiryCard } from './InquiryCard'
export { default as InquiryCarousel } from './InquiryCarousel'
export { default as ReplayTimeline } from './ReplayTimeline'
export { default as BeliefStatePanel } from './BeliefStatePanel'
export { default as SurfacesTopology } from './SurfacesTopology'
export { default as BountyPanel } from './BountyPanel'
export type { InquirySummary } from './InquiryCard'

// Re-export types from types/inquiry for convenience
export type {
  InquiryDetail,
  InquiryTrace,
  Contribution,
  InquiryTask,
  Surface,
  BeliefState,
  BeliefSnapshot,
  Observation,
  ContributionInput
} from '../../types/inquiry'
