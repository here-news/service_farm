export { default as InquiryCard } from './InquiryCard'
export { default as InquiryCarousel } from './InquiryCarousel'
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
