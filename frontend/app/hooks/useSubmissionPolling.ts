import { useEffect, useRef } from 'react'
import { EventSubmission } from '../components/home/PendingSubmission'

interface UseSubmissionPollingProps {
  submissions: EventSubmission[]
  onUpdate: (updatedSubmissions: EventSubmission[]) => void
  enabled: boolean
}

/**
 * Polls for submission status updates like classic_app's useSubmissions
 *
 * - Polls every 1 second
 * - Only polls submissions that are pending or extracting
 * - Stops polling when all submissions are completed/failed/blocked
 */
export function useSubmissionPolling({
  submissions,
  onUpdate,
  enabled
}: UseSubmissionPollingProps) {
  const pollIntervalRef = useRef<NodeJS.Timeout | null>(null)

  useEffect(() => {
    if (!enabled) {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current)
        pollIntervalRef.current = null
      }
      return
    }

    // Check if any submissions need polling
    const needsPolling = submissions.some(
      sub => sub.status === 'pending' || sub.status === 'extracting'
    )

    if (!needsPolling) {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current)
        pollIntervalRef.current = null
      }
      return
    }

    // Poll for updates
    const pollSubmissions = async () => {
      try {
        const response = await fetch('/api/events/mine', {
          credentials: 'include'
        })

        if (response.ok) {
          const data: EventSubmission[] = await response.json()
          onUpdate(data)
        }
      } catch (err) {
        console.error('Failed to poll submission status:', err)
      }
    }

    // Initial poll
    pollSubmissions()

    // Set up interval (1 second like classic_app)
    pollIntervalRef.current = setInterval(pollSubmissions, 1000)

    return () => {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current)
        pollIntervalRef.current = null
      }
    }
  }, [submissions, enabled, onUpdate])
}
