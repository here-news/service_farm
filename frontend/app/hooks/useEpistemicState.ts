import { useState, useEffect, useCallback } from 'react';
import { EpistemicState } from '../components/epistemic';

interface UseEpistemicStateResult {
  state: EpistemicState | null;
  loading: boolean;
  error: string | null;
  refetch: () => void;
}

export function useEpistemicState(eventId: string | undefined): UseEpistemicStateResult {
  const [state, setState] = useState<EpistemicState | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchEpistemicState = useCallback(async () => {
    if (!eventId) return;

    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`/api/event/${eventId}/epistemic`);

      if (!response.ok) {
        if (response.status === 404) {
          // Epistemic state not available yet - not an error
          setState(null);
          return;
        }
        throw new Error('Failed to fetch epistemic state');
      }

      const data = await response.json();

      // Transform API response to EpistemicState
      const epistemicState: EpistemicState = {
        source_count: data.source_count,
        source_diversity: data.source_diversity,
        claim_count: data.claim_count,
        coverage: data.coverage,
        heat: data.heat,
        has_contradiction: data.has_contradiction,
        gaps: data.gaps || [],
        last_updated: data.last_updated
      };

      setState(epistemicState);
    } catch (err: any) {
      setError(err.message || 'Failed to load epistemic state');
    } finally {
      setLoading(false);
    }
  }, [eventId]);

  useEffect(() => {
    fetchEpistemicState();
  }, [fetchEpistemicState]);

  return {
    state,
    loading,
    error,
    refetch: fetchEpistemicState
  };
}

export default useEpistemicState;
