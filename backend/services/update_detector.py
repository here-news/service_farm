"""
Update detection for evolving claims.

Minimal MVP: Focus on numeric fields (casualties, alarm levels, missing counts)
to enable timeline convergence visualization.
"""
import re
import logging
from typing import Optional, List, Tuple
from models.claim import Claim

logger = logging.getLogger(__name__)


class UpdateDetector:
    """
    Detect when a claim updates/supersedes another claim.

    MVP scope: Numeric evolving facts
    - Casualty counts ("5 dead" → "50 dead" → "128 dead")
    - Alarm levels ("Level 2" → "Level 3")
    - Missing counts ("10 missing" → "150 missing")
    - Injured counts, area burned, etc.
    """

    # Topic patterns (numeric fields that typically evolve)
    TOPIC_PATTERNS = {
        'casualty_count': [
            r'(\d+)\s+(?:people\s+)?(?:have\s+)?died',
            r'(\d+)\s+(?:people\s+)?(?:have\s+been\s+)?(?:confirmed\s+)?dead',
            r'(\d+)\s+(?:people\s+)?(?:were\s+)?killed',
            r'(?:death\s+toll|fatalities)(?:\s+(?:rises?|jumps?)\s+to\s+)?[:\s]*(\d+)',
            r'claimed?\s+(?:at\s+least\s+)?(\d+)\s+lives',
            r'(?:at\s+least\s+)?(\d+)\s+(?:people\s+)?(?:have\s+been\s+)?(?:confirmed\s+)?dead',
        ],
        'missing_count': [
            r'(\d+)\s+(?:people\s+)?(?:are\s+)?missing',
            r'(\d+)\s+unaccounted\s+for',
        ],
        'injured_count': [
            r'(\d+)\s+(?:people\s+)?(?:were\s+)?injured',
            r'(\d+)\s+(?:people\s+)?(?:have\s+been\s+)?wounded',
        ],
        'alarm_level': [
            r'(?:alarm\s+)?level\s+(\d+)',
            r'raised\s+to\s+level\s+(\d+)',
        ],
        'evacuated_count': [
            r'(\d+)\s+(?:people\s+)?evacuated',
            r'(\d+)\s+(?:residents\s+)?rescued',
        ],
    }

    def extract_numeric_value(self, claim: Claim, topic_key: str) -> Optional[int]:
        """
        Extract numeric value for a topic from claim text.

        Returns:
            Integer value if found, None otherwise
        """
        if topic_key not in self.TOPIC_PATTERNS:
            return None

        text_lower = claim.text.lower()

        for pattern in self.TOPIC_PATTERNS[topic_key]:
            match = re.search(pattern, text_lower)
            if match:
                try:
                    value = int(match.group(1))
                    logger.debug(
                        f"Extracted {topic_key}={value} from: {claim.text[:80]}..."
                    )
                    return value
                except (ValueError, IndexError):
                    continue

        return None

    def detect_topic_key(self, claim: Claim) -> Optional[str]:
        """
        Auto-detect which topic this claim is about.

        Returns:
            topic_key if claim matches a known numeric pattern, None otherwise
        """
        for topic_key in self.TOPIC_PATTERNS.keys():
            if self.extract_numeric_value(claim, topic_key) is not None:
                return topic_key

        return None

    def is_update_of(
        self,
        new_claim: Claim,
        old_claim: Claim
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if new_claim updates/supersedes old_claim.

        Logic (minimal MVP):
        1. Both claims must have same topic_key (or auto-detect it)
        2. new_claim.reported_time > old_claim.reported_time
        3. Different numeric values (evolution)

        Returns:
            (is_update, topic_key)
        """
        # Auto-detect topic for both claims
        new_topic = new_claim.topic_key or self.detect_topic_key(new_claim)
        old_topic = old_claim.topic_key or self.detect_topic_key(old_claim)

        if not new_topic or not old_topic:
            return False, None

        if new_topic != old_topic:
            return False, None

        # Extract values
        new_value = self.extract_numeric_value(new_claim, new_topic)
        old_value = self.extract_numeric_value(old_claim, old_topic)

        if new_value is None or old_value is None:
            return False, None

        # Check temporal ordering
        # Priority: reported_time (when we learned it), fallback to event_time (when it happened)
        if new_claim.reported_time and old_claim.reported_time:
            if new_claim.reported_time < old_claim.reported_time:
                # New claim reported earlier - not an update
                return False, None
            elif new_claim.reported_time == old_claim.reported_time:
                # Same reported_time (e.g., timeline article) - use event_time
                if new_claim.event_time and old_claim.event_time:
                    if new_claim.event_time < old_claim.event_time:
                        # New claim about earlier event - not chronological update
                        return False, None
                # If event_times also equal or missing, proceed to value check

        # Different values = update
        if new_value != old_value:
            logger.info(
                f"📊 Update detected: {new_topic} {old_value} → {new_value}"
            )
            return True, new_topic

        return False, None

    def find_updates_in_event(
        self,
        claims: List[Claim]
    ) -> List[Tuple[Claim, Claim, str]]:
        """
        Find all update relationships within a list of claims.

        Returns:
            List of (new_claim, old_claim, topic_key) tuples
        """
        updates = []

        # Sort by reported_time to process chronologically
        sorted_claims = sorted(
            [c for c in claims if c.reported_time],
            key=lambda c: c.reported_time
        )

        for i, new_claim in enumerate(sorted_claims):
            # Check if this claim updates any earlier claim
            for old_claim in sorted_claims[:i]:
                is_update, topic_key = self.is_update_of(new_claim, old_claim)
                if is_update:
                    updates.append((new_claim, old_claim, topic_key))

        return updates

    def build_update_chains(
        self,
        claims: List[Claim]
    ) -> dict[str, List[Claim]]:
        """
        Build update chains for each topic.

        Returns:
            Dict mapping topic_key → list of claims (chronologically sorted)

        Example:
            {
                'casualty_count': [
                    Claim("5 dead", reported=Nov26),
                    Claim("50 dead", reported=Nov26),
                    Claim("128 dead", reported=Nov28)
                ]
            }
        """
        chains = {}

        # Auto-tag claims with topic_key
        for claim in claims:
            if not claim.topic_key:
                claim.topic_key = self.detect_topic_key(claim)

            if claim.topic_key:
                if claim.topic_key not in chains:
                    chains[claim.topic_key] = []
                chains[claim.topic_key].append(claim)

        # Sort each chain by reported_time
        for topic_key in chains:
            chains[topic_key].sort(
                key=lambda c: c.reported_time or c.created_at
            )

        return chains

    def get_current_value(
        self,
        chain: List[Claim]
    ) -> Optional[Claim]:
        """
        Get the most current (non-superseded) claim in a chain.

        Returns:
            Latest claim in chain, or None if chain is empty
        """
        if not chain:
            return None

        # Find latest non-superseded claim
        current = [c for c in chain if not c.is_superseded]

        if current:
            return current[-1]  # Last one chronologically

        # If all superseded, return latest anyway
        return chain[-1]
