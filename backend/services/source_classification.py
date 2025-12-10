"""
Source Classification - Conservative priors for news sources

Assigns source_type and base_prior to publishers at extraction time.
These values are stored on the Entity node and used during event analysis.

Source types are determined by:
1. Wikidata P31 (instance of) → mapped to source_type via WIKIDATA_SOURCE_TYPES
2. Fallback: domain heuristics (for publishers without QID)

Bayesian Priors (Jaynes-informed):
- Priors represent UNCERTAINTY about source accuracy
- Conservative values near 0.5 (maximum entropy)
- Byline adds accountability (+0.05)
- No prejudice - track record will refine over time
"""
from dataclasses import dataclass
from typing import Tuple, Optional, List
import re


@dataclass
class SourcePrior:
    """Source reliability prior - our state of knowledge about source accuracy"""
    base: float
    byline_bonus: float = 0.05
    reason: str = ""


# Conservative priors - represent UNCERTAINTY, not trust
# These are our starting points before track record accumulates
SOURCE_PRIORS = {
    'wire': SourcePrior(0.60, 0.05, "Wire services have editorial standards but still make errors"),
    'newspaper': SourcePrior(0.55, 0.05, "Newspapers vary in quality; byline adds accountability"),
    'broadcaster': SourcePrior(0.58, 0.05, "Broadcasters have editorial oversight"),
    'magazine': SourcePrior(0.55, 0.05, "Magazines vary; longer editorial cycles"),
    'official': SourcePrior(0.55, 0.0, "Official sources may be preliminary, political, or face-saving"),
    'aggregator': SourcePrior(0.50, 0.0, "Maximum entropy - unknown reliability"),
    'unknown': SourcePrior(0.50, 0.0, "Maximum entropy - no information"),
}

# Wikidata P31 QIDs mapped to source types
# These are checked directly AND via P279 subclass hierarchy
WIKIDATA_SOURCE_TYPES = {
    # Wire services / News agencies
    'Q192283': 'wire',      # news agency
    'Q1153191': 'wire',     # news agency (alternate)

    # Newspapers
    'Q11032': 'newspaper',    # newspaper
    'Q1110794': 'newspaper',  # daily newspaper
    'Q106650967': 'newspaper', # tabloid newspaper
    'Q7094076': 'newspaper',  # online newspaper

    # Broadcasters
    'Q1002697': 'broadcaster',  # news broadcaster
    'Q1126006': 'broadcaster',  # public broadcaster
    'Q15265344': 'broadcaster', # broadcaster
    'Q1616075': 'broadcaster',  # television station
    'Q17232649': 'broadcaster', # TV news program

    # Magazines
    'Q41298': 'magazine',     # magazine
    'Q1684600': 'magazine',   # news magazine

    # Government / Official
    'Q327333': 'official',    # government agency
    'Q2659904': 'official',   # government organization
}

# Fallback: domain patterns (when no Wikidata QID)
DOMAIN_PATTERNS = {
    'wire': {'ap', 'reuters', 'afp', 'apnews', 'xinhua', 'efe', 'ansa', 'dpa'},
    'official': {'gov', '.gov.'},
    'aggregator': {'msn.com', 'yahoo.com', 'news.google', 'flipboard'},
}


def classify_source_from_wikidata(p31_qids: List[str]) -> Optional[str]:
    """
    Classify source type from Wikidata P31 (instance of) values.

    This is the preferred method when publisher has a Wikidata QID.

    Args:
        p31_qids: List of P31 QIDs from Wikidata entity

    Returns:
        source_type or None if no match found
    """
    if not p31_qids:
        return None

    for qid in p31_qids:
        if qid in WIKIDATA_SOURCE_TYPES:
            return WIKIDATA_SOURCE_TYPES[qid]

    return None


def classify_source_by_domain(domain: str, site_name: Optional[str] = None) -> Tuple[str, bool]:
    """
    Classify source type from domain (fallback when no Wikidata QID).

    Called during publisher identification in knowledge worker
    when Wikidata doesn't provide P31 information.

    Args:
        domain: Domain of the source (e.g., "reuters.com")
        site_name: Optional site name (e.g., "Reuters")

    Returns:
        (source_type, has_byline): Source classification
    """
    if not domain:
        return ('unknown', False)

    domain_lower = domain.lower()
    site_lower = (site_name or '').lower()

    # Check domain patterns by source type
    for source_type, patterns in DOMAIN_PATTERNS.items():
        for pattern in patterns:
            if pattern in domain_lower or pattern in site_lower:
                has_byline = source_type not in ('official', 'aggregator')
                return (source_type, has_byline)

    # Default heuristics for news-looking domains
    news_indicators = ['news', 'times', 'post', 'herald', 'tribune', 'daily', 'gazette']
    for indicator in news_indicators:
        if indicator in domain_lower:
            return ('newspaper', True)

    return ('unknown', False)


def compute_base_prior(source_type: str, has_byline: bool = False) -> float:
    """
    Compute base prior probability for a source type.

    Args:
        source_type: One of 'wire', 'official', 'local_news', 'international', 'aggregator', 'unknown'
        has_byline: Whether source typically has bylined articles

    Returns:
        Base prior probability (0.50 - 0.65)
    """
    prior_info = SOURCE_PRIORS.get(source_type, SOURCE_PRIORS['unknown'])
    prior = prior_info.base
    if has_byline:
        prior += prior_info.byline_bonus
    return round(prior, 2)


def get_prior_reason(source_type: str) -> str:
    """Get explanation for source prior."""
    prior_info = SOURCE_PRIORS.get(source_type, SOURCE_PRIORS['unknown'])
    return prior_info.reason
