"""
Author/Byline Parser - Extract clean author names from messy byline fields

Byline fields often contain:
- Multiple authors: "John Doe, Jane Smith"
- CSS/HTML artifacts: "Position Relative Display Flex Width..."
- Author bios mixed in: "John covers China for AP..."
- Organizational credits: "AP Staff", "Reuters"

This parser extracts clean, individual author names for entity resolution.
"""
import re
import logging
from typing import List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ParsedAuthor:
    """Parsed author information."""
    name: str
    is_person: bool = True  # False if it's an organization like "AP Staff"
    confidence: float = 1.0


# Patterns that indicate CSS/HTML junk
CSS_PATTERNS = [
    r'\b(display|flex|width|height|position|relative|absolute|padding|margin|align|justify)\b',
    r'\b(font-size|font-weight|color|background|border|overflow|opacity)\b',
    r'\.wp-block',
    r'\bpx\b',
    r':\s*\d+',
    r'#[0-9a-fA-F]{3,6}\b',
]

# Patterns for author bio text (to remove)
BIO_PATTERNS = [
    r'\b(covers?|report(s|ing)?|based in|joined|received|graduated)\b',
    r'\b(editor|correspondent|contributor|writer|journalist)\b.*\b(at|for|with)\b',
    r'\b\d{4}\b',  # Years like "2023"
    r'\bB\.S\.|M\.A\.|Ph\.D\.',
]

# Organizational/generic author names (not real people)
ORG_AUTHORS = {
    'ap staff', 'reuters staff', 'staff', 'staff writer', 'staff reporter',
    'associated press', 'reuters', 'afp', 'ap', 'editorial board',
    'news desk', 'newsroom', 'the editors', 'admin', 'contributor',
}

# Common name separators
SEPARATORS = [',', ' and ', ' & ', ';', '|', '/']


def parse_byline(byline: str) -> List[ParsedAuthor]:
    """
    Parse a byline string into individual author names.

    Args:
        byline: Raw byline string from page metadata

    Returns:
        List of ParsedAuthor objects with clean names
    """
    if not byline or not byline.strip():
        return []

    # Check for CSS junk - if detected, likely corrupted
    for pattern in CSS_PATTERNS:
        if re.search(pattern, byline, re.IGNORECASE):
            logger.debug(f"Byline contains CSS artifacts, skipping: {byline[:50]}...")
            return []

    # Check for bio text - extract only the name portion
    byline_clean = byline
    for pattern in BIO_PATTERNS:
        match = re.search(pattern, byline, re.IGNORECASE)
        if match:
            # Take only text before the bio starts
            byline_clean = byline[:match.start()].strip()
            break

    # Split by common separators
    parts = [byline_clean]
    for sep in SEPARATORS:
        new_parts = []
        for part in parts:
            if sep == ' and ':
                # Be careful with "and" - only split if it looks like multiple names
                # e.g., "John and Jane" but not "John Anderson"
                subparts = re.split(r'\s+and\s+', part, flags=re.IGNORECASE)
                if len(subparts) > 1 and all(_looks_like_name(p) for p in subparts):
                    new_parts.extend(subparts)
                else:
                    new_parts.append(part)
            else:
                new_parts.extend(part.split(sep))
        parts = new_parts

    # Clean and validate each part
    authors = []
    seen_names = set()

    for part in parts:
        name = _clean_name(part)
        if not name:
            continue

        name_lower = name.lower()

        # Skip duplicates
        if name_lower in seen_names:
            continue
        seen_names.add(name_lower)

        # Check if it's an organization
        if name_lower in ORG_AUTHORS:
            authors.append(ParsedAuthor(
                name=name,
                is_person=False,
                confidence=0.9
            ))
            continue

        # Validate it looks like a person's name
        if _looks_like_name(name):
            # Higher confidence if it has typical name structure
            confidence = 0.95 if _has_name_structure(name) else 0.8
            authors.append(ParsedAuthor(
                name=name,
                is_person=True,
                confidence=confidence
            ))

    return authors


def _clean_name(name: str) -> Optional[str]:
    """Clean a potential author name."""
    if not name:
        return None

    # Strip whitespace and common prefixes
    name = name.strip()
    name = re.sub(r'^(by|written by|author[s]?:?)\s*', '', name, flags=re.IGNORECASE)
    name = re.sub(r'^(more|additional)\s+', '', name, flags=re.IGNORECASE)

    # Remove trailing punctuation
    name = name.rstrip('.,;:')

    # Remove content in parentheses (usually credentials or roles)
    name = re.sub(r'\s*\([^)]+\)\s*', ' ', name)

    # Remove email addresses
    name = re.sub(r'\S+@\S+\.\S+', '', name)

    # Normalize whitespace
    name = ' '.join(name.split())

    # Too short or too long
    if len(name) < 3 or len(name) > 50:
        return None

    # Too many words (likely not a name)
    if len(name.split()) > 5:
        return None

    return name


def _looks_like_name(text: str) -> bool:
    """Check if text looks like a person's name."""
    if not text:
        return False

    words = text.split()

    # Must have 2-4 words (single word names too ambiguous for authors)
    if not (2 <= len(words) <= 4):
        return False

    # First word should start with capital (or be all caps)
    if not (words[0][0].isupper() or words[0].isupper()):
        return False

    # Should not have numbers (except Roman numerals)
    if re.search(r'\d', text):
        return False

    # Should not have certain characters
    if re.search(r'[<>{}[\]@#$%^*+=]', text):
        return False

    return True


def _has_name_structure(name: str) -> bool:
    """Check if name has typical Western or Asian name structure."""
    words = name.split()

    if len(words) == 1:
        # Single word names (rare but valid for some cultures)
        return len(name) >= 3 and name[0].isupper()

    if len(words) == 2:
        # First Last or Last First
        return all(w[0].isupper() for w in words if w)

    if len(words) == 3:
        # First Middle Last or with hyphenated parts
        return all(w[0].isupper() for w in words if w and not w.startswith('-'))

    return False


# Convenience function for testing
def extract_authors(byline: str) -> List[str]:
    """Extract just the author names as strings."""
    return [a.name for a in parse_byline(byline) if a.is_person]
