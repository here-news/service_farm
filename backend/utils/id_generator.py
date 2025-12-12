"""
Short prefixed ID generator for HereNews entities.

Format: {prefix}_{base36_random}
- pg_xxxxxxxx  - page
- cl_xxxxxxxx  - claim
- en_xxxxxxxx  - entity
- ev_xxxxxxxx  - event
- sr_xxxxxxxx  - source
- cm_xxxxxxxx  - comment
- cs_xxxxxxxx  - chat_session

8 chars base36 = 36^8 = 2.8 trillion unique IDs per type
Total length: 11 chars (2-3 prefix + 8 random)
"""
import secrets
import re
from typing import Optional

# Base36 alphabet (lowercase letters + digits)
ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyz"
BASE = len(ALPHABET)  # 36

# Valid prefixes
PREFIXES = {
    'page': 'pg',
    'claim': 'cl',
    'entity': 'en',
    'event': 'ev',
    'source': 'sr',
    'comment': 'cm',
    'chat_session': 'cs',
    'thought': 'th',
}

# Reverse mapping for validation
PREFIX_TO_TYPE = {v: k for k, v in PREFIXES.items()}

# Regex for validation
ID_PATTERN = re.compile(r'^(pg|cl|en|ev|sr|cm|cs|th)_[0-9a-z]{8}$')


def _random_base36(length: int = 8) -> str:
    """Generate random base36 string"""
    result = []
    for _ in range(length):
        result.append(ALPHABET[secrets.randbelow(BASE)])
    return ''.join(result)


def generate_id(entity_type: str) -> str:
    """
    Generate a new short ID for the given entity type.

    Args:
        entity_type: One of 'page', 'claim', 'entity', 'event', 'source'

    Returns:
        Short ID like 'en_x5b8r2yj'

    Raises:
        ValueError: If entity_type is invalid
    """
    if entity_type not in PREFIXES:
        raise ValueError(f"Invalid entity type: {entity_type}. "
                        f"Must be one of: {list(PREFIXES.keys())}")

    prefix = PREFIXES[entity_type]
    random_part = _random_base36(8)
    return f"{prefix}_{random_part}"


def validate_id(id_str: str) -> bool:
    """
    Check if a string is a valid short ID.

    Args:
        id_str: String to validate

    Returns:
        True if valid, False otherwise
    """
    if not id_str or not isinstance(id_str, str):
        return False
    return bool(ID_PATTERN.match(id_str))


def get_id_type(id_str: str) -> Optional[str]:
    """
    Extract the entity type from an ID.

    Args:
        id_str: A short ID like 'en_x5b8r2yj'

    Returns:
        Entity type ('page', 'claim', etc.) or None if invalid
    """
    if not validate_id(id_str):
        return None
    prefix = id_str[:2]
    return PREFIX_TO_TYPE.get(prefix)


def is_uuid(id_str: str) -> bool:
    """
    Check if a string looks like a UUID (for migration compatibility).

    Args:
        id_str: String to check

    Returns:
        True if it looks like a UUID
    """
    if not id_str or not isinstance(id_str, str):
        return False
    # UUID format: 8-4-4-4-12 hex chars
    uuid_pattern = re.compile(
        r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
        re.IGNORECASE
    )
    return bool(uuid_pattern.match(id_str))


def uuid_to_short_id(uuid_str: str, entity_type: str) -> str:
    """
    Convert a UUID to a deterministic short ID.

    This creates a reproducible mapping from UUID to short ID,
    useful for migration. Uses first 8 chars of UUID hex (no dashes)
    converted to base36.

    Args:
        uuid_str: UUID string like 'df21cbf9-6ba2-460f-9c91-d28dbd4b2037'
        entity_type: One of 'page', 'claim', 'entity', 'event', 'source'

    Returns:
        Short ID like 'en_df21cbf9'
    """
    if entity_type not in PREFIXES:
        raise ValueError(f"Invalid entity type: {entity_type}")

    # Remove dashes and take first 8 hex chars
    hex_part = uuid_str.replace('-', '')[:8].lower()

    # Convert hex to int, then to base36
    num = int(hex_part, 16)

    # Convert to base36
    result = []
    while num > 0:
        result.append(ALPHABET[num % BASE])
        num //= BASE

    # Pad to 8 chars if needed
    base36 = ''.join(reversed(result)).zfill(8)[-8:]

    prefix = PREFIXES[entity_type]
    return f"{prefix}_{base36}"


# Convenience functions for each type
def generate_page_id() -> str:
    """Generate a new page ID"""
    return generate_id('page')


def generate_claim_id() -> str:
    """Generate a new claim ID"""
    return generate_id('claim')


def generate_entity_id() -> str:
    """Generate a new entity ID"""
    return generate_id('entity')


def generate_event_id() -> str:
    """Generate a new event ID"""
    return generate_id('event')


def generate_source_id() -> str:
    """Generate a new source ID"""
    return generate_id('source')


def generate_comment_id() -> str:
    """Generate a new comment ID"""
    return generate_id('comment')


def generate_chat_session_id() -> str:
    """Generate a new chat session ID"""
    return generate_id('chat_session')


# For backwards compatibility during migration
def ensure_short_id(id_str: str, entity_type: str) -> str:
    """
    Ensure an ID is in short format. If it's a UUID, convert it.

    Args:
        id_str: Either a short ID or UUID
        entity_type: Entity type for conversion

    Returns:
        Short ID
    """
    if validate_id(id_str):
        return id_str
    if is_uuid(id_str):
        return uuid_to_short_id(id_str, entity_type)
    raise ValueError(f"Invalid ID format: {id_str}")
