"""
Datetime utility functions for handling Neo4j DateTime objects
"""
from datetime import datetime
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def neo4j_datetime_to_python(neo4j_dt) -> Optional[datetime]:
    """
    Convert Neo4j DateTime to Python datetime

    Handles multiple cases:
    - None -> None
    - Neo4j DateTime with to_native() -> Python datetime
    - Already Python datetime -> return as-is
    - String ISO format -> parse to datetime
    - Other -> None with warning

    Args:
        neo4j_dt: Neo4j DateTime, Python datetime, string, or None

    Returns:
        Python datetime or None
    """
    if neo4j_dt is None:
        return None

    # Already Python datetime
    if isinstance(neo4j_dt, datetime):
        return neo4j_dt

    # Neo4j DateTime with to_native() method
    if hasattr(neo4j_dt, 'to_native'):
        try:
            return neo4j_dt.to_native()
        except Exception as e:
            logger.warning(f"Failed to call to_native() on {type(neo4j_dt)}: {e}")

    # Neo4j DateTime with iso_format() method
    if hasattr(neo4j_dt, 'iso_format'):
        try:
            iso_str = neo4j_dt.iso_format()
            return datetime.fromisoformat(iso_str.replace('Z', '+00:00'))
        except Exception as e:
            logger.warning(f"Failed to parse iso_format from {type(neo4j_dt)}: {e}")

    # String (try parsing ISO format)
    if isinstance(neo4j_dt, str):
        try:
            return datetime.fromisoformat(neo4j_dt.replace('Z', '+00:00'))
        except Exception as e:
            logger.warning(f"Failed to parse datetime string '{neo4j_dt}': {e}")

    # Unknown type
    logger.warning(f"Cannot convert {type(neo4j_dt)} to Python datetime: {neo4j_dt}")
    return None
