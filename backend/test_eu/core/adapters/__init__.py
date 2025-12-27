"""
Domain Adapters
================

Optional adapters for domain-specific enhancements.

The kernel is domain-agnostic. Adapters provide:
- Domain-specific hint extraction
- Custom prompt templates
- Specialized prose generation
"""

from .news import NewsHintExtractor, NEWS_PROMPT_TEMPLATE

__all__ = ['NewsHintExtractor', 'NEWS_PROMPT_TEMPLATE']
