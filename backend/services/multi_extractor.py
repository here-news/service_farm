"""
Multi-method content extractor with fallback chain

Extraction methods (in order):
1. Trafilatura - Fast, clean text extraction
2. Newspaper3k - Good for news articles
3. Readability - Handles complex layouts
4. Playwright - Full browser rendering (last resort)

Goal: ~90% success rate with graceful degradation
"""
import logging
from typing import Optional, Tuple
from dataclasses import dataclass
import trafilatura
from bs4 import BeautifulSoup
from newspaper import Article
from readability import Document

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """Result from content extraction"""
    success: bool
    content: str
    word_count: int
    method_used: str
    error_message: Optional[str] = None
    # Metadata extraction
    title: Optional[str] = None
    description: Optional[str] = None
    authors: Optional[list] = None
    site_name: Optional[str] = None  # og:site_name - publisher name
    publish_date: Optional[str] = None  # Publication date (from newspaper3k)
    # Image extraction (from newspaper3k)
    top_image: Optional[str] = None  # Main article image
    images: Optional[list] = None    # All images found


class MultiMethodExtractor:
    """
    Tries multiple extraction methods with fallback

    Based on Gen1's SmartExtractionOrchestrator but simplified for Gen2
    """

    def __init__(self, min_words: int = 100):
        """
        Initialize extractor

        Args:
            min_words: Minimum word count to consider extraction successful
        """
        self.min_words = min_words

    def _extract_site_name(self, html: str) -> Optional[str]:
        """
        Extract site name from og:site_name meta tag.

        Args:
            html: HTML content

        Returns:
            Site name or None
        """
        try:
            soup = BeautifulSoup(html, 'lxml')

            # Try og:site_name first
            og_site = soup.find('meta', property='og:site_name')
            if og_site and og_site.get('content'):
                return og_site['content'].strip()

            # Try application-name as fallback
            app_name = soup.find('meta', attrs={'name': 'application-name'})
            if app_name and app_name.get('content'):
                return app_name['content'].strip()

            # Try publisher from JSON-LD
            import json
            import re
            jsonld_pattern = r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>'
            jsonld_matches = re.findall(jsonld_pattern, html, re.DOTALL | re.IGNORECASE)

            for jsonld_str in jsonld_matches:
                try:
                    data = json.loads(jsonld_str)
                    if isinstance(data, list):
                        data = data[0] if data else {}
                    if '@graph' in data and isinstance(data['@graph'], list):
                        for item in data['@graph']:
                            if item.get('@type') in ('NewsMediaOrganization', 'Organization', 'WebSite'):
                                if item.get('name'):
                                    return item['name']
                    if data.get('publisher') and isinstance(data['publisher'], dict):
                        if data['publisher'].get('name'):
                            return data['publisher']['name']
                except:
                    continue

            return None
        except Exception as e:
            logger.debug(f"Failed to extract site_name: {e}")
            return None

    async def extract(self, url: str, html: str) -> ExtractionResult:
        """
        Extract content using multiple methods with fallback

        Priority order:
        1. Newspaper3k - Best for news, extracts images
        2. Trafilatura - Fast, clean text
        3. Readability - Complex layouts fallback

        Args:
            url: URL being extracted
            html: HTML content

        Returns:
            ExtractionResult with extracted content and images
        """
        # Extract site_name once (used for all methods)
        site_name = self._extract_site_name(html)

        # Method 1: Newspaper3k (extracts images + best for news)
        result = self._try_newspaper(url, html)
        if result.success:
            result.site_name = site_name
            logger.info(f"✅ Newspaper3k extracted {result.word_count} words, {len(result.images or [])} images from {url}")
            return result

        # Method 2: Trafilatura (fast, clean text fallback)
        result = self._try_trafilatura(html)
        if result.success:
            result.site_name = site_name
            logger.info(f"✅ Trafilatura extracted {result.word_count} words from {url}")
            return result

        # Method 3: Readability (handles complex layouts)
        result = self._try_readability(html)
        if result.success:
            result.site_name = site_name
            logger.info(f"✅ Readability extracted {result.word_count} words from {url}")
            return result

        # All methods failed
        logger.warning(f"❌ All extraction methods failed for {url}")
        return ExtractionResult(
            success=False,
            content="",
            word_count=0,
            method_used="none",
            error_message="All extraction methods failed",
            site_name=site_name  # Still include site_name even on failure
        )

    def _try_trafilatura(self, html: str) -> ExtractionResult:
        """
        Try Trafilatura extraction

        Args:
            html: HTML content

        Returns:
            ExtractionResult
        """
        try:
            extracted = trafilatura.extract(
                html,
                include_comments=False,
                include_tables=False,
                no_fallback=False
            )

            if extracted and len(extracted.strip()) >= self.min_words:
                word_count = len(extracted.split())
                return ExtractionResult(
                    success=True,
                    content=extracted,
                    word_count=word_count,
                    method_used="trafilatura"
                )

            return ExtractionResult(
                success=False,
                content=extracted or "",
                word_count=len(extracted.split()) if extracted else 0,
                method_used="trafilatura",
                error_message=f"Content too short ({len(extracted.split()) if extracted else 0} words)"
            )

        except Exception as e:
            logger.warning(f"Trafilatura failed: {e}")
            return ExtractionResult(
                success=False,
                content="",
                word_count=0,
                method_used="trafilatura",
                error_message=str(e)
            )

    def _try_newspaper(self, url: str, html: str) -> ExtractionResult:
        """
        Try Newspaper3k extraction with metadata and images

        Args:
            url: Article URL
            html: HTML content

        Returns:
            ExtractionResult with metadata and images
        """
        try:
            article = Article(url)
            article.set_html(html)
            article.parse()

            text = article.text
            if text and len(text.strip()) >= self.min_words:
                word_count = len(text.split())

                # Extract metadata
                title = article.title if article.title else None
                description = article.meta_description if article.meta_description else None
                authors = list(article.authors) if article.authors else []

                # Extract publish date (newspaper3k's publish_date attribute)
                publish_date = None
                if article.publish_date:
                    try:
                        # Convert datetime to ISO string
                        publish_date = article.publish_date.isoformat()
                        logger.info(f"📅 Newspaper3k extracted publish_date: {publish_date}")
                    except Exception as e:
                        logger.warning(f"Failed to convert publish_date: {e}")

                # Extract images
                top_image = article.top_image if article.top_image else None
                images = list(article.images) if article.images else []

                return ExtractionResult(
                    success=True,
                    content=text,
                    word_count=word_count,
                    method_used="newspaper3k",
                    title=title,
                    description=description,
                    authors=authors,
                    publish_date=publish_date,
                    top_image=top_image,
                    images=images
                )

            return ExtractionResult(
                success=False,
                content=text or "",
                word_count=len(text.split()) if text else 0,
                method_used="newspaper3k",
                error_message=f"Content too short ({len(text.split()) if text else 0} words)"
            )

        except Exception as e:
            logger.warning(f"Newspaper3k failed: {e}")
            return ExtractionResult(
                success=False,
                content="",
                word_count=0,
                method_used="newspaper3k",
                error_message=str(e)
            )

    def _try_readability(self, html: str) -> ExtractionResult:
        """
        Try Readability extraction

        Args:
            html: HTML content

        Returns:
            ExtractionResult
        """
        try:
            doc = Document(html)
            summary_html = doc.summary()

            # Extract text from HTML
            soup = BeautifulSoup(summary_html, 'lxml')
            text = soup.get_text(separator=' ', strip=True)

            if text and len(text.strip()) >= self.min_words:
                word_count = len(text.split())
                return ExtractionResult(
                    success=True,
                    content=text,
                    word_count=word_count,
                    method_used="readability"
                )

            return ExtractionResult(
                success=False,
                content=text or "",
                word_count=len(text.split()) if text else 0,
                method_used="readability",
                error_message=f"Content too short ({len(text.split()) if text else 0} words)"
            )

        except Exception as e:
            logger.warning(f"Readability failed: {e}")
            return ExtractionResult(
                success=False,
                content="",
                word_count=0,
                method_used="readability",
                error_message=str(e)
            )
