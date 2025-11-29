/**
 * Content Script - Extracts Open Graph metadata from loaded page
 *
 * This script runs on the actual webpage and extracts metadata
 * using the DOM (just like social media bots would).
 */

(function() {
  'use strict';

  console.log('🔍 HERE News - Content script loaded');

  /**
   * Extract Open Graph metadata AND full article text from current page
   */
  function extractMetadata() {
    const metadata = {
      title: null,
      description: null,
      thumbnail: null,
      author: null,
      site_name: null,
      published_date: null,
      canonical_url: null,
      url: window.location.href,
      extracted_at: new Date().toISOString(),
      content_text: null,
      word_count: 0
    };

    // Extract Open Graph title
    const ogTitle = document.querySelector('meta[property="og:title"]');
    if (ogTitle && ogTitle.content) {
      metadata.title = ogTitle.content;
    } else {
      // Fallback to <title> tag
      metadata.title = document.title || null;
    }

    // Extract Open Graph description
    const ogDescription = document.querySelector('meta[property="og:description"]');
    if (ogDescription && ogDescription.content) {
      metadata.description = ogDescription.content;
    } else {
      // Fallback to meta description
      const metaDesc = document.querySelector('meta[name="description"]');
      if (metaDesc && metaDesc.content) {
        metadata.description = metaDesc.content;
      }
    }

    // Extract Open Graph image (thumbnail)
    const ogImage = document.querySelector('meta[property="og:image"]');
    if (ogImage && ogImage.content) {
      metadata.thumbnail = makeAbsoluteUrl(ogImage.content);
    } else {
      // Fallback to Twitter card image
      const twitterImage = document.querySelector('meta[name="twitter:image"]');
      if (twitterImage && twitterImage.content) {
        metadata.thumbnail = makeAbsoluteUrl(twitterImage.content);
      }
    }

    // Extract site name
    const ogSiteName = document.querySelector('meta[property="og:site_name"]');
    if (ogSiteName && ogSiteName.content) {
      metadata.site_name = ogSiteName.content;
    } else {
      // Fallback to domain
      metadata.site_name = window.location.hostname.replace('www.', '');
    }

    // Extract canonical URL
    const ogUrl = document.querySelector('meta[property="og:url"]');
    if (ogUrl && ogUrl.content) {
      metadata.canonical_url = ogUrl.content;
    } else {
      const canonical = document.querySelector('link[rel="canonical"]');
      if (canonical && canonical.href) {
        metadata.canonical_url = canonical.href;
      } else {
        metadata.canonical_url = window.location.href;
      }
    }

    // Extract author
    const authorMeta = document.querySelector('meta[name="author"]');
    if (authorMeta && authorMeta.content) {
      metadata.author = authorMeta.content;
    } else {
      const articleAuthor = document.querySelector('meta[property="article:author"]');
      if (articleAuthor && articleAuthor.content) {
        metadata.author = articleAuthor.content;
      }
    }

    // Extract publication date
    const publishedTime = document.querySelector('meta[property="article:published_time"]');
    if (publishedTime && publishedTime.content) {
      metadata.published_date = publishedTime.content;
    } else {
      const ogPublished = document.querySelector('meta[property="og:published_time"]');
      if (ogPublished && ogPublished.content) {
        metadata.published_date = ogPublished.content;
      }
    }

    // Also check for Twitter Card metadata as fallback
    if (!metadata.title) {
      const twitterTitle = document.querySelector('meta[name="twitter:title"]');
      if (twitterTitle && twitterTitle.content) {
        metadata.title = twitterTitle.content;
      }
    }

    if (!metadata.description) {
      const twitterDesc = document.querySelector('meta[name="twitter:description"]');
      if (twitterDesc && twitterDesc.content) {
        metadata.description = twitterDesc.content;
      }
    }

    // Extract full article text using Mozilla Readability
    try {
      // Clone the document to avoid modifying the original
      const documentClone = document.cloneNode(true);

      // Create Readability instance and parse
      const reader = new Readability(documentClone);
      const article = reader.parse();

      if (article && article.textContent) {
        // Extract clean article text
        metadata.content_text = article.textContent.trim();

        // Calculate word count (split by whitespace)
        const words = metadata.content_text.split(/\s+/).filter(word => word.length > 0);
        metadata.word_count = words.length;

        console.log(`📝 Extracted article: ${metadata.word_count} words`);

        // If Readability found a better title, use it
        if (article.title && !metadata.title) {
          metadata.title = article.title;
        }

        // If Readability found author and we didn't, use it
        if (article.byline && !metadata.author) {
          metadata.author = article.byline;
        }
      } else {
        console.log('⚠️  Readability could not extract article content');
        // Fallback: try to get body text
        const bodyText = document.body ? document.body.innerText : '';
        if (bodyText) {
          metadata.content_text = bodyText.trim();
          const words = metadata.content_text.split(/\s+/).filter(word => word.length > 0);
          metadata.word_count = words.length;
          console.log(`📝 Fallback extraction: ${metadata.word_count} words from body`);
        }
      }
    } catch (error) {
      console.error('❌ Readability extraction error:', error);
      // Fallback to body text if Readability fails
      try {
        const bodyText = document.body ? document.body.innerText : '';
        if (bodyText) {
          metadata.content_text = bodyText.trim();
          const words = metadata.content_text.split(/\s+/).filter(word => word.length > 0);
          metadata.word_count = words.length;
          console.log(`📝 Emergency fallback: ${metadata.word_count} words`);
        }
      } catch (fallbackError) {
        console.error('❌ Fallback extraction also failed:', fallbackError);
      }
    }

    return metadata;
  }

  /**
   * Convert relative URLs to absolute URLs
   */
  function makeAbsoluteUrl(url) {
    if (!url) return null;
    if (url.startsWith('http://') || url.startsWith('https://')) {
      return url;
    }
    // Handle protocol-relative URLs
    if (url.startsWith('//')) {
      return window.location.protocol + url;
    }
    // Handle absolute paths
    if (url.startsWith('/')) {
      return window.location.origin + url;
    }
    // Handle relative paths
    return new URL(url, window.location.href).href;
  }

  /**
   * Wait for page to be fully loaded before extracting
   */
  function waitForPageLoad() {
    return new Promise((resolve) => {
      console.log(`📄 Page readyState: ${document.readyState}`);

      if (document.readyState === 'complete') {
        console.log('✅ Page already loaded');
        resolve();
      } else {
        console.log('⏳ Waiting for page load...');

        const onLoad = () => {
          console.log('✅ Page load event fired');
          window.removeEventListener('load', onLoad);
          resolve();
        };

        window.addEventListener('load', onLoad);

        // Also timeout after 15 seconds (increased from 10s)
        setTimeout(() => {
          console.log('⏱️  Page load timeout (15s), proceeding anyway');
          window.removeEventListener('load', onLoad);
          resolve();
        }, 15000);
      }
    });
  }

  /**
   * Main extraction flow
   */
  async function main() {
    const startTime = Date.now();
    console.log(`🚀 Starting extraction for: ${window.location.href}`);

    try {
      // Wait for page to fully load
      await waitForPageLoad();

      // Extract metadata
      const metadata = extractMetadata();

      const elapsedTime = ((Date.now() - startTime) / 1000).toFixed(2);
      console.log(`✅ Metadata extracted in ${elapsedTime}s:`, {
        title: metadata.title?.substring(0, 60),
        word_count: metadata.word_count,
        has_content: !!metadata.content_text
      });

      // Send metadata back to background script
      chrome.runtime.sendMessage({
        type: 'METADATA_EXTRACTED',
        metadata: metadata
      }, (response) => {
        // Check if message was received
        if (chrome.runtime.lastError) {
          console.error('❌ Failed to send metadata:', chrome.runtime.lastError);
        } else {
          console.log('✅ Metadata sent to background script');
        }
      });

    } catch (error) {
      console.error('❌ Metadata extraction error:', error);

      // Send error back to background script
      chrome.runtime.sendMessage({
        type: 'METADATA_EXTRACTION_ERROR',
        error: error.message
      });
    }
  }

  // Run extraction immediately
  main();

})();
