/**
 * Story URL utilities
 * Provides consistent URL generation for stories with SEO-friendly slugs
 */

/**
 * Generate a URL-friendly slug from a story title
 */
export function generateSlug(title: string): string {
    return title
        .toLowerCase()
        .replace(/[^\w\s-]/g, '') // Remove special chars
        .replace(/\s+/g, '-')      // Replace spaces with hyphens
        .replace(/-+/g, '-')       // Replace multiple hyphens with single
        .replace(/^-|-$/g, '')     // Trim hyphens from start/end
        .substring(0, 80);         // Limit length for practical URLs
}

/**
 * Get canonical story URL with SEO-friendly slug
 *
 * @param id - Story ID (required)
 * @param title - Story title (optional, for slug generation)
 *
 * @returns Canonical story URL
 *
 * @example
 * getStoryUrl('abc123', 'Israel Gaza Conflict')
 * // => '/story/abc123/israel-gaza-conflict'
 */
export function getStoryUrl(id: string, title?: string): string {
    let url = `/story/${id}`;

    if (title) {
        const slug = generateSlug(title);
        if (slug) {
            url += `/${slug}`;
        }
    }

    return url;
}

/**
 * Parse story URL to extract ID
 * Supports both /story/:id and /story/:id/:slug formats
 *
 * @param pathname - URL pathname (e.g., '/story/abc123/slug')
 *
 * @returns Story ID or null
 */
export function parseStoryUrl(pathname: string): string | null {
    // Pattern: /story/:id or /story/:id/:slug
    const match = pathname.match(/^\/story\/([^\/]+)(?:\/[^\/]+)?$/);

    if (!match) {
        return null;
    }

    return match[1];
}
