/**
 * Extract URLs from text
 */
export function extractUrls(text: string): string[] {
  // URL regex pattern - matches http/https URLs
  const urlRegex = /(https?:\/\/[^\s]+)/gi
  const matches = text.match(urlRegex)

  if (!matches) return []

  // Clean up URLs (remove trailing punctuation)
  return matches.map(url => {
    // Remove trailing punctuation like periods, commas, etc.
    return url.replace(/[.,;:!?)]$/, '')
  })
}

/**
 * Check if text contains URLs
 */
export function hasUrls(text: string): boolean {
  return extractUrls(text).length > 0
}
