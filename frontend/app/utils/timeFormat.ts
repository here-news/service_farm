/**
 * Time formatting utilities for displaying timestamps
 * Based on classic_app's formatDateTime function
 */

export interface TimeFormat {
  absolute: string
  relative: string
  reliable: boolean
  fullDateTime: string
}

export function formatTime(timestamp?: string): TimeFormat {
  if (!timestamp) {
    return {
      absolute: 'Date unknown',
      relative: 'unknown',
      reliable: false,
      fullDateTime: ''
    }
  }

  try {
    const then = new Date(timestamp)

    // Check if date is valid
    if (isNaN(then.getTime())) {
      return {
        absolute: 'Date unknown',
        relative: 'unknown',
        reliable: false,
        fullDateTime: ''
      }
    }

    const now = new Date()
    const diffMs = now.getTime() - then.getTime()
    const diffMins = Math.floor(diffMs / 60000)
    const diffHours = Math.floor(diffMins / 60)
    const diffDays = Math.floor(diffHours / 24)

    // Format absolute date/time
    const today = new Date()
    today.setHours(0, 0, 0, 0)
    const thenDate = new Date(then)
    thenDate.setHours(0, 0, 0, 0)
    const daysDiff = Math.floor((today.getTime() - thenDate.getTime()) / (1000 * 60 * 60 * 24))

    let absoluteText = ''
    let reliable = true

    // Check if date is in the future (likely extraction error)
    if (diffMs < 0) {
      absoluteText = then.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })
      reliable = false
    }
    // Check if date is too old (likely extraction error - over 2 years)
    else if (diffDays > 730) {
      absoluteText = then.toLocaleDateString('en-US', { month: 'short', year: 'numeric' })
      reliable = false
    }
    // Today
    else if (daysDiff === 0) {
      absoluteText = then.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit', hour12: true })
    }
    // Yesterday
    else if (daysDiff === 1) {
      absoluteText = 'Yesterday ' + then.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit', hour12: true })
    }
    // Within last 7 days - show day name
    else if (daysDiff < 7) {
      absoluteText = then.toLocaleDateString('en-US', { weekday: 'short', hour: 'numeric', minute: '2-digit', hour12: true })
    }
    // This year - show month and day
    else if (then.getFullYear() === now.getFullYear()) {
      absoluteText = then.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
    }
    // Previous years - show month, day, year
    else {
      absoluteText = then.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })
    }

    // Format relative time
    let relativeText = ''
    if (diffMs < 0) {
      relativeText = 'in the future'
    } else if (diffDays > 365) {
      const years = Math.floor(diffDays / 365)
      relativeText = `${years} year${years > 1 ? 's' : ''} ago`
    } else if (diffDays > 30) {
      const months = Math.floor(diffDays / 30)
      relativeText = `${months} month${months > 1 ? 's' : ''} ago`
    } else if (diffDays > 0) {
      relativeText = `${diffDays} day${diffDays > 1 ? 's' : ''} ago`
    } else if (diffHours > 0) {
      relativeText = `${diffHours} hour${diffHours > 1 ? 's' : ''} ago`
    } else if (diffMins > 0) {
      relativeText = `${diffMins} minute${diffMins > 1 ? 's' : ''} ago`
    } else {
      relativeText = 'just now'
    }

    // Full datetime for tooltip
    const fullDateTime = then.toLocaleString('en-US', {
      weekday: 'long',
      year: 'numeric',
      month: 'long',
      day: 'numeric',
      hour: 'numeric',
      minute: '2-digit',
      hour12: true
    })

    return {
      absolute: absoluteText,
      relative: relativeText,
      reliable,
      fullDateTime
    }
  } catch {
    return {
      absolute: 'Date unknown',
      relative: 'unknown',
      reliable: false,
      fullDateTime: ''
    }
  }
}
