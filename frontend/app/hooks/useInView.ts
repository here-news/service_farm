import { useState, useEffect, useRef, RefObject } from 'react'

interface UseInViewOptions {
  threshold?: number
  rootMargin?: string
  triggerOnce?: boolean
}

/**
 * Hook to detect when an element crosses the middle of the viewport
 * Used for scroll-based reveal of UI elements
 * Once revealed, stays revealed permanently (triggerOnce behavior)
 */
export function useInView<T extends HTMLElement = HTMLElement>(
  options: UseInViewOptions = {}
): [RefObject<T>, boolean] {
  const { threshold = 0.1, triggerOnce = true } = options
  const ref = useRef<T>(null)
  const [isRevealed, setIsRevealed] = useState(false)

  useEffect(() => {
    const element = ref.current
    if (!element) return

    // Already revealed - no need to observe
    if (triggerOnce && isRevealed) return

    // Use rootMargin to create a trigger zone at the middle of the screen
    // Negative top margin means element must pass into the middle area
    // -40% from top means trigger when element is in the middle 60% of screen
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsRevealed(true)
          // Once revealed, disconnect - it stays revealed permanently
          if (triggerOnce) {
            observer.disconnect()
          }
        }
      },
      {
        threshold,
        // Trigger when element enters the middle portion of viewport
        // -40% top margin, 0 sides, -10% bottom - creates a "reveal zone" in the middle
        rootMargin: '-40% 0px -10% 0px',
      }
    )

    observer.observe(element)

    return () => {
      observer.disconnect()
    }
  }, [threshold, triggerOnce, isRevealed])

  return [ref, isRevealed]
}

export default useInView
