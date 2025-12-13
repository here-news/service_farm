import React, { useMemo, createContext, useContext } from 'react'
import EventEntityLink, { EventEntityData } from './EventEntityLink'
import EventClaimLink, { EventClaimData } from './EventClaimLink'
import { useInView } from '../../hooks/useInView'

// Context to pass visibility state to child components
export const ParagraphVisibilityContext = createContext<boolean>(true)

interface Entity {
  id: string
  canonical_name: string
  entity_type: string
  mention_count?: number
  wikidata_qid?: string
  wikidata_description?: string
  image_url?: string
}

interface Claim {
  id: string
  text: string
  event_time?: string
  confidence?: number
}

interface EventNarrativeContentProps {
  content: string
  entities: Entity[]
  claims: Claim[]
}

interface ParsedPart {
  type: 'text' | 'entity' | 'claim' | 'heading' | 'bullet'
  content: string
  entityId?: string
  entityName?: string
  claimId?: string
  level?: number // for headings
}

// Track which entities have been mentioned
const mentionedEntities = new Set<string>()

// Wrapper component for paragraphs with scroll-based visibility
interface VisibilityWrapperProps {
  children: React.ReactNode
  className?: string
  as?: 'p' | 'h2' | 'h3' | 'li' | 'ul'
}

function VisibilityWrapper({ children, className, as: Component = 'p' }: VisibilityWrapperProps) {
  // Reveal when element passes middle of screen, stay revealed permanently
  const [ref, isRevealed] = useInView<HTMLElement>({
    triggerOnce: true, // Once revealed, stays revealed
  })

  return (
    <ParagraphVisibilityContext.Provider value={isRevealed}>
      <Component
        ref={ref as React.RefObject<any>}
        className={className}
      >
        {children}
      </Component>
    </ParagraphVisibilityContext.Provider>
  )
}

function EventNarrativeContent({ content, entities, claims }: EventNarrativeContentProps) {
  // Reset mentions for each render
  useMemo(() => {
    mentionedEntities.clear()
  }, [content])

  // Build entity lookup map
  const entityMap = useMemo(() => {
    const map = new Map<string, Entity>()
    entities.forEach(e => {
      map.set(e.id, e)
      // Also map by canonical name (lowercased) for matching
      map.set(e.canonical_name.toLowerCase(), e)
    })
    return map
  }, [entities])

  // Build claim lookup map
  const claimMap = useMemo(() => {
    const map = new Map<string, Claim>()
    claims.forEach(c => {
      map.set(c.id, c)
    })
    return map
  }, [claims])

  // Parse content into structured parts
  // Supports:
  // - [[EntityName|en_id]] or [[EntityName]] for entities
  // - {{claim:cl_id}} or [cl_id] for claims
  // - ## Heading for h2, ### Heading for h3
  // - - Bullet point items
  const parseContent = (text: string): ParsedPart[][] => {
    const lines = text.split('\n')
    const paragraphs: ParsedPart[][] = []
    let currentParagraph: ParsedPart[] = []

    for (const line of lines) {
      const trimmed = line.trim()

      // Empty line = end of paragraph
      if (!trimmed) {
        if (currentParagraph.length > 0) {
          paragraphs.push(currentParagraph)
          currentParagraph = []
        }
        continue
      }

      // Heading detection
      const h3Match = trimmed.match(/^###\s+(.+)$/)
      if (h3Match) {
        if (currentParagraph.length > 0) {
          paragraphs.push(currentParagraph)
          currentParagraph = []
        }
        paragraphs.push([{ type: 'heading', content: h3Match[1], level: 3 }])
        continue
      }

      const h2Match = trimmed.match(/^##\s+(.+)$/)
      if (h2Match) {
        if (currentParagraph.length > 0) {
          paragraphs.push(currentParagraph)
          currentParagraph = []
        }
        paragraphs.push([{ type: 'heading', content: h2Match[1], level: 2 }])
        continue
      }

      // Bullet point detection
      const bulletMatch = trimmed.match(/^[-*]\s+(.+)$/)
      if (bulletMatch) {
        const bulletParts = parseInlineContent(bulletMatch[1])
        paragraphs.push([{ type: 'bullet', content: '', }, ...bulletParts])
        continue
      }

      // Regular content line
      const parts = parseInlineContent(trimmed)
      currentParagraph.push(...parts)

      // Add space between lines in same paragraph
      if (currentParagraph.length > 0) {
        currentParagraph.push({ type: 'text', content: ' ' })
      }
    }

    // Don't forget last paragraph
    if (currentParagraph.length > 0) {
      paragraphs.push(currentParagraph)
    }

    return paragraphs
  }

  // Parse inline entities and claims
  const parseInlineContent = (text: string): ParsedPart[] => {
    const parts: ParsedPart[] = []
    let lastIndex = 0

    // Two-pass parsing to handle adjacent entities correctly:
    // Pass 1: Match entity IDs [en_xxx] and claim IDs [cl_xxx]
    // Pass 2: Link entity names to their IDs

    // First, find all bracketed references
    // Matches: [en_xxxxxxxx] or [cl_xxxxxxxx] or [[Name|id]] or {{claim:id}}
    const bracketRegex = /\[(en_[a-zA-Z0-9]+)\]|\[(cl_[a-zA-Z0-9]+)\]|\[\[([^\]|]+)(?:\|([^\]]+))?\]\]|\{\{(?:claim|cite):([^}]+)\}\}/g

    // Collect all matches first to build entity name mapping
    const matches: Array<{
      index: number
      length: number
      type: 'entity' | 'claim' | 'entity_wiki'
      id?: string
      name?: string
    }> = []

    let match
    while ((match = bracketRegex.exec(text)) !== null) {
      if (match[1]) {
        // [en_xxx] entity reference
        matches.push({
          index: match.index,
          length: match[0].length,
          type: 'entity',
          id: match[1]
        })
      } else if (match[2]) {
        // [cl_xxx] claim reference
        matches.push({
          index: match.index,
          length: match[0].length,
          type: 'claim',
          id: match[2]
        })
      } else if (match[3]) {
        // [[Name|id]] or [[Name]] wiki-style
        matches.push({
          index: match.index,
          length: match[0].length,
          type: 'entity_wiki',
          name: match[3],
          id: match[4]
        })
      } else if (match[5]) {
        // {{claim:id}} mustache-style
        matches.push({
          index: match.index,
          length: match[0].length,
          type: 'claim',
          id: match[5]
        })
      }
    }

    // Now process text, looking backwards from [en_xxx] to find entity name
    let currentPos = 0

    for (const m of matches) {
      // Add text before this match
      if (m.index > currentPos) {
        const beforeText = text.substring(currentPos, m.index)

        if (m.type === 'entity') {
          // Look for entity name immediately before [en_xxx]
          // Only match on the SAME LINE to avoid picking up text from previous lines
          // Split by newline and only look at the last line
          const lines = beforeText.split('\n')
          const lastLine = lines[lines.length - 1]
          const textBeforeLastLine = lines.length > 1 ? lines.slice(0, -1).join('\n') + '\n' : ''

          // Pattern: Entity names are capitalized words, possibly with spaces
          // Match greedily but only within the last line
          const nameMatch = lastLine.match(/([A-Z][a-zA-Z''\-]*(?:\s+(?:of\s+)?[A-Z][a-zA-Z''\-]*){0,5})\s*$/)

          if (nameMatch) {
            // Add text before the entity name (including previous lines)
            const textBeforeName = textBeforeLastLine + lastLine.substring(0, lastLine.length - nameMatch[0].length)
            if (textBeforeName) {
              parts.push({ type: 'text', content: textBeforeName })
            }
            // Add entity with its name
            parts.push({
              type: 'entity',
              content: nameMatch[1].trim(),
              entityName: nameMatch[1].trim(),
              entityId: m.id
            })
          } else {
            // No name found, add all text before and lookup entity name from map
            if (beforeText) {
              parts.push({ type: 'text', content: beforeText })
            }
            // Lookup entity name from entityMap
            const entity = entityMap.get(m.id || '')
            parts.push({
              type: 'entity',
              content: entity?.canonical_name || m.id || 'Unknown',
              entityName: entity?.canonical_name,
              entityId: m.id
            })
          }
        } else {
          // For claims and wiki entities, just add the text before
          if (beforeText) {
            parts.push({ type: 'text', content: beforeText })
          }

          if (m.type === 'claim') {
            parts.push({
              type: 'claim',
              content: m.id || '',
              claimId: m.id
            })
          } else if (m.type === 'entity_wiki') {
            parts.push({
              type: 'entity',
              content: m.name || '',
              entityName: m.name,
              entityId: m.id
            })
          }
        }
      } else {
        // Match starts at currentPos (adjacent to previous match)
        if (m.type === 'entity') {
          const entity = entityMap.get(m.id || '')
          parts.push({
            type: 'entity',
            content: entity?.canonical_name || m.id || 'Unknown',
            entityName: entity?.canonical_name,
            entityId: m.id
          })
        } else if (m.type === 'claim') {
          parts.push({
            type: 'claim',
            content: m.id || '',
            claimId: m.id
          })
        } else if (m.type === 'entity_wiki') {
          parts.push({
            type: 'entity',
            content: m.name || '',
            entityName: m.name,
            entityId: m.id
          })
        }
      }

      currentPos = m.index + m.length
    }

    // Add remaining text
    if (currentPos < text.length) {
      parts.push({
        type: 'text',
        content: text.substring(currentPos)
      })
    }

    return parts
  }

  // Render a single part
  const renderPart = (part: ParsedPart, index: number) => {
    if (part.type === 'text') {
      // Handle **bold** and *italic* markdown
      const formatted = part.content
        .replace(/\*\*([^*]+)\*\*/g, '<strong class="text-slate-900 font-semibold">$1</strong>')
        .replace(/\*([^*]+)\*/g, '<em>$1</em>')

      return <span key={index} dangerouslySetInnerHTML={{ __html: formatted }} />
    }

    if (part.type === 'entity') {
      // Try to find entity in our map
      let entity: Entity | undefined
      if (part.entityId) {
        entity = entityMap.get(part.entityId)
      }
      if (!entity && part.entityName) {
        entity = entityMap.get(part.entityName.toLowerCase())
      }

      // Determine if this is the first mention
      const entityKey = entity?.id || part.entityId || part.entityName || ''
      const isFirstMention = !mentionedEntities.has(entityKey)
      if (entityKey) {
        mentionedEntities.add(entityKey)
      }

      return (
        <EventEntityLink
          key={index}
          entityId={entity?.id || part.entityId || ''}
          displayName={part.entityName || entity?.canonical_name || 'Unknown'}
          isFirstMention={isFirstMention}
          eventEntities={entities}
        />
      )
    }

    if (part.type === 'claim') {
      const claim = part.claimId ? claimMap.get(part.claimId) : undefined

      return (
        <EventClaimLink
          key={index}
          claimId={part.claimId || ''}
          preloadedClaim={claim ? {
            id: claim.id,
            text: claim.text,
            event_time: claim.event_time,
            confidence: claim.confidence
          } : undefined}
          eventClaims={claims}
        />
      )
    }

    return null
  }

  // Render a paragraph/block
  const renderBlock = (parts: ParsedPart[], blockIndex: number) => {
    // Check if this is a heading
    if (parts.length === 1 && parts[0].type === 'heading') {
      const heading = parts[0]
      if (heading.level === 2) {
        return (
          <VisibilityWrapper key={blockIndex} as="h2" className="text-2xl font-bold text-slate-800 mt-8 mb-4">
            {heading.content}
          </VisibilityWrapper>
        )
      }
      if (heading.level === 3) {
        return (
          <VisibilityWrapper key={blockIndex} as="h3" className="text-xl font-bold text-slate-700 mt-6 mb-3">
            {heading.content}
          </VisibilityWrapper>
        )
      }
    }

    // Check if this is a bullet point
    if (parts.length > 0 && parts[0].type === 'bullet') {
      return (
        <VisibilityWrapper key={blockIndex} as="li" className="text-slate-700 pl-2 mb-2">
          {parts.slice(1).map((part, i) => renderPart(part, i))}
        </VisibilityWrapper>
      )
    }

    // Regular paragraph
    return (
      <VisibilityWrapper key={blockIndex} as="p" className="text-slate-700 leading-relaxed mb-4">
        {parts.map((part, i) => renderPart(part, i))}
      </VisibilityWrapper>
    )
  }

  const parsedContent = useMemo(() => parseContent(content), [content])

  // Group consecutive bullets into lists
  const renderContent = () => {
    const elements: React.ReactNode[] = []
    let bulletBuffer: ParsedPart[][] = []

    const flushBullets = () => {
      if (bulletBuffer.length > 0) {
        elements.push(
          <ul key={`ul-${elements.length}`} className="list-disc list-inside space-y-1 my-4 pl-4">
            {bulletBuffer.map((parts, i) => renderBlock(parts, i))}
          </ul>
        )
        bulletBuffer = []
      }
    }

    parsedContent.forEach((parts, index) => {
      const isBullet = parts.length > 0 && parts[0].type === 'bullet'

      if (isBullet) {
        bulletBuffer.push(parts)
      } else {
        flushBullets()
        elements.push(renderBlock(parts, index))
      }
    })

    flushBullets()
    return elements
  }

  return (
    <div className="prose prose-slate prose-lg max-w-none">
      {renderContent()}
    </div>
  )
}

export default EventNarrativeContent
