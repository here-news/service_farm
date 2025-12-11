import React, { useMemo } from 'react'
import EventEntityLink, { EventEntityData } from './EventEntityLink'
import EventClaimLink, { EventClaimData } from './EventClaimLink'

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

    // Combined regex for:
    // - EntityName [en_id] - name followed by entity ID in brackets (current format)
    //   Entity names are typically 1-4 words before [en_id], not containing brackets
    // - [[EntityName|en_id]] or [[EntityName]] - wiki-style links
    // - {{claim:cl_id}} or {{cite:cl_id}} - mustache-style claims
    // - [cl_xxxxxxxx] standalone claim references
    // Order matters: entity name pattern must come first to capture the name
    const regex = /((?:[A-Z][a-zA-Z'-]*(?:\s+(?:of\s+)?[A-Z][a-zA-Z'-]*){0,5}))\s*\[(en_[a-zA-Z0-9]+)\]|\[\[([^\]|]+)(?:\|([^\]]+))?\]\]|\{\{(?:claim|cite):([^}]+)\}\}|\[(cl_[a-zA-Z0-9]+)\]/g
    let match

    while ((match = regex.exec(text)) !== null) {
      // Add text before match
      if (match.index > lastIndex) {
        parts.push({
          type: 'text',
          content: text.substring(lastIndex, match.index)
        })
      }

      if (match[1] && match[2]) {
        // Entity match: "Name [en_id]" format
        const entityName = match[1].trim()
        const entityId = match[2]

        parts.push({
          type: 'entity',
          content: entityName,
          entityName,
          entityId
        })
      } else if (match[3]) {
        // Entity match: [[Name|id]] or [[Name]]
        const entityName = match[3]
        const entityId = match[4] || undefined

        parts.push({
          type: 'entity',
          content: entityName,
          entityName,
          entityId
        })
      } else if (match[5]) {
        // Claim match: {{claim:id}} or {{cite:id}}
        parts.push({
          type: 'claim',
          content: match[0],
          claimId: match[5]
        })
      } else if (match[6]) {
        // Standalone claim reference: [cl_xxxxxxxx]
        parts.push({
          type: 'claim',
          content: match[0],
          claimId: match[6]
        })
      }

      lastIndex = regex.lastIndex
    }

    // Add remaining text
    if (lastIndex < text.length) {
      parts.push({
        type: 'text',
        content: text.substring(lastIndex)
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
          <h2 key={blockIndex} className="text-2xl font-bold text-slate-800 mt-8 mb-4">
            {heading.content}
          </h2>
        )
      }
      if (heading.level === 3) {
        return (
          <h3 key={blockIndex} className="text-xl font-bold text-slate-700 mt-6 mb-3">
            {heading.content}
          </h3>
        )
      }
    }

    // Check if this is a bullet point
    if (parts.length > 0 && parts[0].type === 'bullet') {
      return (
        <li key={blockIndex} className="text-slate-700 pl-2 mb-2">
          {parts.slice(1).map((part, i) => renderPart(part, i))}
        </li>
      )
    }

    // Regular paragraph
    return (
      <p key={blockIndex} className="text-slate-700 leading-relaxed mb-4">
        {parts.map((part, i) => renderPart(part, i))}
      </p>
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
