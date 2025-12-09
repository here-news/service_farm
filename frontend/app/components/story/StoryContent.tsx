import React, { useState, useEffect } from 'react'
import EntityLink, { EntityMetadata } from './EntityLink'

interface StoryContentProps {
  content: string
  entities?: {
    people?: Array<{ name: string; canonical_id?: string; wikidata_qid?: string; wikidata_thumbnail?: string; wikidata_description?: string }>
    organizations?: Array<{ name: string; canonical_id?: string; wikidata_qid?: string; wikidata_thumbnail?: string; wikidata_description?: string }>
    locations?: Array<{ name: string; canonical_id?: string; wikidata_qid?: string; wikidata_thumbnail?: string; wikidata_description?: string }>
  }
}

interface ParsedPart {
  type: 'text' | 'entity' | 'citation'
  content: string
  entityName?: string
  entityId?: string
  citationId?: string
}

function StoryContent({ content, entities }: StoryContentProps) {
  const [entitiesMetadata, setEntitiesMetadata] = useState<Record<string, EntityMetadata>>({})

  // Build entities metadata map from story entities
  useEffect(() => {
    if (!entities) return

    const metadata: Record<string, EntityMetadata> = {}

    const addEntity = (entity: any, type: string) => {
      const id = entity.canonical_id || entity.name
      metadata[id] = {
        name: entity.name,
        canonical_id: entity.canonical_id,
        qid: entity.wikidata_qid,
        wikidata_qid: entity.wikidata_qid,
        description: entity.description,
        wikidata_description: entity.wikidata_description,
        wikidata_thumbnail: entity.wikidata_thumbnail,
        entity_type: type,
        claim_count: entity.claim_count
      }
      // Also index by name for easy lookup
      metadata[entity.name] = metadata[id]
    }

    entities.people?.forEach(e => addEntity(e, 'person'))
    entities.organizations?.forEach(e => addEntity(e, 'organization'))
    entities.locations?.forEach(e => addEntity(e, 'location'))

    setEntitiesMetadata(metadata)
  }, [entities])

  // Parse content into parts (text, entities, citations)
  const parseParagraph = (text: string): ParsedPart[] => {
    const parts: ParsedPart[] = []
    let lastIndex = 0

    // Combined regex for entities and citations
    const regex = /\[\[([^\]|]+)(?:\|([^\]]+))?\]\]|\{\{cite:([^}]+)\}\}/g
    let match

    while ((match = regex.exec(text)) !== null) {
      // Add text before match
      if (match.index > lastIndex) {
        parts.push({
          type: 'text',
          content: text.substring(lastIndex, match.index)
        })
      }

      // Add entity or citation
      if (match[1]) {
        // Entity match
        parts.push({
          type: 'entity',
          content: match[0],
          entityName: match[1],
          entityId: match[2] || match[1]
        })
      } else if (match[3]) {
        // Citation match
        parts.push({
          type: 'citation',
          content: match[0],
          citationId: match[3]
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

  // Render a single paragraph
  const renderParagraph = (text: string, index: number) => {
    const parts = parseParagraph(text)

    return (
      <p key={index} className="mb-4 leading-relaxed text-slate-700">
        {parts.map((part, i) => {
          if (part.type === 'text') {
            return <span key={i}>{part.content}</span>
          } else if (part.type === 'entity') {
            const metadata = entitiesMetadata[part.entityId!] || entitiesMetadata[part.entityName!]
            return (
              <EntityLink
                key={i}
                name={part.entityName!}
                canonicalId={part.entityId}
                metadata={metadata}
              />
            )
          } else if (part.type === 'citation') {
            return (
              <sup
                key={i}
                className="citation-marker"
                data-citation-id={part.citationId}
                title={`Citation: ${part.citationId}`}
              >
                [cite]
              </sup>
            )
          }
          return null
        })}
      </p>
    )
  }

  // Split content into paragraphs and render
  const paragraphs = content.split('\n\n').filter(p => p.trim())

  return (
    <div className="prose max-w-none">
      <style dangerouslySetInnerHTML={{
        __html: `
          .citation-marker {
            color: #6366f1;
            font-size: 0.75em;
            font-weight: 600;
            margin-left: 0.1em;
            cursor: pointer;
          }
          .citation-marker:hover {
            text-decoration: underline;
          }
        `
      }} />
      {paragraphs.map((paragraph, index) => renderParagraph(paragraph, index))}
    </div>
  )
}

export default StoryContent
