import React from 'react'
import { Surface, SurfaceRelation } from '../../types/inquiry'

interface SurfacesTopologyProps {
  surfaces: Surface[]
  onSurfaceClick?: (surface: Surface) => void
}

function getRelationColor(type: SurfaceRelation['type']) {
  switch (type) {
    case 'CONFIRMS': return 'text-green-600 bg-green-100'
    case 'SUPERSEDES': return 'text-indigo-600 bg-indigo-100'
    case 'CONFLICTS': return 'text-red-600 bg-red-100'
    case 'REFINES': return 'text-amber-600 bg-amber-100'
    default: return 'text-slate-600 bg-slate-100'
  }
}

function getRelationArrow(type: SurfaceRelation['type']) {
  switch (type) {
    case 'CONFIRMS': return '✓'
    case 'SUPERSEDES': return '→'
    case 'CONFLICTS': return '⚡'
    case 'REFINES': return '~'
    default: return '?'
  }
}

function SurfacesTopology({ surfaces, onSurfaceClick }: SurfacesTopologyProps) {
  if (!surfaces || surfaces.length === 0) {
    return (
      <div className="bg-white rounded-2xl p-5 shadow-sm border border-slate-100">
        <h3 className="font-semibold text-slate-800 mb-4 flex items-center gap-2">
          <span>🔷</span> Surfaces (Claim Clusters)
        </h3>
        <div className="text-center py-6 text-slate-400">
          <div className="text-3xl mb-2">🔍</div>
          <p className="text-sm">No surfaces formed yet</p>
          <p className="text-xs mt-1">Surfaces emerge as claims cluster by identity</p>
        </div>
      </div>
    )
  }

  // Group surfaces by scope status
  const inScopeSurfaces = surfaces.filter(s => s.in_scope)
  const outOfScopeSurfaces = surfaces.filter(s => !s.in_scope)

  return (
    <div className="bg-white rounded-2xl p-5 shadow-sm border border-slate-100">
      <h3 className="font-semibold text-slate-800 mb-4 flex items-center gap-2">
        <span>🔷</span> Surfaces (Claim Clusters)
        <span className="text-xs bg-slate-100 text-slate-500 px-2 py-0.5 rounded-full">
          {surfaces.length} surfaces
        </span>
      </h3>

      {/* In-scope surfaces */}
      <div className="space-y-3">
        {inScopeSurfaces.map(surface => (
          <SurfaceCard
            key={surface.id}
            surface={surface}
            allSurfaces={surfaces}
            onClick={() => onSurfaceClick?.(surface)}
          />
        ))}
      </div>

      {/* Out-of-scope surfaces (if any) */}
      {outOfScopeSurfaces.length > 0 && (
        <div className="mt-4 pt-4 border-t border-slate-100">
          <div className="text-xs text-slate-500 mb-2 flex items-center gap-1">
            <span>🚫</span> Flagged as Out of Scope
          </div>
          <div className="space-y-2">
            {outOfScopeSurfaces.map(surface => (
              <SurfaceCard
                key={surface.id}
                surface={surface}
                allSurfaces={surfaces}
                onClick={() => onSurfaceClick?.(surface)}
                dimmed
              />
            ))}
          </div>
        </div>
      )}

      {/* Legend */}
      <div className="mt-4 pt-3 border-t border-slate-100">
        <div className="text-xs text-slate-400 mb-2">Relations</div>
        <div className="flex flex-wrap gap-2">
          <span className="text-xs bg-green-100 text-green-700 px-2 py-0.5 rounded">
            ✓ CONFIRMS
          </span>
          <span className="text-xs bg-indigo-100 text-indigo-700 px-2 py-0.5 rounded">
            → SUPERSEDES
          </span>
          <span className="text-xs bg-red-100 text-red-700 px-2 py-0.5 rounded">
            ⚡ CONFLICTS
          </span>
          <span className="text-xs bg-amber-100 text-amber-700 px-2 py-0.5 rounded">
            ~ REFINES
          </span>
        </div>
      </div>
    </div>
  )
}

interface SurfaceCardProps {
  surface: Surface
  allSurfaces: Surface[]
  onClick?: () => void
  dimmed?: boolean
}

function SurfaceCard({ surface, allSurfaces, onClick, dimmed }: SurfaceCardProps) {
  // Find incoming relations (surfaces that point TO this one)
  const incomingRelations: Array<{ from: Surface; relation: SurfaceRelation }> = []
  allSurfaces.forEach(s => {
    s.relations.forEach(rel => {
      if (rel.target === surface.id || rel.target === surface.name) {
        incomingRelations.push({ from: s, relation: rel })
      }
    })
  })

  return (
    <div
      onClick={onClick}
      className={`
        border rounded-lg p-3 transition cursor-pointer
        ${dimmed ? 'border-red-200 bg-red-50 opacity-60' : 'border-slate-200 hover:border-indigo-200 hover:bg-indigo-50'}
      `}
    >
      {/* Header */}
      <div className="flex items-start justify-between mb-2">
        <div>
          <div className="font-medium text-slate-700 text-sm">{surface.name}</div>
          <div className="text-xs text-slate-500">
            {surface.claim_count} claims · {surface.sources.join(', ')}
          </div>
        </div>
        {surface.canonical_value !== undefined && (
          <div className="text-right">
            <div className="font-mono text-sm font-semibold text-indigo-600">
              {typeof surface.canonical_value === 'number'
                ? surface.canonical_value.toLocaleString()
                : String(surface.canonical_value)}
            </div>
            {surface.entropy !== undefined && (
              <div className="text-xs text-slate-400">{surface.entropy.toFixed(2)} bits</div>
            )}
          </div>
        )}
      </div>

      {/* Outgoing relations */}
      {surface.relations.length > 0 && (
        <div className="flex flex-wrap gap-1 mt-2">
          {surface.relations.map((rel, i) => (
            <span
              key={i}
              className={`text-xs px-1.5 py-0.5 rounded ${getRelationColor(rel.type)}`}
            >
              {getRelationArrow(rel.type)} {rel.target}
            </span>
          ))}
        </div>
      )}

      {/* Incoming relations */}
      {incomingRelations.length > 0 && (
        <div className="flex flex-wrap gap-1 mt-1">
          {incomingRelations.map((inc, i) => (
            <span
              key={i}
              className="text-xs px-1.5 py-0.5 rounded bg-slate-100 text-slate-500"
            >
              ← {inc.from.name} {getRelationArrow(inc.relation.type)}
            </span>
          ))}
        </div>
      )}

      {/* Entities */}
      {surface.entities && surface.entities.length > 0 && (
        <div className="flex flex-wrap gap-1 mt-2">
          {surface.entities.slice(0, 3).map((entity, i) => (
            <span key={i} className="text-xs bg-purple-50 text-purple-600 px-1.5 py-0.5 rounded">
              {entity}
            </span>
          ))}
          {surface.entities.length > 3 && (
            <span className="text-xs text-slate-400">+{surface.entities.length - 3}</span>
          )}
        </div>
      )}
    </div>
  )
}

export default SurfacesTopology
