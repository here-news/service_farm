import React, { useRef, useEffect, useState } from 'react';

interface Entity {
  id: string;
  canonical_name: string;
  entity_type: string;
}

interface Claim {
  id: string;
  text: string;
}

interface GraphViewProps {
  entities: Entity[];
  claims: Claim[];
  eventName: string;
}

interface Node {
  id: string;
  label: string;
  type: 'event' | 'entity' | 'claim';
  color: string;
  x: number;
  y: number;
  radius: number;
  fullText?: string;
}

const GraphView: React.FC<GraphViewProps> = ({ entities, claims, eventName }) => {
  const [hoveredNode, setHoveredNode] = useState<Node | null>(null);
  const [selectedType, setSelectedType] = useState<'all' | 'entities' | 'claims'>('all');

  // Build graph data with better spacing
  const buildNodes = () => {
    const nodes: Node[] = [];
    const centerX = 400;
    const centerY = 300;

    // Center event node
    nodes.push({
      id: 'event',
      label: eventName.length > 30 ? eventName.slice(0, 30) + '...' : eventName,
      type: 'event',
      color: '#667eea',
      x: centerX,
      y: centerY,
      radius: 40,
      fullText: eventName
    });

    // Entity nodes - larger radius, more spread
    const entityRadius = 180;
    entities.forEach((entity, i) => {
      const angle = (i / entities.length) * Math.PI * 2;
      nodes.push({
        id: entity.id,
        label: entity.canonical_name.length > 20 ? entity.canonical_name.slice(0, 20) + '...' : entity.canonical_name,
        type: 'entity',
        color: getEntityColor(entity.entity_type),
        x: centerX + Math.cos(angle) * entityRadius,
        y: centerY + Math.sin(angle) * entityRadius,
        radius: 20,
        fullText: entity.canonical_name
      });
    });

    // Claim nodes - smaller, on outer ring
    const claimRadius = 280;
    const displayClaims = claims.slice(0, 30);
    displayClaims.forEach((claim, i) => {
      const angle = (i / displayClaims.length) * Math.PI * 2;
      nodes.push({
        id: claim.id,
        label: claim.text.slice(0, 25) + '...',
        type: 'claim',
        color: '#34d399',
        x: centerX + Math.cos(angle) * claimRadius,
        y: centerY + Math.sin(angle) * claimRadius,
        radius: 8,
        fullText: claim.text
      });
    });

    return nodes;
  };

  const getEntityColor = (type: string): string => {
    const colors: Record<string, string> = {
      PERSON: '#f59e0b',
      ORGANIZATION: '#8b5cf6',
      LOCATION: '#10b981',
      GPE: '#10b981',
      EVENT: '#667eea',
    };
    return colors[type.toUpperCase()] || '#6b7280';
  };

  const nodes = buildNodes();
  const filteredNodes = nodes.filter(n => {
    if (selectedType === 'all') return true;
    if (selectedType === 'entities') return n.type === 'event' || n.type === 'entity';
    if (selectedType === 'claims') return n.type === 'event' || n.type === 'claim';
    return true;
  });

  if (entities.length === 0 && claims.length === 0) {
    return (
      <div className="h-[600px] bg-gray-900 rounded-lg flex items-center justify-center">
        <p className="text-gray-400">No graph data available</p>
      </div>
    );
  }

  return (
    <div className="h-[600px] bg-gray-900 rounded-lg overflow-hidden relative">
      {/* Filter tabs */}
      <div className="absolute top-4 left-4 z-10 flex gap-2">
        <button
          onClick={() => setSelectedType('all')}
          className={`px-3 py-1.5 rounded text-sm font-medium transition-colors ${
            selectedType === 'all'
              ? 'bg-blue-500 text-white'
              : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
          }`}
        >
          All
        </button>
        <button
          onClick={() => setSelectedType('entities')}
          className={`px-3 py-1.5 rounded text-sm font-medium transition-colors ${
            selectedType === 'entities'
              ? 'bg-blue-500 text-white'
              : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
          }`}
        >
          Entities
        </button>
        <button
          onClick={() => setSelectedType('claims')}
          className={`px-3 py-1.5 rounded text-sm font-medium transition-colors ${
            selectedType === 'claims'
              ? 'bg-blue-500 text-white'
              : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
          }`}
        >
          Claims
        </button>
      </div>

      {/* SVG Graph */}
      <svg className="w-full h-full" viewBox="0 0 800 600">
        {/* Background gradient */}
        <defs>
          <radialGradient id="bg-gradient" cx="50%" cy="50%">
            <stop offset="0%" stopColor="#1a1a2e" />
            <stop offset="100%" stopColor="#0a0a0a" />
          </radialGradient>
        </defs>
        <rect width="800" height="600" fill="url(#bg-gradient)" />

        {/* Connection lines */}
        <g opacity="0.3">
          {filteredNodes
            .filter(n => n.type !== 'event')
            .map(node => (
              <line
                key={`link-${node.id}`}
                x1={400}
                y1={300}
                x2={node.x}
                y2={node.y}
                stroke={node.color}
                strokeWidth={node.type === 'entity' ? 2 : 1}
                strokeDasharray={node.type === 'claim' ? '4,4' : undefined}
              />
            ))}
        </g>

        {/* Nodes */}
        {filteredNodes.map(node => (
          <g
            key={node.id}
            onMouseEnter={() => setHoveredNode(node)}
            onMouseLeave={() => setHoveredNode(null)}
            className="cursor-pointer transition-transform hover:scale-110"
          >
            {/* Glow effect on hover */}
            {hoveredNode?.id === node.id && (
              <circle
                cx={node.x}
                cy={node.y}
                r={node.radius + 8}
                fill={node.color}
                opacity="0.3"
              />
            )}

            {/* Node circle */}
            <circle
              cx={node.x}
              cy={node.y}
              r={node.radius}
              fill={node.color}
              stroke="white"
              strokeWidth={node.type === 'event' ? 3 : 2}
              opacity={hoveredNode && hoveredNode.id !== node.id ? 0.5 : 1}
            />

            {/* Label for event and entities */}
            {node.type !== 'claim' && (
              <text
                x={node.x}
                y={node.y + node.radius + 15}
                textAnchor="middle"
                fill="white"
                fontSize={node.type === 'event' ? 14 : 11}
                fontWeight={node.type === 'event' ? 'bold' : 'normal'}
                opacity={hoveredNode && hoveredNode.id !== node.id ? 0.5 : 0.9}
              >
                {node.label}
              </text>
            )}
          </g>
        ))}
      </svg>

      {/* Hover tooltip */}
      {hoveredNode && (
        <div className="absolute top-20 left-1/2 -translate-x-1/2 bg-gray-800 rounded-lg p-4 max-w-md shadow-2xl border border-gray-700 z-20">
          <div className="flex items-center gap-2 mb-2">
            <div
              className="w-3 h-3 rounded-full"
              style={{ backgroundColor: hoveredNode.color }}
            />
            <span className="font-semibold capitalize">{hoveredNode.type}</span>
          </div>
          <p className="text-sm text-gray-300">{hoveredNode.fullText || hoveredNode.label}</p>
        </div>
      )}

      {/* Legend */}
      <div className="absolute top-4 right-4 bg-gray-800 rounded-lg p-3 text-xs space-y-2">
        <div className="font-semibold mb-2 text-gray-300">Legend</div>
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 rounded-full bg-[#667eea] border-2 border-white" />
          <span>Event</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-[#f59e0b] border-2 border-white" />
          <span>Person</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-[#8b5cf6] border-2 border-white" />
          <span>Organization</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-[#10b981] border-2 border-white" />
          <span>Location</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-[#34d399] border border-white" />
          <span>Claim</span>
        </div>
      </div>

      <div className="absolute bottom-4 left-4 text-xs text-gray-500">
        {filteredNodes.length} nodes visible • Hover to see details
      </div>
    </div>
  );
};

export default GraphView;
