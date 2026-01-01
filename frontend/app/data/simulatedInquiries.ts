import { InquirySummary } from '../types/inquiry'

// Simulated data for demo (will be replaced by real API)
export const SIMULATED_INQUIRIES: InquirySummary[] = [
  // Resolved
  {
    id: 'sim_resolved_1',
    title: 'Did Elon Musk acquire Twitter in 2022?',
    status: 'resolved',
    rigor: 'A',
    schema_type: 'boolean',
    posterior_map: 'true',
    posterior_probability: 0.99,
    entropy_bits: 0.08,
    stake: 250.00,
    contributions: 12,
    open_tasks: 0,
    resolvable: true,
    resolved_ago: '3 days ago',
    scope_entities: ['Elon Musk', 'Twitter', 'X Corp'],
    cover_image: 'https://images.unsplash.com/photo-1611162617474-5b21e879e113?w=200&h=200&fit=crop'
  },
  {
    id: 'sim_resolved_2',
    title: 'How many people attended the 2024 Super Bowl?',
    status: 'resolved',
    rigor: 'A',
    schema_type: 'monotone_count',
    posterior_map: 61629,
    posterior_probability: 0.97,
    entropy_bits: 0.21,
    stake: 180.50,
    contributions: 8,
    open_tasks: 0,
    resolvable: true,
    resolved_ago: '1 week ago',
    scope_entities: ['Super Bowl LVIII', 'Las Vegas'],
    cover_image: 'https://images.unsplash.com/photo-1566577739112-5180d4bf9390?w=200&h=200&fit=crop'
  },
  // High Bounty
  {
    id: 'sim_bounty_1',
    title: 'How many Russian soldiers have died in Ukraine as of Dec 2024?',
    status: 'open',
    rigor: 'B',
    schema_type: 'monotone_count',
    posterior_map: 315000,
    posterior_probability: 0.32,
    entropy_bits: 4.8,
    stake: 5000.00,
    contributions: 24,
    open_tasks: 4,
    resolvable: false,
    scope_entities: ['Russian Armed Forces', 'Ukraine'],
    cover_image: 'https://images.unsplash.com/photo-1646662231083-f0b2ba8b8f32?w=200&h=200&fit=crop'
  },
  {
    id: 'sim_bounty_2',
    title: 'Will GPT-5 be released before July 2025?',
    status: 'open',
    rigor: 'B',
    schema_type: 'forecast',
    posterior_map: 'true',
    posterior_probability: 0.62,
    entropy_bits: 0.96,
    stake: 2500.00,
    contributions: 15,
    open_tasks: 2,
    resolvable: false,
    scope_entities: ['OpenAI', 'GPT-5'],
    cover_image: 'https://images.unsplash.com/photo-1677442136019-21780ecad995?w=200&h=200&fit=crop'
  },
  {
    id: 'sim_bounty_3',
    title: 'What is the actual unemployment rate in China (Dec 2024)?',
    status: 'open',
    rigor: 'C',
    schema_type: 'monotone_count',
    posterior_map: 18,
    posterior_probability: 0.28,
    entropy_bits: 3.2,
    stake: 1800.00,
    contributions: 9,
    open_tasks: 3,
    resolvable: false,
    scope_entities: ['China', 'NBS'],
    cover_image: 'https://images.unsplash.com/photo-1474181487882-5abf3f0ba6c2?w=200&h=200&fit=crop'
  },
  // Contested
  {
    id: 'sim_contested_1',
    title: 'How many people were killed in the Gaza hospital explosion (Oct 2023)?',
    status: 'open',
    rigor: 'B',
    schema_type: 'monotone_count',
    posterior_map: 300,
    posterior_probability: 0.18,
    entropy_bits: 5.2,
    stake: 800.00,
    contributions: 31,
    open_tasks: 5,
    resolvable: false,
    scope_entities: ['Al-Ahli Hospital', 'Gaza'],
    cover_image: 'https://images.unsplash.com/photo-1584036561566-baf8f5f1b144?w=200&h=200&fit=crop'
  },
  {
    id: 'sim_contested_2',
    title: 'Did lab leak cause COVID-19?',
    status: 'open',
    rigor: 'C',
    schema_type: 'categorical',
    posterior_map: 'uncertain',
    posterior_probability: 0.35,
    entropy_bits: 1.58,
    stake: 1200.00,
    contributions: 45,
    open_tasks: 6,
    resolvable: false,
    scope_entities: ['Wuhan', 'SARS-CoV-2'],
    cover_image: 'https://images.unsplash.com/photo-1584036561566-baf8f5f1b144?w=200&h=200&fit=crop'
  },
  // More open
  {
    id: 'sim_open_1',
    title: 'How many cars did Tesla deliver in Q4 2024?',
    status: 'open',
    rigor: 'A',
    schema_type: 'monotone_count',
    posterior_map: 485000,
    posterior_probability: 0.65,
    entropy_bits: 1.8,
    stake: 400.00,
    contributions: 6,
    open_tasks: 1,
    resolvable: false,
    scope_entities: ['Tesla', 'Q4 2024'],
    cover_image: 'https://images.unsplash.com/photo-1617788138017-80ad40651399?w=200&h=200&fit=crop'
  }
]

// Generate simulated trace data based on an inquiry
export function generateSimulatedTrace(inquiry: InquirySummary) {
  const isCount = inquiry.schema_type === 'monotone_count'
  const mapValue = inquiry.posterior_map

  // Generate claims based on the inquiry type
  const claims = isCount ? [
    { id: 'c1', icon: '📰', source: 'Reuters', text: `"Sources report approximately ${mapValue}"`, extracted_value: mapValue, observation_kind: 'approximate' },
    { id: 'c2', icon: '📺', source: 'BBC News', text: `"Officials confirmed at least ${Math.round(Number(mapValue) * 0.95)}"`, extracted_value: Math.round(Number(mapValue) * 0.95), observation_kind: 'lower_bound' },
    { id: 'c3', icon: '🏛️', source: 'Official Statement', text: `"The verified count stands at ${Math.round(Number(mapValue) * 0.98)}"`, extracted_value: Math.round(Number(mapValue) * 0.98), observation_kind: 'point' },
  ] : [
    { id: 'c1', icon: '📰', source: 'Reuters', text: `"Evidence suggests ${mapValue}"`, extracted_value: mapValue, observation_kind: 'point' },
    { id: 'c2', icon: '📺', source: 'AP News', text: `"Analysis indicates ${mapValue}"`, extracted_value: mapValue, observation_kind: 'approximate' },
  ]

  // Assign relationship types to claims for demo
  const relationTypes = ['CONFIRMS', 'REFINES', 'SUPERSEDES', 'CONFLICTS', 'DIVERGENT', 'NOVEL']

  return {
    inquiry: { id: inquiry.id, title: inquiry.title },
    belief_state: {
      map: mapValue,
      map_probability: inquiry.posterior_probability,
      entropy_bits: inquiry.entropy_bits,
      normalized_entropy: Math.min(1, inquiry.entropy_bits / 5),
      observation_count: inquiry.contributions,
      total_log_score: -inquiry.entropy_bits * 2
    },
    surfaces: [
      {
        id: 's1',
        name: 'Primary Sources',
        claim_count: Math.ceil(inquiry.contributions / 3),
        sources: ['Official', 'Gov'],
        in_scope: true,
        relations: [{ type: 'CONFIRMS', target: 'Wire Services' }]
      },
      {
        id: 's2',
        name: 'Wire Services',
        claim_count: Math.ceil(inquiry.contributions / 2),
        sources: ['Reuters', 'AP'],
        in_scope: true,
        relations: [{ type: 'REFINES', target: 'Primary Sources' }]
      },
      {
        id: 's3',
        name: 'Expert Analysis',
        claim_count: Math.floor(inquiry.contributions / 4),
        sources: ['BBC', 'NYT'],
        in_scope: true,
        relations: [{ type: 'SUPERSEDES', target: 'Primary Sources' }]
      },
      ...(inquiry.entropy_bits > 3 ? [{
        id: 's4',
        name: 'Contested Claims',
        claim_count: Math.floor(inquiry.contributions / 5),
        sources: ['Social Media', 'Blogs'],
        in_scope: false,
        relations: [
          { type: 'CONFLICTS', target: 'Primary Sources' },
          { type: 'DIVERGENT', target: 'Wire Services' }
        ]
      }] : [])
    ],
    observations: claims.map(c => ({
      kind: c.observation_kind,
      value_distribution: { [c.extracted_value]: 0.5 },
      source: c.source
    })),
    contributions: claims.map((c, i) => ({
      id: c.id,
      type: 'evidence',
      text: c.text,
      source: c.source,
      source_name: c.source,
      user_name: ['Sarah Chen', 'Mike Torres', 'Anonymous'][i % 3],
      extracted_value: c.extracted_value,
      observation_kind: c.observation_kind,
      impact: 0.05 + Math.random() * 0.1,
      created_at: new Date(Date.now() - i * 86400000).toISOString(),
      // Add relationship info for demo
      relation: i === 0 ? 'CONFIRMS' : i === 1 ? 'REFINES' : 'NOVEL',
      relation_target: i === 0 ? 'Wire Services' : i === 1 ? 'Primary' : undefined
    })),
    tasks: inquiry.open_tasks > 0 ? [
      { id: 't1', inquiry_id: inquiry.id, type: 'need_primary_source', description: 'Verify with independent primary source', bounty: inquiry.stake * 0.1, completed: false, created_at: new Date().toISOString() },
      ...(inquiry.open_tasks > 1 ? [{ id: 't2', inquiry_id: inquiry.id, type: 'high_entropy', description: 'Reduce uncertainty with authoritative data', bounty: inquiry.stake * 0.15, completed: false, created_at: new Date().toISOString() }] : [])
    ] : [],
    resolution: {
      status: inquiry.status,
      resolvable: inquiry.resolvable || false,
      blocking_tasks: inquiry.open_tasks > 0 ? ['t1'] : []
    },
    posterior_top_10: isCount ? generateCountDistribution(Number(mapValue), inquiry.posterior_probability) : generateCategoricalDistribution(mapValue, inquiry.posterior_probability),
    claims // Include claims for the UI
  }
}

function generateCountDistribution(map: number, mapProb: number) {
  const results = [{ value: map, probability: mapProb }]
  let remaining = 1 - mapProb
  const offsets = [-5, 5, -10, 10, -15, 15, -20]

  for (const offset of offsets) {
    if (remaining <= 0.01) break
    const prob = remaining * (0.3 + Math.random() * 0.3)
    results.push({ value: map + offset, probability: prob })
    remaining -= prob
  }

  return results.sort((a, b) => b.probability - a.probability).slice(0, 8)
}

function generateCategoricalDistribution(map: any, mapProb: number) {
  if (typeof map === 'boolean' || map === 'true' || map === 'false') {
    return [
      { value: 'true', probability: map === 'true' || map === true ? mapProb : 1 - mapProb },
      { value: 'false', probability: map === 'true' || map === true ? 1 - mapProb : mapProb }
    ]
  }
  return [
    { value: map, probability: mapProb },
    { value: 'other', probability: 1 - mapProb }
  ]
}
