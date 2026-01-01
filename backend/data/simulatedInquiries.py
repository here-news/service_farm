"""
Simulated Inquiry Data

Used for demo mode when no real data exists in database.
Mirror of frontend/app/data/simulatedInquiries.ts
"""
from typing import List, Dict, Any
import random

# Simulated data for demo
SIMULATED_INQUIRIES: List[Dict[str, Any]] = [
    # Resolved
    {
        "id": "sim_resolved_1",
        "title": "Did Elon Musk acquire Twitter in 2022?",
        "status": "resolved",
        "rigor": "A",
        "schema_type": "boolean",
        "posterior_map": "true",
        "posterior_probability": 0.99,
        "entropy_bits": 0.08,
        "stake": 250.00,
        "contributions": 12,
        "open_tasks": 0,
        "resolvable": True,
        "resolved_ago": "3 days ago",
        "scope_entities": ["Elon Musk", "Twitter", "X Corp"],
        "cover_image": "https://images.unsplash.com/photo-1611162617474-5b21e879e113?w=200&h=200&fit=crop"
    },
    {
        "id": "sim_resolved_2",
        "title": "How many people attended the 2024 Super Bowl?",
        "status": "resolved",
        "rigor": "A",
        "schema_type": "monotone_count",
        "posterior_map": 61629,
        "posterior_probability": 0.97,
        "entropy_bits": 0.21,
        "stake": 180.50,
        "contributions": 8,
        "open_tasks": 0,
        "resolvable": True,
        "resolved_ago": "1 week ago",
        "scope_entities": ["Super Bowl LVIII", "Las Vegas"],
        "cover_image": "https://images.unsplash.com/photo-1566577739112-5180d4bf9390?w=200&h=200&fit=crop"
    },
    # High Bounty
    {
        "id": "sim_bounty_1",
        "title": "How many Russian soldiers have died in Ukraine as of Dec 2024?",
        "status": "open",
        "rigor": "B",
        "schema_type": "monotone_count",
        "posterior_map": 315000,
        "posterior_probability": 0.32,
        "entropy_bits": 4.8,
        "stake": 5000.00,
        "contributions": 24,
        "open_tasks": 4,
        "resolvable": False,
        "scope_entities": ["Russian Armed Forces", "Ukraine"],
        "cover_image": "https://images.unsplash.com/photo-1646662231083-f0b2ba8b8f32?w=200&h=200&fit=crop"
    },
    {
        "id": "sim_bounty_2",
        "title": "Will GPT-5 be released before July 2025?",
        "status": "open",
        "rigor": "B",
        "schema_type": "forecast",
        "posterior_map": "true",
        "posterior_probability": 0.62,
        "entropy_bits": 0.96,
        "stake": 2500.00,
        "contributions": 15,
        "open_tasks": 2,
        "resolvable": False,
        "scope_entities": ["OpenAI", "GPT-5"],
        "cover_image": "https://images.unsplash.com/photo-1677442136019-21780ecad995?w=200&h=200&fit=crop"
    },
    {
        "id": "sim_bounty_3",
        "title": "What is the actual unemployment rate in China (Dec 2024)?",
        "status": "open",
        "rigor": "C",
        "schema_type": "monotone_count",
        "posterior_map": 18,
        "posterior_probability": 0.28,
        "entropy_bits": 3.2,
        "stake": 1800.00,
        "contributions": 9,
        "open_tasks": 3,
        "resolvable": False,
        "scope_entities": ["China", "NBS"],
        "cover_image": "https://images.unsplash.com/photo-1474181487882-5abf3f0ba6c2?w=200&h=200&fit=crop"
    },
    # Contested
    {
        "id": "sim_contested_1",
        "title": "How many people were killed in the Gaza hospital explosion (Oct 2023)?",
        "status": "open",
        "rigor": "B",
        "schema_type": "monotone_count",
        "posterior_map": 300,
        "posterior_probability": 0.18,
        "entropy_bits": 5.2,
        "stake": 800.00,
        "contributions": 31,
        "open_tasks": 5,
        "resolvable": False,
        "scope_entities": ["Al-Ahli Hospital", "Gaza"],
        "cover_image": "https://images.unsplash.com/photo-1584036561566-baf8f5f1b144?w=200&h=200&fit=crop"
    },
    {
        "id": "sim_contested_2",
        "title": "Did lab leak cause COVID-19?",
        "status": "open",
        "rigor": "C",
        "schema_type": "categorical",
        "posterior_map": "uncertain",
        "posterior_probability": 0.35,
        "entropy_bits": 1.58,
        "stake": 1200.00,
        "contributions": 45,
        "open_tasks": 6,
        "resolvable": False,
        "scope_entities": ["Wuhan", "SARS-CoV-2"],
        "cover_image": "https://images.unsplash.com/photo-1584036561566-baf8f5f1b144?w=200&h=200&fit=crop"
    },
    # More open
    {
        "id": "sim_open_1",
        "title": "How many cars did Tesla deliver in Q4 2024?",
        "status": "open",
        "rigor": "A",
        "schema_type": "monotone_count",
        "posterior_map": 485000,
        "posterior_probability": 0.65,
        "entropy_bits": 1.8,
        "stake": 400.00,
        "contributions": 6,
        "open_tasks": 1,
        "resolvable": False,
        "scope_entities": ["Tesla", "Q4 2024"],
        "cover_image": "https://images.unsplash.com/photo-1617788138017-80ad40651399?w=200&h=200&fit=crop"
    }
]


def generate_simulated_trace(inquiry: Dict[str, Any]) -> Dict[str, Any]:
    """Generate simulated trace data based on an inquiry."""
    is_count = inquiry.get("schema_type") == "monotone_count"
    map_value = inquiry.get("posterior_map")

    # Generate claims based on the inquiry type
    if is_count:
        claims = [
            {
                "id": "c1",
                "icon": "📰",
                "source": "Reuters",
                "text": f'"Sources report approximately {map_value}"',
                "extracted_value": map_value,
                "observation_kind": "approximate"
            },
            {
                "id": "c2",
                "icon": "📺",
                "source": "BBC News",
                "text": f'"Officials confirmed at least {int(float(map_value) * 0.95)}"',
                "extracted_value": int(float(map_value) * 0.95),
                "observation_kind": "lower_bound"
            },
            {
                "id": "c3",
                "icon": "🏛️",
                "source": "Official Statement",
                "text": f'"The verified count stands at {int(float(map_value) * 0.98)}"',
                "extracted_value": int(float(map_value) * 0.98),
                "observation_kind": "point"
            },
        ]
    else:
        claims = [
            {
                "id": "c1",
                "icon": "📰",
                "source": "Reuters",
                "text": f'"Evidence suggests {map_value}"',
                "extracted_value": map_value,
                "observation_kind": "point"
            },
            {
                "id": "c2",
                "icon": "📺",
                "source": "AP News",
                "text": f'"Analysis indicates {map_value}"',
                "extracted_value": map_value,
                "observation_kind": "approximate"
            },
        ]

    contributions_count = inquiry.get("contributions", 3)

    return {
        "inquiry": {"id": inquiry["id"], "title": inquiry["title"]},
        "belief_state": {
            "map": map_value,
            "map_probability": inquiry.get("posterior_probability", 0),
            "entropy_bits": inquiry.get("entropy_bits", 0),
            "normalized_entropy": min(1, inquiry.get("entropy_bits", 0) / 5),
            "observation_count": contributions_count,
            "total_log_score": -inquiry.get("entropy_bits", 0) * 2
        },
        "surfaces": [
            {
                "id": "s1",
                "name": "Primary Sources",
                "claim_count": max(1, contributions_count // 3),
                "sources": ["Official", "Gov"],
                "in_scope": True,
                "relations": [{"type": "CONFIRMS", "target": "S2"}]
            },
            {
                "id": "s2",
                "name": "Wire Services",
                "claim_count": max(1, contributions_count // 2),
                "sources": ["Reuters", "AP"],
                "in_scope": True,
                "relations": []
            },
            {
                "id": "s3",
                "name": "Analysis",
                "claim_count": max(1, contributions_count // 4),
                "sources": ["BBC", "NYT"],
                "in_scope": True,
                "relations": [{"type": "SUPERSEDES", "target": "S1"}]
            },
        ],
        "observations": [
            {
                "kind": c["observation_kind"],
                "value_distribution": {str(c["extracted_value"]): 0.5},
                "source": c["source"]
            }
            for c in claims
        ],
        "contributions": [
            {
                "id": c["id"],
                "type": "evidence",
                "text": c["text"],
                "source": c["source"],
                "source_name": c["source"],
                "extracted_value": c["extracted_value"],
                "impact": 0.05 + random.random() * 0.1,
                "posterior_impact": 0.05 + random.random() * 0.1,
                "created_at": "2024-12-20T10:00:00Z"
            }
            for i, c in enumerate(claims)
        ],
        "tasks": [
            {
                "id": "t1",
                "type": "verification_needed",
                "description": "Verify with independent primary source",
                "bounty": inquiry.get("stake", 100) * 0.1,
                "completed": False
            },
            {
                "id": "t2",
                "type": "high_entropy",
                "description": "Reduce uncertainty with authoritative data",
                "bounty": inquiry.get("stake", 100) * 0.15,
                "completed": False
            }
        ] if inquiry.get("open_tasks", 0) > 0 else [],
        "resolution": {
            "status": inquiry.get("status", "open"),
            "resolvable": inquiry.get("resolvable", False),
            "blocking_tasks": ["t1"] if inquiry.get("open_tasks", 0) > 0 else []
        },
        "posterior_top_10": _generate_count_distribution(map_value, inquiry.get("posterior_probability", 0.5))
            if is_count else _generate_categorical_distribution(map_value, inquiry.get("posterior_probability", 0.5)),
        "claims": claims
    }


def _generate_count_distribution(map_val: Any, map_prob: float) -> List[Dict[str, Any]]:
    """Generate count distribution for posterior visualization."""
    try:
        map_num = int(float(map_val))
    except (TypeError, ValueError):
        return [{"value": map_val, "probability": map_prob}]

    results = [{"value": map_num, "probability": map_prob}]
    remaining = 1 - map_prob
    offsets = [-5, 5, -10, 10, -15, 15, -20]

    for offset in offsets:
        if remaining <= 0.01:
            break
        prob = remaining * (0.3 + random.random() * 0.3)
        results.append({"value": map_num + offset, "probability": prob})
        remaining -= prob

    return sorted(results, key=lambda x: -x["probability"])[:8]


def _generate_categorical_distribution(map_val: Any, map_prob: float) -> List[Dict[str, Any]]:
    """Generate categorical distribution for posterior visualization."""
    if str(map_val).lower() in ("true", "false"):
        return [
            {"value": "true", "probability": map_prob if str(map_val).lower() == "true" else 1 - map_prob},
            {"value": "false", "probability": 1 - map_prob if str(map_val).lower() == "true" else map_prob}
        ]
    return [
        {"value": map_val, "probability": map_prob},
        {"value": "other", "probability": 1 - map_prob}
    ]
