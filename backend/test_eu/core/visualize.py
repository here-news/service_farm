"""
Epistemic Visualization Suite
=============================

Visualizations for understanding emergence behavior:

1. Emergence Curve: B³/Completeness vs claims-per-event
2. Threshold Sweep: Precision/Recall tradeoff
3. Cluster Structure: Surface composition heatmap
4. Coherence Dashboard: Temporal/Anchor/Semantic signals
5. LLM ROI Curve: F1 gain vs LLM calls
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from datetime import datetime

# Use non-interactive backend for server
import matplotlib
matplotlib.use('Agg')


# =============================================================================
# 1. EMERGENCE CURVE
# =============================================================================

def plot_emergence_curve(
    results: List[Dict],  # [{claims_per_event, b3_f1, completeness, num_clusters, ...}]
    output_path: str = None,
    title: str = "Emergence Curve"
) -> str:
    """
    Plot B³ F1 and Completeness vs claims-per-event.

    Shows where emergence "kicks in" - when do clusters form?
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Sort by claims_per_event
    results = sorted(results, key=lambda x: x['claims_per_event'])

    x = [r['claims_per_event'] for r in results]
    b3_f1 = [r['b3_f1'] for r in results]
    completeness = [r['completeness'] for r in results]
    num_clusters = [r['num_clusters'] for r in results]
    gt_events = results[0].get('num_gt_events', 1) if results else 1

    # Primary axis: B³ F1 and Completeness
    color1, color2 = '#2ecc71', '#3498db'
    ax1.plot(x, b3_f1, 'o-', color=color1, linewidth=2, markersize=8, label='B³ F1')
    ax1.plot(x, completeness, 's-', color=color2, linewidth=2, markersize=8, label='Completeness')
    ax1.set_xlabel('Claims per Event', fontsize=12)
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_ylim(0, 1.05)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Secondary axis: Number of clusters
    ax2 = ax1.twinx()
    ax2.bar(x, num_clusters, alpha=0.3, color='#e74c3c', width=0.5, label='Clusters')
    ax2.axhline(y=gt_events, color='#e74c3c', linestyle='--', alpha=0.7, label=f'GT Events ({gt_events})')
    ax2.set_ylabel('Number of Clusters', fontsize=12, color='#e74c3c')
    ax2.tick_params(axis='y', labelcolor='#e74c3c')

    # Title and annotations
    plt.title(title, fontsize=14, fontweight='bold')

    # Find emergence point (where B³ F1 > 0.5)
    emergence_point = None
    for r in results:
        if r['b3_f1'] > 0.5:
            emergence_point = r['claims_per_event']
            break

    if emergence_point:
        ax1.axvline(x=emergence_point, color='#9b59b6', linestyle=':', alpha=0.7)
        ax1.annotate(f'Emergence ≈{emergence_point}',
                     xy=(emergence_point, 0.5),
                     xytext=(emergence_point + 1, 0.6),
                     fontsize=10, color='#9b59b6',
                     arrowprops=dict(arrowstyle='->', color='#9b59b6'))

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return output_path
    else:
        # Return as base64 for web display
        import io
        import base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        return base64.b64encode(buf.read()).decode('utf-8')


# =============================================================================
# 2. THRESHOLD SWEEP
# =============================================================================

def plot_threshold_sweep(
    results: List[Dict],  # [{threshold, precision, recall, f1, num_clusters}]
    output_path: str = None,
    title: str = "Threshold Sensitivity"
) -> str:
    """
    Plot Precision/Recall/F1 across affinity thresholds.

    Shows precision-recall tradeoff and optimal operating point.
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))

    results = sorted(results, key=lambda x: x['threshold'])

    x = [r['threshold'] for r in results]
    precision = [r['precision'] for r in results]
    recall = [r['recall'] for r in results]
    f1 = [r['f1'] for r in results]

    ax1.plot(x, precision, 'o-', color='#2ecc71', linewidth=2, markersize=8, label='Precision')
    ax1.plot(x, recall, 's-', color='#3498db', linewidth=2, markersize=8, label='Recall')
    ax1.plot(x, f1, '^-', color='#e74c3c', linewidth=2, markersize=8, label='F1')

    ax1.set_xlabel('Affinity Threshold', fontsize=12)
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_ylim(0, 1.05)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # Mark optimal F1
    best_idx = np.argmax(f1)
    best_thresh = x[best_idx]
    best_f1 = f1[best_idx]
    ax1.scatter([best_thresh], [best_f1], s=200, c='#e74c3c', marker='*', zorder=5)
    ax1.annotate(f'Best F1={best_f1:.2f}\n@{best_thresh}',
                 xy=(best_thresh, best_f1),
                 xytext=(best_thresh + 0.05, best_f1 - 0.1),
                 fontsize=10)

    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return output_path
    else:
        import io, base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        return base64.b64encode(buf.read()).decode('utf-8')


# =============================================================================
# 3. CLUSTER COMPOSITION HEATMAP
# =============================================================================

def plot_cluster_heatmap(
    surfaces: List,  # List of Surface objects
    gt_map: Dict[str, str],  # claim_id -> event_label
    output_path: str = None,
    title: str = "Cluster Composition"
) -> str:
    """
    Heatmap showing which GT events are in which predicted clusters.

    Perfect clustering = diagonal only.
    """
    # Get unique events and surfaces
    event_labels = sorted(set(gt_map.values()))
    surface_ids = [s.id for s in surfaces if len(s.claim_ids) > 0]

    # Build matrix
    matrix = np.zeros((len(event_labels), len(surface_ids)))

    for j, surface in enumerate(surfaces):
        if surface.id not in surface_ids:
            continue
        col_idx = surface_ids.index(surface.id)
        for claim_id in surface.claim_ids:
            if claim_id in gt_map:
                row_idx = event_labels.index(gt_map[claim_id])
                matrix[row_idx, col_idx] += 1

    # Normalize by row (per event)
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    matrix_norm = matrix / row_sums

    fig, ax = plt.subplots(figsize=(12, 6))

    im = ax.imshow(matrix_norm, cmap='Blues', aspect='auto', vmin=0, vmax=1)

    # Labels
    ax.set_xticks(range(len(surface_ids)))
    ax.set_xticklabels(surface_ids, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(len(event_labels)))
    ax.set_yticklabels([e[:25] for e in event_labels], fontsize=9)

    ax.set_xlabel('Predicted Surfaces', fontsize=12)
    ax.set_ylabel('Ground Truth Events', fontsize=12)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Fraction of Event', fontsize=10)

    # Add text annotations for significant values
    for i in range(len(event_labels)):
        for j in range(len(surface_ids)):
            val = matrix_norm[i, j]
            if val > 0.1:
                count = int(matrix[i, j])
                ax.text(j, i, f'{count}', ha='center', va='center',
                       fontsize=8, color='white' if val > 0.5 else 'black')

    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return output_path
    else:
        import io, base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        return base64.b64encode(buf.read()).decode('utf-8')


# =============================================================================
# 4. COHERENCE DASHBOARD
# =============================================================================

def plot_coherence_dashboard(
    coherence_metrics: Dict[str, 'CoherenceMetrics'],
    output_path: str = None,
    title: str = "Coherence Signals"
) -> str:
    """
    Multi-panel dashboard showing coherence signals per surface.
    """
    surfaces = list(coherence_metrics.keys())
    n = len(surfaces)

    if n == 0:
        return None

    # Extract metrics
    temp_viols = [coherence_metrics[s].temporal_violations for s in surfaces]
    anchor_coh = [coherence_metrics[s].anchor_coherence for s in surfaces]
    sem_disp = [coherence_metrics[s].semantic_dispersion for s in surfaces]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # 1. Temporal violations
    ax1 = axes[0]
    colors1 = ['#e74c3c' if v > 0.2 else '#2ecc71' for v in temp_viols]
    ax1.barh(range(n), temp_viols, color=colors1, alpha=0.7)
    ax1.set_yticks(range(n))
    ax1.set_yticklabels(surfaces, fontsize=8)
    ax1.set_xlabel('Violation Rate')
    ax1.set_title('Temporal Violations', fontweight='bold')
    ax1.axvline(x=0.2, color='#e74c3c', linestyle='--', alpha=0.5)
    ax1.set_xlim(0, 1)

    # 2. Anchor coherence
    ax2 = axes[1]
    colors2 = ['#2ecc71' if c > 0.5 else '#f39c12' if c > 0.2 else '#e74c3c' for c in anchor_coh]
    ax2.barh(range(n), anchor_coh, color=colors2, alpha=0.7)
    ax2.set_yticks(range(n))
    ax2.set_yticklabels(surfaces, fontsize=8)
    ax2.set_xlabel('Coherence Score')
    ax2.set_title('Anchor Coherence', fontweight='bold')
    ax2.set_xlim(0, 1)

    # 3. Semantic dispersion
    ax3 = axes[2]
    colors3 = ['#2ecc71' if d < 0.2 else '#f39c12' if d < 0.4 else '#e74c3c' for d in sem_disp]
    ax3.barh(range(n), sem_disp, color=colors3, alpha=0.7)
    ax3.set_yticks(range(n))
    ax3.set_yticklabels(surfaces, fontsize=8)
    ax3.set_xlabel('Dispersion (lower = tighter)')
    ax3.set_title('Semantic Dispersion', fontweight='bold')
    ax3.set_xlim(0, 0.6)

    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return output_path
    else:
        import io, base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        return base64.b64encode(buf.read()).decode('utf-8')


# =============================================================================
# 5. LLM ROI CURVE
# =============================================================================

def plot_llm_roi(
    results: List[Dict],  # [{llm_calls, f1_base, f1_with_llm, f1_gain}]
    output_path: str = None,
    title: str = "LLM ROI Analysis"
) -> str:
    """
    Plot F1 gain vs LLM calls to understand ROI.
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))

    results = sorted(results, key=lambda x: x['llm_calls'])

    x = [r['llm_calls'] for r in results]
    f1_base = [r['f1_base'] for r in results]
    f1_llm = [r['f1_with_llm'] for r in results]
    f1_gain = [r['f1_gain'] for r in results]

    ax1.plot(x, f1_base, 'o--', color='#95a5a6', linewidth=2, markersize=8, label='Base (no LLM)')
    ax1.plot(x, f1_llm, 'o-', color='#2ecc71', linewidth=2, markersize=8, label='With LLM')

    # Fill between to show gain
    ax1.fill_between(x, f1_base, f1_llm, alpha=0.2, color='#2ecc71')

    ax1.set_xlabel('LLM Calls', fontsize=12)
    ax1.set_ylabel('B³ F1 Score', fontsize=12)
    ax1.set_ylim(0, 1.05)
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)

    # Annotate ROI
    if len(results) > 0:
        total_calls = sum(x)
        avg_gain = np.mean(f1_gain)
        ax1.annotate(f'Avg gain: {avg_gain:.1%}',
                     xy=(max(x)*0.7, max(f1_llm)*0.9),
                     fontsize=11, color='#27ae60', fontweight='bold')

    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return output_path
    else:
        import io, base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        return base64.b64encode(buf.read()).decode('utf-8')


# =============================================================================
# 6. ABLATION COMPARISON
# =============================================================================

def plot_ablation_comparison(
    ablations: List['AblationResult'],
    output_path: str = None,
    title: str = "Ablation Study"
) -> str:
    """
    Bar chart comparing ablation configurations.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    configs = [a.config for a in ablations]
    b3_f1 = [a.b3_f1 for a in ablations]
    completeness = [a.completeness for a in ablations]

    x = np.arange(len(configs))
    width = 0.35

    bars1 = ax.bar(x - width/2, b3_f1, width, label='B³ F1', color='#2ecc71', alpha=0.8)
    bars2 = ax.bar(x + width/2, completeness, width, label='Completeness', color='#3498db', alpha=0.8)

    ax.set_xlabel('Configuration', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=15, ha='right')
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.0%}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return output_path
    else:
        import io, base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        return base64.b64encode(buf.read()).decode('utf-8')


# =============================================================================
# 7. SURFACE NETWORK GRAPH
# =============================================================================

def plot_surface_network(
    surfaces: List,
    claims: Dict[str, 'Claim'],
    gt_map: Dict[str, str] = None,
    output_path: str = None,
    title: str = "Surface Network"
) -> str:
    """
    Network graph showing surfaces as nodes, with internal claim structure.
    """
    try:
        import networkx as nx
    except ImportError:
        return None

    fig, ax = plt.subplots(figsize=(12, 10))

    G = nx.Graph()

    # Add surface nodes
    colors = []
    sizes = []
    labels = {}

    event_colors = {}
    color_palette = plt.cm.Set3(np.linspace(0, 1, 12))

    for i, surface in enumerate(surfaces):
        G.add_node(surface.id)

        # Size by claim count
        sizes.append(100 + len(surface.claim_ids) * 50)

        # Color by GT event (if available)
        if gt_map:
            events_in = set()
            for cid in surface.claim_ids:
                if cid in gt_map:
                    events_in.add(gt_map[cid])

            if len(events_in) == 1:
                event = list(events_in)[0]
                if event not in event_colors:
                    event_colors[event] = color_palette[len(event_colors) % 12]
                colors.append(event_colors[event])
            else:
                colors.append('#cccccc')  # Mixed = gray
        else:
            colors.append('#3498db')

        labels[surface.id] = f"{surface.id}\n({len(surface.claim_ids)})"

    # Add edges between surfaces with shared entities
    for i, s1 in enumerate(surfaces):
        for s2 in surfaces[i+1:]:
            shared = s1.anchor_entities & s2.anchor_entities
            if shared:
                G.add_edge(s1.id, s2.id, weight=len(shared))

    # Layout
    pos = nx.spring_layout(G, k=2, iterations=50)

    # Draw
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors, node_size=sizes, alpha=0.8)
    nx.draw_networkx_labels(G, pos, ax=ax, labels=labels, font_size=8)
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3, edge_color='#888888')

    # Legend for events
    if gt_map and event_colors:
        legend_elements = [Patch(facecolor=c, label=e[:20]) for e, c in event_colors.items()]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=9)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return output_path
    else:
        import io, base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        return base64.b64encode(buf.read()).decode('utf-8')


# =============================================================================
# MAIN: Generate all visualizations
# =============================================================================

async def generate_all_visualizations(
    engine: 'EmergenceEngine',
    ground_truth: 'GroundTruth',
    output_dir: str
) -> Dict[str, str]:
    """Generate all visualizations and save to output directory."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    from .evaluation import evaluate_clustering, evaluate_coherence

    surfaces = list(engine.surfaces.values())
    gt_map = ground_truth.claim_to_event()

    paths = {}

    # 1. Cluster heatmap
    path = os.path.join(output_dir, 'cluster_heatmap.png')
    plot_cluster_heatmap(surfaces, gt_map, output_path=path)
    paths['cluster_heatmap'] = path

    # 2. Coherence dashboard
    coherence = evaluate_coherence(surfaces, engine.claims)
    path = os.path.join(output_dir, 'coherence_dashboard.png')
    plot_coherence_dashboard(coherence, output_path=path)
    paths['coherence_dashboard'] = path

    # 3. Surface network
    path = os.path.join(output_dir, 'surface_network.png')
    plot_surface_network(surfaces, engine.claims, gt_map, output_path=path)
    paths['surface_network'] = path

    return paths


if __name__ == "__main__":
    # Test with synthetic data
    print("Generating test visualizations...")

    # Test emergence curve
    test_results = [
        {'claims_per_event': 2, 'b3_f1': 0.2, 'completeness': 0.3, 'num_clusters': 8, 'num_gt_events': 3},
        {'claims_per_event': 5, 'b3_f1': 0.4, 'completeness': 0.5, 'num_clusters': 6, 'num_gt_events': 3},
        {'claims_per_event': 10, 'b3_f1': 0.6, 'completeness': 0.7, 'num_clusters': 4, 'num_gt_events': 3},
        {'claims_per_event': 20, 'b3_f1': 0.8, 'completeness': 0.85, 'num_clusters': 3, 'num_gt_events': 3},
    ]

    plot_emergence_curve(test_results, output_path='/tmp/emergence_curve.png')
    print("Saved: /tmp/emergence_curve.png")

    # Test threshold sweep
    threshold_results = [
        {'threshold': 0.3, 'precision': 0.6, 'recall': 0.9, 'f1': 0.72},
        {'threshold': 0.4, 'precision': 0.75, 'recall': 0.7, 'f1': 0.72},
        {'threshold': 0.5, 'precision': 0.85, 'recall': 0.5, 'f1': 0.63},
        {'threshold': 0.6, 'precision': 0.95, 'recall': 0.3, 'f1': 0.46},
        {'threshold': 0.7, 'precision': 1.0, 'recall': 0.15, 'f1': 0.26},
    ]

    plot_threshold_sweep(threshold_results, output_path='/tmp/threshold_sweep.png')
    print("Saved: /tmp/threshold_sweep.png")

    print("\nDone!")
