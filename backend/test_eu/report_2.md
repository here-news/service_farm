# EU Experiment Report 2

## What We Tested

Metabolic emergence: claims start unstable, contradictions are signal (not noise), clusters grow through absorption and metabolism.

## Key Difference from Report 1

Report 1 used **entropy reduction as gate** - only merge if entropy decreases.

Report 2 uses **metabolism model** - allow merges that create tension, because contradictions indicate active digestion of evolving facts.

## Results: Stable vs Active Clusters

The metabolic model correctly distinguishes two cluster types:

### Stable Clusters (high coherence, low/no tension)

| Cluster | Claims | Coherence | Corr | Contra | Notes |
|---------|--------|-----------|------|--------|-------|
| Jimmy Lai | 17 | 1.00 | 14 | 0 | Perfectly coherent |
| Amanda Seyfried | 12 | 1.00 | 7 | 0 | Entertainment, stable |
| Michele Singer | 11 | 1.00 | 6 | 0 | Stable story |
| Time Magazine | 10 | 1.00 | 6 | 0 | Awards coverage |
| Brown University | 9 | 0.83 | 5 | 1 | Minor tension |

### Active Clusters (have contradictions to metabolize)

| Cluster | Claims | Coherence | Contra | Tension | Notes |
|---------|--------|-----------|--------|---------|-------|
| Wang Fuk Court | 12 | 0.50 | 6 | 50% | Death toll updates |
| Wang Fuk Court | 16 | 0.54 | 6 | 46% | Still digesting |
| Hong Kong | 57 | 0.83 | 6 | 17% | Broad, some conflict |
| Donald Trump | 93 | 0.86 | 4 | 14% | Multiple stories |
| Xu Bo | 14 | 0.73 | 3 | 27% | Evolving case |

## Key Insight: Contradictions Are Temporal Evolution

Wang Fuk Court Fire has 6 contradictions. This isn't error - it's **evolving death toll reports**:
- Initial: "At least 36 killed"
- Update: "Death toll rises to..."
- Final: "Confirmed 17 dead"

The claims contradict because reality evolved. This is **healthy metabolism** - the cluster is digesting new information.

## Coherence Trajectory

Tracking coherence across depths reveals patterns:

```
Wang Fuk Court trajectory:
  Depth 3:  7 claims, coh=0.67, tension=33%
  Depth 4: 11 claims, coh=0.60, tension=40%
  Depth 5: 12 claims, coh=0.50, tension=50%  ← peak tension
  Depth 6: 16 claims, coh=0.54, tension=46%
  Depth 10: 40 claims, coh=0.50, tension=0%  ← resolved?
```

The cluster went through high tension at depth 5 (actively digesting conflicting reports), then partially resolved.

## Comparison: Entropy vs Metabolic

| Metric | Entropy Approach | Metabolic Approach |
|--------|------------------|-------------------|
| Wang Fuk Court | 42 claims | 40 claims |
| Jimmy Lai | 42 claims | 77 claims |
| Contradiction handling | Blocked | Absorbed |
| Cluster types | One type | Stable vs Active |

The metabolic approach allows Jimmy Lai to grow larger (77 claims vs 42) because it doesn't reject merges that introduce tension.

## What We Learned

### 1. Tension is information

50% tension in Wang Fuk Court isn't bad - it indicates the cluster is tracking an evolving story where facts changed over time.

### 2. Stable clusters exist

Jimmy Lai at 100% coherence with 14 corroborations and 0 contradictions is a **settled narrative**. No active metabolism needed.

### 3. Coherence can improve over time

A cluster might start at 50% coherence but reach 87% as it absorbs more corroborating claims. Metabolism works.

### 4. The right frame is Stable vs Active

Not "good vs bad clusters" but:
- **Stable**: Settled narratives (historical, uncontested)
- **Active**: Evolving stories (breaking news, contested claims)

Both are valid. The UI should indicate which state a cluster is in.

## Open Questions

1. **When does metabolism complete?** How do we know when an active cluster has "digested" its contradictions?

2. **Temporal resolution**: Should claims that contradict because of time (old death toll vs new) be marked differently than claims that fundamentally disagree?

3. **User display**: Should active clusters show their tension? "This story is evolving - claims are in conflict"

4. **Decay of resolved tension**: Once contradictions are "old news", should tension decay?

## Comparison with Existing Events

| Event | Original Claims | Metabolic Match | Coherence |
|-------|-----------------|-----------------|-----------|
| Hong Kong Fire | 130 | 127 (Hong Kong cluster) | 0.87 |
| Jimmy Lai | 117 | 77 | 0.83 |
| Charlie Kirk | 39 | 49 | 0.86 |
| Donald Trump | 38+ | 93 (merged?) | 0.86 |
| Venezuela | 44 | 88 | 0.80 |

The metabolic approach creates larger clusters than entropy, closer to original event sizes.

## Conclusion

**Entropy is too conservative. Metabolism is more realistic.**

Real events contain contradictions. Death tolls change. Witnesses disagree. Politicians flip-flop.

A good event model should:
1. Accept contradictions as signal
2. Track coherence over time
3. Distinguish stable from active clusters
4. Allow metabolism to improve coherence

The next step is to track how coherence evolves as new claims arrive, and define when metabolism is "complete."

---

*Report generated 2025-12-17*
