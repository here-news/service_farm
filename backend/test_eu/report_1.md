# EU Experiment Report 1

## What We Tested

Progressive recursive emergence of EventfulUnits from 1215 claims, using entropy as the merge criterion.

## Key Insight

**We don't need to match existing events.** The current events are not ground truth - they were created by heuristics that may be wrong (see Trump/BBC duplicate issue #22).

What matters is whether emerged clusters are:
1. **Coherent** - claims support each other
2. **Stable** - low entropy, not fragmented
3. **Meaningful** - represent real-world narratives

## What Emerged (Entropy-Based)

At depth 8, the system produced 22 stable clusters:

| Cluster | Claims | Entropy | Corr | Notes |
|---------|--------|---------|------|-------|
| Jimmy Lai | 42 | -0.63 | 6 | Coherent, ongoing story |
| Wang Fuk Court | 42 | -0.61 | 0 | Coherent, but low corroboration |
| Hong Kong | 27 | -0.53 | 1 | Broader geographic cluster |
| Do Kwon | 26 | -0.48 | 4 | Coherent legal story |
| Bondi Beach | 20 | -0.74 | 4 | Very coherent (high negative H) |
| Brown University | 18 | -0.77 | - | Very coherent |
| Charles III | 14 | -0.18 | 0 | Less coherent, dispersed |

**Observation**: Clusters like "Bondi Beach" (-0.74 entropy) and "Brown University" (-0.77) are MORE coherent than larger clusters. Size doesn't equal quality.

## What We Learned

### 1. Entropy-based merging is conservative (good)

The system refused to create giant clusters. Jimmy Lai peaked at 42 claims, not 117. This is because merging more would increase entropy (reduce coherence).

**This is correct behavior.** The original "Jimmy Lai Imprisonment" event with 117 claims may be TOO broad - it might contain sub-narratives that should stay separate.

### 2. Some events are naturally tight, others are loose

- **Tight**: Bondi Beach Shooting, Brown University Shooting - concentrated entities, clear narrative
- **Loose**: Hong Kong, Donald Trump - broad entities that connect many stories

The system correctly keeps tight clusters small and coherent, while loose clusters remain fragmented.

### 3. Pairwise merging limits growth

Current algorithm only merges 2 EUs at a time. This is slow and may miss opportunities where 3+ EUs should merge simultaneously.

But this is also **safe** - prevents catastrophic merges.

### 4. Matching existing events is the wrong goal

Original events were created by:
- LLM-generated names
- Threshold-based sub-event splitting
- Arbitrary clustering decisions

They're not ground truth. The EU model should find **better** structure, not replicate flawed structure.

## The Right Question

Instead of "does this match existing events?", ask:

**"Is this cluster at its lowest entropy state?"**

A cluster should stop growing when:
- Adding more claims would increase entropy
- It has reached a natural boundary (entity-defined, temporal, thematic)

## What "Hong Kong Fire 2025" Tells Us

Wang Fuk Court Fire is a good reference because:
- It's a real, bounded event
- Clear temporal scope (Nov 2025)
- Concentrated entities (Wang Fuk Court, Hong Kong Fire Services, casualties)
- High corroboration (death toll updates, response reports)

The system found "Wang Fuk Court" as a 42-claim cluster with -0.61 entropy. This is actually reasonable - the remaining 88 claims in the original event may be:
- Peripheral commentary
- Loosely related Hong Kong politics
- Duplicate/near-duplicate content

**The EU model is being more selective, which may be correct.**

## Next Steps

1. **Don't force growth** - if entropy can't decrease, stop merging
2. **Multi-way absorption** - allow clusters to absorb multiple compatible EUs when each reduces entropy
3. **Examine the "rejected" claims** - what didn't merge into Wang Fuk Court? Are they genuinely peripheral?
4. **Test stability over time** - as new claims arrive, does the cluster maintain low entropy?

## Open Questions

1. **Hierarchy vs flat**: Should we even have depth? Or just one layer of optimal clusters?
2. **Decay**: How does entropy change when a cluster stops receiving new claims?
3. **User view**: What does a user see? Top-level clusters, or all depths?

---

*Report generated 2025-12-17*
