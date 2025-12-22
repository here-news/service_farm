# EU Experiment Report 6

## Frame Emergence: Three-Level Hierarchy

### Experiment Goal

Test whether Events (level 1) can automatically merge into Frames (level 2) using the same semantic emergence mechanism.

### Results

```
Total claims: 1215
Sub-events (level 0): 553
Events (level 1): 17
Frames (level 2): 1
LLM calls: 450
```

### The Emerged Frame

Only ONE frame emerged naturally:

```
[FRAME] "Wang Fuk Court Fire Incident"
        77 claims, 23 pages, ACTIVE ⚡
        coherence=67%, mass=29.6
        └─ [EVENT] Wang Fuk Court Fire details (66 claims)
        └─ [EVENT] Fire ignition/mesh netting (11 claims)
```

### Why Only One Frame?

The other events remained unframed because they **don't share narratives**:

| Event | Claims | Would Need |
|-------|--------|------------|
| Jimmy Lai Trial | 99 | More Hong Kong press freedom events |
| Bondi Beach Shooting | 42 | More Australian crime/violence events |
| Brown University Shooting | 39 | More US campus violence events |
| Do Kwon Sentencing | 31 | More crypto fraud cases |
| Venezuela Oil Tanker | 27 | More Venezuela crisis events |

These are **isolated events** in our dataset. They need sibling events from the same narrative to form frames.

### Hypothesis: Frame Emergence Requires Narrative Density

```
Frame emergence threshold:

  Events in same geographic/thematic region
  + Similar time period
  + Shared entities OR causal links
  ─────────────────────────────────
  = Frame candidate
```

Our 1215 claims cover **diverse topics** without enough density in any single narrative (except Hong Kong Fire).

### What Would Enable More Frames?

**With more data**, these frames could emerge:

```
[FRAME] "Hong Kong 2025"
  └─ Wang Fuk Court Fire (66 claims)  ← Already have
  └─ Jimmy Lai Trial (100 claims)     ← Have, but semantically distant
  └─ National Security Law cases      ← Need more
  └─ Press freedom incidents          ← Need more

[FRAME] "US Campus Safety 2025"
  └─ Brown University Shooting (39 claims) ← Have
  └─ Other university incidents            ← Need more

[FRAME] "Crypto/Financial Crime"
  └─ Do Kwon Sentencing (31 claims) ← Have
  └─ FTX developments               ← Need more
  └─ Other crypto fraud cases       ← Need more
```

### The Jimmy Lai Problem

Jimmy Lai (99 claims) and Wang Fuk Court Fire (66 claims) didn't merge into a "Hong Kong" frame because:

1. **Semantic distance**: Human rights trial vs building fire - very different content
2. **Entity overlap**: Different key entities (Jimmy Lai vs fire victims)
3. **LLM judgment**: "Same broader narrative?" → NO (correctly)

This is **correct behavior**! Forcing them into the same frame would create an incoherent "junk drawer."

### Frame Emergence is Working

The algorithm **correctly** identified:
- Wang Fuk Court Fire + Fire ignition details = same incident (merged)
- Jimmy Lai Trial vs Wang Fuk Court Fire = different narratives (not merged)

The lack of more frames is a **data density issue**, not an algorithm failure.

### Stopping Criteria Validated

The system naturally stopped at the right level:
- "Wang Fuk Court Fire Incident" = meaningful, bounded frame
- Would NOT merge further to "Hong Kong Events" unless more data creates semantic bridges

### Cost Analysis

```
Phase 1 (Claims → Sub-events): ~400 LLM calls
Phase 2 (Sub-events → Events): ~45 LLM calls
Phase 3 (Events → Frames): 2 LLM calls

Total: 450 LLM calls ≈ $0.05
```

Frame merging is cheap because there are fewer candidates at higher levels.

### Architecture Implications

```
Level 0→1: Many merges (claims to sub-events)
Level 1→2: Moderate merges (sub-events to events)
Level 2→3: Few merges (events to frames) ← Requires narrative density
Level 3→4: Very few (frames to meta-frames) ← Requires massive data
```

The pyramid narrows as we go up. This is expected.

### Next Steps to Test

1. **Cross-frame entity linking**: Jimmy Lai appears in both "Hong Kong Crackdown" and "US-China Relations" - can we detect this?

2. **Forced frame exploration**: What if we lower the threshold? Does it create junk frames?

3. **More data**: Add more Hong Kong-related claims to see if frames emerge.

### Conclusion

Frame emergence (Level 3) **works** but requires:
- Sufficient data density in a narrative
- Multiple events sharing geographic/thematic/causal links
- NOT just same location or time

The single emerged frame "Wang Fuk Court Fire Incident" correctly merged two related events about the same physical incident. Other events remained isolated because they lack narrative siblings in our dataset.

**The algorithm is conservative, which is correct.** Better to under-merge than create incoherent frames.

---

*Report generated 2025-12-17*
