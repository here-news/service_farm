# 02 — Epistemic Analysis: Why HERE.news, Why Now

Related:
- `docs/vision.md`
- `docs/architecture.principles.md`
- `docs/business.plan.md`
- `docs/business.roadmap.md`
- Protocol framing: `docs/protocol.event.md`
- Deep dive: `docs/philosophy.theory-topology.md` (Jaynes-style plausibility, priors/posteriors, MaxEnt/coherence, calibration gaps, and concrete hardening steps)

## The Context: Broken Epistemics

Modern society is running on an information substrate that is increasingly hostile to shared reality:
- Information volume grows faster than human attention and institutional review capacity.
- Incentives reward virality, identity signaling, and speed over accuracy and provenance.
- Media fragmentation and algorithmic feeds reduce shared context and increase polarization.
- Synthetic media and LLM-generated text lower the cost of persuasive falsehoods.
- Institutions (governments, platforms, newsrooms, academia) are overloaded and distrusted.

The result is not merely “misinformation”; it is **epistemic degradation**: a collapse of the practical ability to answer “what happened?” with confidence, at scale, under time pressure.

## Why Wikipedia-Style Knowledge Is Not Enough

Static encyclopedic models work for relatively stable facts. They struggle in the environment that matters most now:
- fast-moving events (conflicts, disasters, elections, pandemics),
- claims that evolve (death tolls, policy changes, investigations),
- contested narratives and coordinated manipulation,
- multi-lingual and cross-platform evidence trails.

In these domains, “a page” is the wrong primitive. We need a system that treats knowledge as a **living belief state** that updates with new evidence, tracks uncertainty, and can be audited.

## HERE.news: A Breathing Knowledge System

HERE.news is built around three commitments:

1. **Evidence-first**: every claim is anchored to sources; provenance is the default interface.
2. **Evolving confidence**: confidence is not a vibe; it is computed, updated, and explained as evidence changes.
3. **Living structure**: events, entities, and relationships grow over time (branching into phases/aspects; emerging into higher-order narratives).

This is the practical alternative to “trust me” summaries: a system where the user can inspect the evidence trail and see what changed.

## Epistemic Safety as a Product Requirement

If the system becomes an amplifier of error, it fails its mission. “Epistemic safety” means:
- conservative defaults (unknown is allowed; refusal is allowed),
- explicit contradiction surfacing instead of smoothing it away,
- source diversity and independence as first-class signals,
- auditability of major belief updates (why did confidence change? which evidence moved it?).

These are not academic concerns; they are survival constraints in domains where decisions affect lives.

## Legacy + Dynamic Engagement (Intergenerational Value)

Most systems optimize for the present moment: feeds, reactions, and churn. Future generations need something different:

### 1) Legacy (Durable Evidence)
Human memory is fragile, and links rot. For history to be learnable:
- sources must be preserved with integrity (hashing, availability health),
- provenance must remain attached to claims,
- major revisions must be traceable (what changed, when, and why).

This is why archiving is not a side feature; it is part of epistemic infrastructure.

### 2) Dynamic Engagement (Living Understanding)
Reality evolves. A future-proof knowledge system must:
- incorporate new evidence without erasing past states,
- represent uncertainty rather than collapse it into a single narrative,
- support structured disagreement and resolution (debates, evidence submission, review thresholds).

The goal is not unanimity; it is **legible disagreement** and **auditable convergence**.

## Human Judgment as Signal (Without Becoming a Mob)

The internet proved that pure “engagement” does not converge to truth. HERE.news treats human input as signal only when it is:
- accountable (identity/reputation),
- costly enough to resist spam (credits/stakes),
- traceable (audit trail),
- and grounded (evidence submission pathways).

This creates a channel for collective intelligence while resisting the failure modes of pure popularity.

## Why This Matters for Human Existence

Many of the highest-impact risks of the 21st century are **coordination problems** under uncertainty:
- conflict escalation and propaganda cycles,
- climate and infrastructure failures,
- biosecurity and public health response,
- AI governance and misuse,
- economic instability and social fracture.

In these regimes, epistemic degradation is not a cultural nuisance; it is a material hazard. Societies that cannot reliably perceive reality cannot reliably coordinate to survive.

HERE.news is an attempt to build epistemic infrastructure: a system that helps communities and institutions maintain contact with evidence, track uncertainty, and update beliefs over time without erasing history.

## What Success Looks Like

- Users can answer “what happened?” with a navigable chain of claims and sources.
- Contradictions are visible and resolvable, not hidden.
- Trust signals (publisher/author/user reputation) become measurable and correctable.
- Archived evidence survives link rot and preserves a readable timeline for future learners.

Next: `docs/how-it-works.md`
