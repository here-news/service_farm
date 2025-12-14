# Business Plan

Related:
- `docs/vision.md`
- `docs/business.roadmap.md`
- `docs/architecture.principles.md`

## Executive Summary

**HERE.news** is building a **Breathing Knowledge System** (powered by the EVENT protocol) that turns sources (URLs, documents, media) into **claim-grounded events** that update as reality changes. The system is fast (instant best-shot), evidence-first (citations per claim), and designed for evolving truth (confidence + contradictions + resolution).

**Initial wedge (B2B)**: analysts and research desks who need reliable event timelines, provenance, and alerts.

**Expansion**: community resolution (credits + reputation), premium “chat with an event”, and credit-backed archiving.

## Problem

Teams tracking real-world developments face:
- fragmented sources, duplicated effort, and slow synthesis,
- unclear provenance (“where did this claim come from?”),
- low trust in summaries when contradictions exist,
- weak tooling for continuous monitoring and change detection.

## Product

### Core Loop (MVP)
1. User submits an artifact (URL, file, media, or data payload).
2. System returns an instant best-shot artifact and starts background enrichment.
3. Workers extract content → entities → claims → events.
4. UI shows event timeline, claims, sources, and confidence.

### Differentiators (From Current Issue Threads)
- **Fractal events**: events branch into phases/aspects and can emerge into higher-order narratives (issues #3, #15).
- **Epistemic hardening**: reliability, observability, and explicit uncertainty (issues #9, #4).
- **Trust signals**: dynamic reputations for publishers/authors/users feed confidence (issue #17).
- **Community resolution**: debates + evidence submissions become Bayesian priors (issues #13, #12).
- **Premium engagement**: “chat with an event” grounded in claims (issue #16).
- **Persistence**: credit-backed archiving and storage health transparency (issue #14).

## Target Customers (ICP)

**Primary (B2B)**:
- Newsrooms and research desks (fact-checking, investigations)
- Policy / think-tank analysts (issue monitoring, event timelines)
- Risk & compliance teams (entity/event monitoring, change detection)

**Secondary**:
- OSINT / intelligence contractors
- Market research and PR monitoring

## Go-To-Market (High-Level)

1. **Design partners**: 3–5 teams in one vertical; ship weekly; measure time-to-understanding.
2. **Land with workflows**: saved queries, watchlists, exports, and provenance UI.
3. **Expand within accounts**: alerts, collaboration, audit trails, and enterprise controls.
4. **Monetize power features**: API/export tiering; premium chat; archival/persistence tiers.

## Pricing & Packaging (Draft)

**B2B subscriptions** (seat-based + usage):
- Base seats include dashboards, timelines, and exports.
- Usage metering on ingestion volume, LLM-heavy actions, and alert delivery.

**Credits** (cross-cutting currency):
- used for community actions (votes/debates/evidence/reporting),
- used for premium experiences (event chat),
- used for persistence (archive backing).

## Operations (Overview)

### Product & Engineering
- Reliability-first pipeline: transactional outbox, DLQs, replay, idempotency (issue #9).
- Versioned contracts between workers; schema evolution discipline.
- Evaluation harness for event formation quality (precision/recall + human rubric).

### Data & Quality
- Source independence and diversity signals to prevent “single outlet dominance” (issue #9).
- Cross-lingual entity resolution to prevent duplicate events (issue #2).
- Time reasoning with timezone + uncertainty propagation (issue #4).

### Trust & Safety / Community Ops
- Reputation + credit-weighted participation; anti-sybil rules; audit logs (issues #17, #12, #13).
- Moderation workflows for abuse, takedowns, and disputes.

### Infra & Security
- Observability: tracing across API → workers → topology; SLOs and incident response (issue #9).
- Enterprise-ready posture: audit logs, access controls, retention policies.

### Customer Success
- Onboarding playbooks per ICP (watchlists, exports, alert tuning).
- Feedback loop: weekly review with design partners until retention targets are met.

## Key Metrics

**North Star (choose one)**:
- Verified events delivered per active account per week
- Time-to-understanding per tracked topic

**Supporting**:
- Ingestion success rate and latency by stage
- Event merge precision/recall and update-detection accuracy
- Citation coverage per claim/event
- Alert precision + latency
- D30 retention (pilots) and expansion within accounts

## Top Risks

- Extraction brittleness and missing coverage → multiple extractors + corpus testing.
- Hallucinated claims → strict claim grounding, refusal modes, conservative confidence defaults.
- Community gaming → credit gating + reputation + audits + rate limits.
- Cost blowups → caching, batching, model tiering, budgets, and observability.
- Legal/abuse (archiving, takedowns) → clear policies and operational runbooks.

## Decisions Needed

1. Pick the first ICP/vertical and success rubric.
2. Confirm database stance: PostgreSQL-first; any “graph index” must remain rebuildable.
3. Choose initial monetization wedge: dashboards/alerts vs API/exports vs premium chat.
4. Set credit policies: pricing table, stake outcomes, and curator selection.
