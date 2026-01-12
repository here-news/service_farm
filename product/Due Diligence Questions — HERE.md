

# Due Diligence Questions — HERE.news

Dear Team HERE, 

After reviewing the whitepaper, business plan, unit economics, competitive analysis, product specs (homepage, live event page), and the working demo prototype.

  —

## I. Business Model & Unit Economics (1-10)

  1\. Your consumer event funding model shows negative margins at lower tiers (-20% to \-10%). You claim B2B subsidizes this, but what happens if B2B growth stalls? How long can you  sustain negative-margin consumer growth?

- In our model, the consumer “event funding” unit is the **inquiry sponsor pool**: sponsors fund task bounties and impact rewards, so variable costs can be matched to per-inquiry demand (vs. platform-wide subsidy).
- Our Year 1 sustainability model assumes early subsidy from seed funding (**~$200k**) while marketplace liquidity builds, with a break-even target of **~800k credits sponsored/month (~$8k/month)** covering fixed costs.
- If B2B/institutional growth stalls, we cap subsidies (e.g., reduce/limit starter credits) and gate expensive processing behind sponsorship/high-entropy gaps so the system degrades by showing more uncertainty—not by burning cash invisibly.

  2\. You project 100K active funders by Year 3 spending $4.50/month average. What's your evidence users will pay recurring fees for news events? Wikipedia and Reddit are free. Why won't  a free competitor emerge?

- We don’t depend on a pure recurring subscription; the primary consumer payment is **episodic credit purchase → sponsor specific inquiries** (“funding investigation”, not “paying for news”).
- Our Year 1 end model uses **~$30/user average credit purchase** for ~500 active buyers (≈ **$15k/month**), which supports the premise that a subset of users will pay when an inquiry matters to them.
- A free competitor can summarize, but our defensibility is the **two-loop system**: machine gap detection (meta-claims like `coverage_gap`, `single_source_only`, `unresolved_conflict`) + a market for epistemic labor (sponsors → tasks → measurable rewards) + full provenance + posterior/entropy. That combination compounds into a resolution library and reputation graph.

Two-loop flywheel:
```
                        ┌─────────────────────────────────┐
                        │      COMMUNITY LOOP             │
                        │                                 │
                        │ Sponsors → Tasks → Contributions │
                        │      ↑              │           │
                        │      │              ↓           │
                        │  Rewards ←── Impact ←── Work    │
                        └─────────┬───────────────────────┘
                                  │
                    ◄─────────────┼─────────────►
                    Contributions │ Tasks emerge
                    become Claims │ from Gaps
                                  │
                        ┌─────────┴───────────────────────┐
                        │      EPISTEMIC LOOP             │
                        │                                 │
                        │  Claims → Surfaces → Events     │
                        │      ↓              │           │
                        │   Identity      Aboutness       │
                        │   Edges ───────── Edges         │
                        │      ↓              ↓           │
                        │     Meta-claims (Gaps)          │
                        └─────────────────────────────────┘
```


  3\. Your B2B pricing ($99-$999/month) seems low for enterprise. Palantir charges $500K+. Are you leaving money on the table, or is this a land-and-expand strategy? What's your expansion revenue assumption?

- Our early “institutional” model is **subscriptions (~20 orgs × $500/month avg = ~$10k/month)** plus **API/research licenses (~$5k/month)**, consistent with design-partner pricing rather than Palantir-style deployments.
- The implied expansion path is **land-and-expand by usage and trust**: seats, claim volume, active inquiries, SLA/compliance, exports, and private graph space—priced as value proves out (case studies show investigations completed for **$450–$3k** vs. far higher traditional costs).
- We’re not publishing a single enterprise expansion multiplier yet; near-term we expect pilot pricing → tiering by volume/SLA as the resolution library and workflow become embedded.

  4\. Credit economics: if users earn rewards for contributions, doesn't this create inflation in the credit system? What prevents a whale from flooding events with low-value  
  contributions?

- Credits are explicitly pegged: **1 credit = $0.01**. Credits enter via purchase/sponsorship (and any platform subsidies), not via uncontrolled “minting”.
- Rewards are tied to **measurable epistemic impact**: bounties come from sponsor pools, and “impact rewards” are proportional to **entropy reduction** and conflict resolution. Low-value contributions reduce little/no entropy → earn little/no reward.
- “Whale spam” is structurally disincentivized: spamming increases volume but not posterior confidence, and manipulation shows up as visible uncertainty rather than silent capture; reputation weighting and cross-verification reduce the value of coordinated low-signal flooding.

  5\. LLM costs are $0.05-0.18 per page. If OpenAI raises prices 3x (which has happened before), does your unit economics model break? What's your fallback?

- We treat model costs as a variable input and **prioritize computation** based on where uncertainty (entropy) is highest and where sponsorship justifies spend.
- Fallbacks: multi-provider routing, use cheaper models for low-risk extraction, reserve expensive models for high-impact gaps, and longer-term fine-tune/self-host for parts of extraction/relationship labeling.

  6\. You say "marketing spend \= $0 until product loops are validated." What if viral loops never kick in? What's your CAC if you have to buy growth?

- We’re designing for non-viral acquisition channels intrinsic to the product: shareable “argument artifacts” (claim cards, confidence shifts, supersession receipts) and embeddable/live inquiry outputs.
- We don’t have a finalized CAC yet; the constraint is that paid CAC must be below expected gross profit per activated buyer (we model **~$30 average credit purchase** for active buyers, plus institutional subscriptions on the B2B side).
- If virality is weak, the go-to-market path is “expert-first”: recruit journalists/OSINT/policy analysts with bounties + reputation, then expand via the growing reference library and institutional distribution (subscriptions/API).

  7\. The two-sided economy (B2B profit \+ consumer volume) is theoretically elegant but operationally complex. How do you prevent product fragmentation serving two masters?

- We keep a single core product: **claims → surfaces → events → meta-claims → tasks → contributions**, with provenance and posterior/entropy as shared primitives.
- “B2B” is an access wrapper (subscriptions, API licenses, private spaces, SLA/compliance) on top of the same epistemic engine, rather than a separate product line with divergent ontology/UX.
- Roadmap discipline: ship core resolution primitives first (gap detection, tasking, rewards, provenance), then add specialized interfaces for institutional workflows without forking the underlying model.

  8\. Your break-even analysis shows you need 2 Enterprise \+ 5 Pro \+ 10 Starter customers. But you're also building complex consumer features. Isn't this premature for a pre-seed company?

- Our break-even framing is **transaction volume**, not customer-count tiers: fixed costs are modeled at **~$8k/month**, with break-even at **~800k credits sponsored/month (~$8k/month)**.
- Consumer features are part of the mechanism that produces value: community contributions are mobilized by the UI that surfaces gaps and sponsorship opportunities.
- Our sequencing: seed with a narrow set of high-signal inquiries, prove “sponsors → tasks → entropy reduction”, and use seed funding (**~$200k**) to bridge early liquidity until the marketplace becomes self-sustaining.

  9\. Consumer funding for events is novel. Do you have any evidence from surveys, landing page tests, or pilot users that people will actually pay to "keep events alive"?

- We have not yet run/published formal survey or landing-page test results; our thesis is grounded in mechanism design and analogous behaviors (donations, crowdfunding).
- The model is “pay to investigate”: users sponsor an inquiry, which auto-funds tasks and rewards; when an inquiry resolves, we can also return unspent credits back to sponsors (example split: **70% to contributors / 30% returned to sponsors**).
- The key retention/monetization hook is visible impact: sponsorship creates tasks; tasks create evidence; evidence shifts the posterior and reduces entropy in public.

  10\. Your 3-year projection shows $13.4M ARR. At what ARR do you become profitable, and how much capital do you need to get there?

- Our break-even model uses **fixed costs ~ $8k/month**, implying profitability at roughly **~$96k ARR** (assuming variable costs scale with sponsor-funded rewards).
- Our Year 1 end target is **~$30k/month revenue (~$360k/year)** with an operating margin of **~$7k/month**, and an early subsidy period bridged by **~$200k seed funding**.
- Longer-term capital requirements beyond that seed bridge (hiring, expansion, regulatory work) are not finalized here.

---

##   II. Technology & Product (11-20)

  11\. Your claim extraction relies on GPT-4. What's your accuracy rate? In the demo, I saw "36 killed" and "44 killed" coexist. How do you actually resolve which is true?

- We treat extraction errors and contradictions as expected input. The system surfaces them explicitly as high-entropy/high-dispersion surfaces rather than forcing a single number.
- Resolution happens through provenance-weighted posteriors and supersession: when higher-quality evidence arrives (e.g., an official release), it’s added as a new claim that supersedes prior claims, and the posterior/entropy update is visible to users.
- We don’t have a published end-to-end accuracy benchmark yet; our design goal is to avoid false certainty by making uncertainty measurable and auditable.

  12\. The "Resolved States" (φ⁰, φ±, φΩ) are philosophically interesting but how does the system actually determine when something is "resolved"? Is there human intervention?

- For us, “resolved” is an evidence state, not an editorial decree: the system tracks a posterior distribution and entropy; resolution corresponds to high posterior confidence + low residual entropy, with provenance attached.
- Humans are “in the loop” only as contributors of evidence. They don’t overwrite history: claims are immutable; changes occur via new claims + relationships (e.g., `supersedes`) and the trace remains visible.
- Practically, φ⁰/φ±/φΩ correspond to low/medium/high uncertainty regimes and remain reversible if new evidence reopens conflict.

  13\. Entity resolution across sources is notoriously hard ("President Xi" vs "Xi Jinping" vs "习近平"). Your demo showed good clustering, but what happens with adversarial inputs?

- We treat identity as probabilistic: entities and claims are linked with confidence, and uncertainty is allowed to persist until corroborated.
- Adversarial input is handled by provenance and reputation effects: low-quality sources don’t automatically dominate; manipulation attempts increase visible uncertainty rather than silently rewriting truth.
- Entity merges/splits should be reversible and traceable (same principle as claim supersession): prefer safe uncertainty over brittle hard merges.

  14\. The demo processed 1215 claims into \~40 L2 events. What's your precision/recall? How many events were wrongly merged or split?

- We don’t have a published precision/recall evaluation yet. The emphasis is on making clustering errors observable through provenance, conflict signals, and community correction workflows.
- The system treats merges/splits as reversible: if an event is over-merged, additional evidence and gap detection can trigger re-segmentation; if under-merged, shared anchors and corroboration edges enable convergence.
- Our evaluation plan is to benchmark clustering against labeled event sets and measure downstream outcomes (entropy reduction speed, conflict resolution accuracy); we will publish results as we collect them.

  15\. "Coherence" and "tension" are core metrics, but the formulas seem heuristic (internal\_corr \- external\_contra). Have you validated these mathematically correlate with actual event  
  quality?

- We ground “quality” in information theory + Bayesian updating: entropy and posterior confidence quantify uncertainty; conflict/meta-claims quantify tension; supersession traces quantify correction dynamics.
- We can show case-study behavior (entropy dropping as corroborating/official evidence arrives, and posteriors shifting visibly instead of premature closure), but we haven’t published a full metric-validation paper yet.
- Our plan is continuous calibration against historical events (did the posterior converge toward later-confirmed outcomes, and how quickly) and then tuning the metrics based on those results.

  16\. Your "canonical name vs byline" separation is smart, but the canonical name generation depends on LLM. What prevents hallucination in names? ("Hong Kong Fire" → "Hong Kong  
  Disaster" drift)

- Our constraint is that the canonical layer is anchored in evidence: events are built from “shared anchors” (high-signal entities/phrases) and traced claims, not free-form summaries.
- Canonical labels are treated as editable metadata with provenance: if a name drifts, it can be superseded/updated like any other derived artifact, without changing underlying claims.
- Practically, canonical naming should be constrained to extracted entities + stable templates, with the byline/title preserved separately to avoid rewriting source language.

  17\. The Knowledge Gaps detection in your demo extracts conflicting numbers. But real epistemic gaps are semantic (e.g., "cause unknown"). How do you detect non-numeric contradictions?

- Gaps are first-class outputs via meta-claims (e.g., `coverage_gap`, `single_source_only`, `unresolved_conflict`) and are not restricted to numeric dispersion.
- Semantic contradictions are represented by explicit relationships between claims (corroboration, contradiction, update/supersedes) combined with provenance; “cause unknown / no independent evidence” is surfaced as a gap that spawns tasks.

  18\. Event "thoughts" are generated by LLM. How do you prevent these from being misleading or biased? Who audits them?

- We treat “system notes” as derived from observable event state: what is high-entropy, what is single-source, what tasks are missing (e.g., “⚠️ no independent verification yet”).
- To prevent misleading outputs, “thoughts” should be constrained to state + citations (linking to the underlying claims/meta-claims) and clearly labeled as non-claims. Auditability comes from the same provenance trace: if a note is wrong, it can be traced and corrected/superseded.

  19\. The demo's timeline shows "Initial Reports → Developing → Current" but this is just positional splitting. How do you capture actual temporal progression across claims?

- Our temporal model is posterior evolution: as new claims arrive (often as updates/supersessions), the system recomputes confidence and displays how beliefs shift over time.
- Temporal ordering is supported by timestamps on sources/claims and by explicit relationships (`update`, `supersedes`) rather than purely by UI buckets.

    
  20\. Your architecture shows PostgreSQL \+ Neo4j. Why both? Graph databases have scaling challenges. Have you stress-tested at 1M claims?

- The ecosystem concept separates concerns: a relational store for transactional/community data and retrieval, and a graph for the claim/event ontology and relationship traversal.
- We designed the architecture to sustain ~100M claims before an infrastructural change; we have not yet published 1M-claim stress-test results.
- The system’s failure mode is designed to be epistemically safe: if scale limits are hit, it should degrade into visible uncertainty/latency, not silently produce confident-but-wrong merges.

---

##   III. Market & Competition (21-30)

  21\. Perplexity is raising billions and adding news features. They have distribution you don't. Why won't they just add claim-level provenance and kill you?

- Our moat isn’t “citations” but a coordination system: meta-claims detect what’s missing, tasks/bounties mobilize humans, and posteriors/entropy track epistemic state over time.
- A summarization product can add provenance, but replicating a two-sided market (sponsors → tasks → contributions → measurable rewards) plus a growing resolution library + reputation graph is a different product category with different incentives.

  22\. Ground News already does bias detection and source comparison. They have 1M+ users. What's your wedge against them?

- Ground News helps users compare sources; our wedge is resolution: explicit uncertainty (entropy/posterior), gap detection, tasking, and paid incentives for corrections.
- Instead of “choose a side,” HereNews keeps competing claims visible with confidence until evidence collapses uncertainty—and the update trace remains public.

  23\. Meltwater/Cision have $1B+ revenue and enterprise relationships. If they acquire a startup like you, isn't that your best exit? Why build independently?

- We’re open to M&A as an option, but we’re building independently because the reference library, provenance graph, and reputation economy compound into a durable platform as more inquiries are resolved.
- We’ll evaluate strategic outcomes after the core loops and marketplace liquidity are proven.

  24\. Your TAM is $15B but SAM is $2B. That's a 7x gap. Why is research-focused news intelligence only $2B when general news intelligence is $15B?

- We’re not relying on TAM/SAM estimates to drive near-term execution. The value proposition we’re proving is investigation cost reduction (10–100× cheaper in case-study economics) and early institutional revenue via subscriptions/API.

  25\. "Fact-checking doesn't scale" \- but neither does your model if every event needs human resolution. How is community resolution fundamentally more scalable than editorial?

- Our scaling claim is that machines do the triage: meta-claims surface where uncertainty is high (single-source, dispersion, conflicts). Humans spend time only where it matters.
- Community resolution scales via parallelized tasks funded by sponsors: instead of one editor doing sequential work, many contributors can fill independent gaps at once, with rewards tied to measurable impact (entropy reduction).
- Low-entropy events stabilize with minimal human input; high-entropy inquiries attract more sponsorship and therefore more investigative labor.

  26\. Wikipedia has a form of community-driven truth with reputation. They don't need credits. Why do you?

- We’re “not Wikipedia” in a few key ways: we’re real-time, machine-assisted via gap detection, and we create an explicit market for epistemic labor rather than relying on volunteer-only incentives.
- Credits make verification timely and sustainable: people can be paid to do document retrieval, OSINT analysis, and cross-verification now, rather than only after slow consensus formation.

  27\. Polymarket already has stake-weighted truth discovery via prediction markets. They have real money at stake. Why is your credit system better?

- We distinguish “funding investigation” from betting: sponsorship funds tasks and evidence gathering; rewards are paid for entropy reduction and conflict resolution, not for picking an outcome.
- The system supports non-binary questions (ranges, multiple hypotheses) and preserves uncertainty when evidence doesn’t justify closure—unlike winner-take-all markets.

  28\. The "data flywheel" and "reputation flywheel" you describe are classic moat claims. But they compound slowly. What's your defensibility in Year 1-2 before flywheels spin?

- Year 1–2 defensibility is the resolution library with provenance (a compounding public record of who claimed what, when, and why the posterior moved) plus a reputation graph tied to accuracy/impact.
- The loop creates workflow lock-in: sponsor-funded tasks coordinate investigators faster/cheaper than ad-hoc effort, and the platform becomes the default place to see the current epistemic state of a contested issue.

  29\. Your demo focused on the Hong Kong fire \- a clear event. How does your system handle ambiguous situations like "Is X a scandal?" or "Was the election fair?"

- We handle long-running contested questions (e.g., “lab leak”) by maintaining posteriors that shift over time rather than declaring closure.
- For ambiguous prompts (“scandal”, “fair”), the system decomposes into explicit sub-inquiries (what happened, what standards apply, what evidence exists), preserves multiple hypotheses, and rewards contributors for reducing uncertainty (even if the final output remains a probability distribution).

  30\. News consumption is declining among younger demographics. Are you building for a shrinking market?

- We’re targeting a shift from passive consumption to participatory sense-making: users engage via tasks, sponsorship, and visible impact on the epistemic state.
- The product is designed to be closer to “social + game + investigation” than a feed: curiosity, status, and rewards are the engagement primitives, not scrolling headlines.

---

##   IV. Team & Execution (31-35)

  31\. The whitepaper is authored by Isaac Mao and Andrés Salgado. Can you share with us the experience/lessons you built and scaled a consumer product before?

- We can share a full team/bio document separately.
- At the product level, our execution advantage is that the architecture is operationally explicit (loops, incentives, unit economics, case-study flows), which reduces ambiguity about what needs to be built and measured.

  32\. Your roadmap shows Phase 0-5 over 18 months. You're asking for $250K-500K pre-seed. Is this enough runway? What's your burn rate?

- We haven’t finalized burn rate and runway in this document.
- Our financial model assumes **~$200k** to bootstrap marketplace liquidity toward a modeled Month-9 break-even (fixed-cost baseline **~$8k/month**). Actual runway depends on hiring and infrastructure choices.

  33\. You need NLP/ML expertise, product design, infrastructure, and community management. How many FTEs do you have today? What's your hiring plan?

- We can share current FTE count and a hiring plan separately.
- Our critical early roles are: (1) extraction/graph engineering, (2) incentive design + abuse resistance, and (3) community/task operations (ensuring bounties and verification workflows function).

  34\. The philosophy behind HERE.news (Jaynesian probability, epistemic organisms) is sophisticated. But most users don't care about epistemics. Who translates this for mainstream  
  adoption?

- We translate the philosophy into mainstream UX primitives: “unverified”, “needs independent evidence”, confidence ranges, and “superseded” receipts.
- The “translator” is product/UX writing and design: keep the epistemic machinery under the hood while exposing only confidence/provenance affordances users can act on.

  35\. Your demo was impressive technically but rough around the edges. What's your iteration velocity? How quickly can you ship production features?

- We can share a concrete release cadence and MVP timeline separately.
- The minimum shippable core we’re prioritizing is: extraction → surfaces/events → meta-claims → tasks/bounties → contribution ingestion → posterior/entropy display. We prioritize loop-closure over broad UI polish early.

---

##   V. Go-to-Market & Growth (36-40)

  36\. You say "design partners first" but who are they? Do you have any signed LOIs or paid pilots?

- We’re still finalizing named design partners / LOIs / paid pilots.
- Our initial design-partner segments are journalism/research orgs that need provenance and faster verification, and institutions that can license API access to the resolution library.

  37\. The "Events That Need You" contribution model requires critical mass of users. How do you bootstrap when events have no contributors?

- We can bootstrap from near-zero users: ingestion creates initial claims and surfaces; meta-claims emit “gaps”; a small number of early sponsors seed the first bounties; the first contributors earn and build reputation.
- Early “liquidity” can be bootstrapped by the founding team + a small invited cohort (journalists/OSINT) taking the first tasks, while the system’s extraction workers continuously seed new surfaces from public sources.

  38\. Your viral mechanics assume content spreads. But controversial news is already viral \- people share sensational articles, not nuanced truth pages. What makes your content more  
  shareable?

- Money as incentives  
- Curiosity   
- HERE content is designed to be shared as **argument artifacts**—single claim cards, confidence shifts, conflict views, and “this was superseded” receipts.

  39\. You mention "Live event embeds" for bloggers. Have you talked to any bloggers/journalists? Would they actually embed your widgets?

- We haven’t published specific blogger/journalist interview results yet.
- Our “artifact” concept (shareable claim cards, confidence shifts, supersession receipts) is compatible with embeds, but adoption needs validation through design partners and pilots.

  40\. Consumer onboarding shows "You start with 100 credits." That's $1. Is that enough to hook someone? What's your activation strategy?

The $1 isn’t the hook—**agency is**. New users are immediately placed inside a live or contested inquiry and prompted to take a meaningful action (sponsor, signal importance, or contribute) within seconds. The activation moment is seeing their action produce visible impact: a task appears, a response arrives, confidence shifts. Many users earn their second credits before ever buying more, which dramatically increases conversion. The system hooks users by making participation consequential, not by dangling free money.  
   
---

##   VI. Legal, Regulatory & Ethics (41-45)

  41\. You archive third-party content and extract claims. What happens when publishers send DMCA takedowns?

- Our approach is provenance-first (trace claims to sources). A DMCA response should preserve the epistemic trace while complying on payload: keep URLs/metadata/claim structure as permitted, and remove stored content when required.
- We expect publisher partnerships to be compatible with the model because sponsored inquiries create demand for high-quality sources; licensing/revenue-share is possible, but not finalized here.

  42\. In the EU, GDPR governs personal data. If a claim contains someone's name and is later disputed, how do you handle "right to be forgotten"?

- We haven’t finalized GDPR/right-to-erasure procedures in this document.
- Our design principles are: provenance-first (claims point to sources), reversibility (supersession/updates rather than silent edits), and minimizing unnecessary personal data storage. Concrete handling requires jurisdiction-specific legal review and product policy (redaction/anonymization flows).

  43\. "Resolved states" sound authoritative. What happens when your system says φ⁰ (fully resolved) but is wrong? What's your liability?

- Again, no binary nor 100%  
- φ⁰ does not mean “true forever.” It means **resolved under current evidence**, with full provenance and confidence attached. All resolution states are explicitly reversible, and supersession is a first-class operation. If new evidence appears, the system reopens the inquiry, updates confidence, and records the change publicly. Because we do not present φ⁰ states as guarantees or advice, and because all conclusions are traceable and revisable, liability risk is closer to an indexed research archive than an authoritative publisher.  
- People will believe step by step is the nature of the system, not a final answer

  44\. You claim to resist "coordinated propaganda networks" via SSP. But sophisticated actors can evade detection. What happens when state actors target your platform?

- The system is designed to **degrade gracefully under attack**, not collapse. Coordinated actors can submit content, but influence requires sustained, high-quality contribution that reduces contradiction—something propaganda campaigns struggle to do. Anomalous patterns (burst coordination, circular sourcing, reputation inflation) raise risk scores and lower weight rather than triggering bans. In the worst case, attacks increase visible uncertainty instead of flipping outcomes, making manipulation legible rather than invisible. This shifts the problem from content moderation to resilience.  
- We also have trackable reputation system to give different weight 

  45\. The credit system with real-money backing could be classified as a security or gambling mechanism in some jurisdictions. Have you gotten legal review?

- We have not completed formal legal review yet.
- Structurally, we treat credits as payment rails for services (“fund investigation”, “pay for verification work”), not speculation: rewards are paid for evidence/entropy reduction, not for betting on outcomes. Designing credits as non-transferable and non-appreciating helps, but jurisdictional analysis is still required.

---

##   VII. Risks & Existential Challenges (46-50)

  46\. Your core assumption is that people care about truth. Social media has proven engagement \> truth. What if users just don't want what you're building?

- We don’t assume “truth-seeking altruism” alone; we assume engagement drivers: curiosity, status/reputation, and money (bounties and impact rewards).
- If most users don’t want to do epistemic work, the system can still succeed as a reference layer: a smaller set of motivated contributors produce a high-quality resolution library that others consume passively (analogous to how Wikipedia is written by few and read by many).

  47\. LLMs are improving rapidly. In 2 years, Claude/GPT might have native fact-checking. Does that commoditize your value proposition?

- We treat LLMs as components, not the moat. Even perfect “fact-checking” doesn’t provide the full system: provenance graphs, posteriors/entropy that evolve over time, incentive coordination, and reputation effects.
- Better LLMs help HereNews (cheaper extraction/relationship labeling) but do not replace the market + state layer that stores and coordinates evidence.

  48\. "Event organisms" is a compelling metaphor but complex UX. Have you user-tested this with non-technical users? Do they understand "coherence" and "tension"?

- We haven’t published user-testing results yet.
- We’ll avoid jargon in the UI: show “unverified”, “needs independent evidence”, confidence ranges, and “superseded” receipts. The organism metaphor can remain architectural while the UX uses plain language.

  49\. What's your single biggest technical risk that could cause the project to fail?

- The biggest technical risk is maintaining correct, scalable identity + contradiction tracking under volume and adversarial pressure (bad merges, noisy sources, coordinated manipulation).
- Mitigation is architectural: reversible operations (supersession/trace), confidence-weighted linkage, gap detection, and a system that degrades into visible uncertainty rather than silently producing confident errors.

  50\. If HERE.news fails, what would be the most likely cause: technology, market, team, funding, or timing?

- The existential risk is market/liquidity: without enough sponsors and contributors, the loop doesn’t close, and the reference library/reputation economy can’t compound.
- The second risk is execution/funding: reaching the early liquidity threshold may require sustained subsidy (we model ~\$200k) and disciplined product focus on loop-closure.

Happy Holiday\!

  \---  
  **3C AGI Partners Investment Committee**  
  **December 2025**
