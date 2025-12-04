# News Intelligence Pipeline - Testing & Evaluation Framework

## Purpose
Standard checklist for evaluating article processing quality through the complete pipeline, from extraction to event formation and entity resolution.

## Test Article Requirements
- Minimum 500 words
- Contains factual claims with attribution
- Has named entities (people, organizations, locations)
- Ideally covers a specific event with clear temporal structure
- Has known ground truth for validation

---

## Stage 1: Content Extraction ✅ (Fully Implemented)

### Checklist
- [ ] **Extraction Success**: Article content successfully extracted
- [ ] **Word Count**: Reasonable word count (>100 words)
- [ ] **Language Detection**: Correct language identified
- [ ] **Metadata Quality**: Title, author, publication date, thumbnail extracted
- [ ] **Metadata Confidence**: Score >0.7 (indicates most fields present)
- [ ] **Content Quality**: No truncation, paywall blocks, or garbage text
- [ ] **Status Transition**: `stub` → `extracted` correctly

### Quality Metrics
```
Word Count: {actual} (expected: >{threshold})
Language: {detected} (expected: {actual_language})
Metadata Confidence: {score}/1.0 (expected: >0.7)
Status: {status} (expected: 'extracted')
```

### Pass Criteria
- ✅ Word count >80% of expected
- ✅ Correct language detected
- ✅ Metadata confidence >0.6
- ✅ No extraction errors in logs

---

## Stage 2: Semantic Analysis ✅ (Fully Implemented)

### Checklist
- [ ] **Claims Extraction**: Claims successfully extracted (expected: 5-15 for typical news article)
- [ ] **Claims Not Empty**: System didn't return 0 claims for valid content
- [ ] **Entities Extraction**: Named entities extracted from claims
- [ ] **Embeddings Generated**: Page and claim embeddings created
- [ ] **Status Transition**: `extracted` → `semantic_complete`
- [ ] **Processing Time**: Reasonable duration (<60 seconds for <2000 words)

### Quality Metrics
```
Claims Extracted: {count}
Entities Extracted: {count} ({people} people, {orgs} orgs, {locations} locations)
Duration: {seconds}s
Status: {status}
```

### Pass Criteria
- ✅ Claims count >0 for news articles
- ✅ At least 1 entity extracted for typical news
- ✅ Embeddings generated successfully
- ✅ No JSON truncation errors

---

## Stage 3: Entity Quality 🟡 (Needs Improvement)

### 3A. Entity Extraction Quality

#### Checklist
- [ ] **Precision**: No false positive entities (entities that don't exist in text)
- [ ] **Recall**: Major entities from article are captured
- [ ] **Type Accuracy**: Entity types correctly classified (PERSON, ORG, LOCATION, GPE)
- [ ] **Deduplication**: Same entity not extracted multiple times with different names
- [ ] **Context Preservation**: Entity context (titles, roles) maintained

#### Quality Metrics
```
Entities Extracted: {total}
- PERSON: {count} (list: {names})
- ORG: {count} (list: {names})
- LOCATION: {count} (list: {names})
- GPE: {count} (list: {names})

Precision: {correct}/{extracted} = {percentage}%
Recall: {extracted}/{expected_major_entities} = {percentage}%
Type Accuracy: {correct_types}/{total} = {percentage}%
```

#### Manual Validation Required
For each entity, verify:
1. Entity actually appears in article text
2. Entity type is correct
3. Entity name is canonical (not abbreviated incorrectly)
4. Important entities weren't missed

#### Pass Criteria
- ✅ Precision ≥90% (max 1 false positive per 10 entities)
- ⚠️ Recall ≥70% (catches most major entities)
- ✅ Type Accuracy ≥90%

### 3B. Entity Enrichment (Wikidata) ⚠️ (Threshold Tuning Needed)

#### Checklist
- [ ] **Candidates Found**: Wikidata candidates retrieved for entities
- [ ] **Match Quality**: Candidates are plausible matches
- [ ] **Enrichment Success**: Entities successfully linked to Wikidata QIDs
- [ ] **Confidence Calibration**: Confidence scores reflect match quality
- [ ] **Common Name Handling**: System handles common names (e.g., "John Lee") appropriately

#### Quality Metrics
```
Entities Queued: {count}
Candidates Found: {count}/{queued}
Enriched (QID assigned): {count}/{found}
Below Threshold: {count} (threshold: 0.65)

Low Confidence Examples:
- {entity_name} → {qid} (confidence: {score})
```

#### Known Issues
- ⚠️ **Threshold Too Strict**: 0.65 threshold rejects valid matches for officials in specific contexts
- ⚠️ **Common Names**: Low confidence for common names without sufficient context
- ⚠️ **Missing Context**: Entity descriptions not always used for disambiguation

#### Pass Criteria (Current)
- ⚠️ Enrichment rate >30% (low due to strict threshold)
- ✅ No false positive QID assignments

#### Pass Criteria (Target - After Fixes)
- ✅ Enrichment rate >60% for named officials/organizations
- ✅ Context-aware confidence scoring
- ✅ Lower threshold (0.45-0.50) for government officials with titles

---

## Stage 4: Claims Quality ✅ (Fully Implemented)

### Checklist
- [ ] **Atomic Claims**: Each claim is a single, verifiable statement
- [ ] **Attribution**: Claims properly attributed (who said it)
- [ ] **Modality Correct**: Correct classification (observation/reported_speech/allegation)
- [ ] **Confidence Calibration**: Higher confidence for direct quotes vs. interpretations
- [ ] **Entity Linking**: Entities correctly linked to claims
- [ ] **Temporal Info**: Event times extracted when available
- [ ] **Factual Accuracy**: Claims match article content (no hallucinations)

### Quality Metrics
```
Total Claims: {count}
Modality Distribution:
- observation: {count}
- reported_speech: {count}
- allegation: {count}

Average Confidence: {score}
With Event Time: {count}/{total}
Entity Links: {count} total claim-entity pairs

Factual Accuracy: {correct}/{checked} = {percentage}% (manual check sample)
```

### Manual Validation (Sample 5-10 Claims)
For each sampled claim:
1. ✅ Claim text matches article content
2. ✅ Attribution is correct (if reported_speech)
3. ✅ Modality classification is appropriate
4. ✅ Entities linked are actually mentioned in claim
5. ✅ Confidence score is reasonable
6. ❌ Claim contains factual errors or hallucinations

### Pass Criteria
- ✅ Factual accuracy ≥95% (max 1 error per 20 claims)
- ✅ Attribution accuracy ≥95% for reported_speech
- ✅ Modality accuracy ≥90%
- ✅ Entity linking accuracy ≥95%
- ✅ No claims below 0.3 confidence

---

## Stage 5: Event Formation 🔴 (Needs Major Work)

### 5A. Event Structure Quality ✅ (Working)

#### Checklist
- [ ] **Event Created**: Event node created in Neo4j
- [ ] **Event Title**: Descriptive, clear event title
- [ ] **Event Type**: Correct event classification (FIRE, PROTEST, etc.)
- [ ] **Phases Created**: Event phases logically structured
- [ ] **Phase Naming**: Phase names are descriptive
- [ ] **Phase Coherence**: Claims within each phase are related
- [ ] **Temporal Flow**: Phases follow logical chronological order
- [ ] **Claim Distribution**: Claims reasonably distributed across phases

#### Quality Metrics
```
Event Created: {title}
Event ID: {id}
Event Type: {type}
Event Confidence: {score}

Phases: {count}
Phase Breakdown:
1. {phase_name}: {claim_count} claims
2. {phase_name}: {claim_count} claims
...

Claims Distribution: {balanced/imbalanced}
```

#### Pass Criteria (For Single Article)
- ✅ Event created with clear title
- ✅ Phases have descriptive names
- ✅ Claims logically grouped into phases

### 5B. Event Consolidation/Deduplication 🔴 (NOT IMPLEMENTED - CRITICAL GAP)

#### Current Problem
**Issue**: When multiple articles cover the SAME event, system creates DUPLICATE events instead of:
1. Recognizing they're the same event
2. Merging into single event
3. Or treating new article as amendment/update to existing event

#### Example (From Testing)
```
Article 1: HKFP - "You are not alone: Countries, officials..."
→ Created Event: "2025 Hong Kong Tai Po Fire"

Article 2: AP News - "Deadly fire raises fears about safety..."
→ Created Event: "2025 Hong Kong High-Rise Apartment Fire"  ❌ DUPLICATE

Expected: Should recognize as SAME event and consolidate/merge
```

#### What Needs to Be Built

**Event Deduplication Logic**:
1. **Event Similarity Detection**:
   - Compare event titles (semantic similarity)
   - Check entity overlap (same location, same entities)
   - Check temporal proximity (same date/time range)
   - Check event type (both FIRE)

2. **Consolidation Strategy**:
   ```
   if (same_event_detected):
       if (existing_event.confidence > 0.7):
           → Merge new claims into existing event
           → Add as new phase(s) if different angle
           → Update event confidence
       else:
           → Full re-analysis with combined claims
           → Regenerate phases with all information
   ```

3. **Cross-Article Entity Resolution**:
   - Same entities mentioned in multiple articles should link to same Neo4j node
   - Wikidata QIDs enable this (when enrichment works)

#### Evaluation Criteria (Once Implemented)

**For Multi-Article Event Coverage**:
- [ ] **Duplicate Detection**: System detects when 2+ articles cover same event
- [ ] **Consolidation Success**: Articles merged into single event
- [ ] **Phase Integration**: New article's claims integrated into existing phases or new phases created
- [ ] **Entity Consistency**: Same entities across articles link to same nodes
- [ ] **Timeline Coherence**: Combined event timeline makes sense
- [ ] **Confidence Update**: Event confidence increases with more sources

**Expected Metrics**:
```
Event: {consolidated_title}
Source Articles: {count}
- {article_1_title} ({claim_count} claims)
- {article_2_title} ({claim_count} claims)

Total Claims: {total}
Total Phases: {count}
Entity Consistency: {shared_entities}/{total_entities} entities shared

Consolidation Quality:
- Same event recognized: ✅
- Claims integrated: ✅
- Phases coherent: ✅
- No duplicate entities: ✅
```

#### Testing Protocol (Once Implemented)

**Test Case: Same Event, Different Sources**
1. Submit Article A about Event X
2. Wait for event formation
3. Submit Article B about same Event X (different source/angle)
4. Verify:
   - [ ] System detects they're the same event
   - [ ] Articles consolidated into single event
   - [ ] Entity nodes reused, not duplicated
   - [ ] Timeline/phases make sense
   - [ ] No contradiction in facts

**Test Case: Related But Different Events**
1. Submit Article A about Event X (e.g., "Fire incident")
2. Submit Article B about Event Y (e.g., "Fire aftermath protest")
3. Verify:
   - [ ] System creates two separate events
   - [ ] But links them as RELATED events
   - [ ] Shared entities properly linked
   - [ ] Temporal relationship captured

#### Current Status
**Status**: 🔴 NOT IMPLEMENTED

**Impact**:
- ❌ Duplicate events fragment the graph
- ❌ Cross-article entity resolution fails
- ❌ Timeline reconstruction incomplete
- ❌ Cannot track event evolution across sources

**Priority**: **CRITICAL** - Core functionality for multi-source intelligence

---

## Stage 6: Graph Integrity 🟡 (Partially Working)

### Checklist
- [ ] **Entities in Neo4j**: Entities stored in Neo4j graph
- [ ] **Claims in PostgreSQL**: Claims stored with embeddings
- [ ] **Dual-Write Success**: Data stored in both databases as designed
- [ ] **Entity Links**: Claim-entity relationships created
- [ ] **Event Links**: Event-claim relationships created
- [ ] **No Orphaned Nodes**: All entities linked to at least one claim
- [ ] **No Duplicate Entities**: Same entity not duplicated with different IDs

### Quality Metrics
```
Neo4j Entities: {count}
PostgreSQL Claims: {count}
Claim-Entity Links: {count}
Event-Claim Links: {count}

Orphaned Entities: {count} (expected: 0)
Duplicate Entity Check: {duplicates_found}
```

### Validation Queries

**Check entities in Neo4j**:
```cypher
MATCH (e:Entity)
WHERE e.updated_at > datetime() - duration('PT1H')
RETURN e.canonical_name, e.entity_type, e.mention_count, e.status
ORDER BY e.mention_count DESC
```

**Check for orphaned entities**:
```cypher
MATCH (e:Entity)
WHERE NOT (e)-[:MENTIONED_IN]->(:Claim)
AND e.created_at > datetime() - duration('PT1H')
RETURN e.canonical_name
```

**Check entity duplication**:
```cypher
MATCH (e1:Entity), (e2:Entity)
WHERE e1.canonical_name = e2.canonical_name
  AND e1.entity_type = e2.entity_type
  AND e1.id < e2.id
RETURN e1.canonical_name, e1.id, e2.id
```

---

## Stage 7: End-to-End Integration ✅ (Working)

### Checklist
- [ ] **Complete Pipeline**: Article flows through all stages
- [ ] **Worker Coordination**: Workers process in correct sequence
- [ ] **No Blocking Errors**: No worker crashes or hangs
- [ ] **Repository Pattern**: Zero direct database access in workers
- [ ] **Logging Quality**: Clear logs for each stage
- [ ] **Error Recovery**: Failed articles don't block queue
- [ ] **Processing Time**: Reasonable end-to-end duration

### Quality Metrics
```
Total Duration: {seconds}s
- Extraction: {seconds}s
- Semantic: {seconds}s
- Wikidata: {seconds}s
- Event: {seconds}s

Worker Errors: {count}
Queue Backlogs: {count}

Architecture Compliance:
- Direct DB access: {violations} (expected: 0)
- Repository usage: ✅
```

### Pass Criteria
- ✅ Complete end-to-end processing
- ✅ Zero direct SQL queries in worker code
- ✅ All workers use repository pattern
- ✅ Processing time <120s for <2000 word article

---

## Summary Testing Checklist

### Quick Validation (Per Article)
```
□ 1. Extraction: {word_count} words extracted, metadata complete
□ 2. Semantic: {claim_count} claims, {entity_count} entities
□ 3. Entities: Types correct, no false positives
□ 4. Claims: Factually accurate, properly attributed
□ 5. Event: Phases logical (ignore duplication for now)
□ 6. Graph: Data in Neo4j and PostgreSQL
□ 7. Logs: No errors or warnings

Quality Score: ___/100

Issues Found:
-
-
-
```

### Known Limitations (As of 2025-12-04)

**Working Well** ✅:
- Content extraction (newspaper3k)
- Semantic claim extraction (fixed max_tokens issue)
- Entity type classification
- Claims attribution and modality
- Phase structure within single article
- Repository pattern architecture

**Needs Improvement** 🟡:
- Entity recall (misses some organizations)
- Wikidata enrichment threshold (too strict at 0.65)
- Event confidence scoring (too low)

**Critical Gaps** 🔴:
- **Event deduplication/consolidation** - Same event from multiple sources creates duplicates
- **Cross-article entity resolution** - Entities not linked across articles
- **Event relationship detection** - Related events not linked

---

## Testing Protocol

### For Each New Article:

1. **Submit URL** and record `page_id`
2. **Monitor logs** for each worker
3. **Check PostgreSQL**:
   ```sql
   -- Final status
   SELECT status, word_count FROM core.pages WHERE id = '{page_id}';

   -- Claims
   SELECT COUNT(*) FROM core.claims WHERE page_id = '{page_id}';

   -- Events
   SELECT title, event_type FROM core.events
   WHERE id IN (SELECT event_id FROM core.page_events WHERE page_id = '{page_id}');
   ```

4. **Check Neo4j** (when authentication fixed):
   ```cypher
   MATCH (e:Entity)-[:MENTIONED_IN]->(c:Claim)-[:PART_OF]->(event:Event)
   WHERE c.page_id = $page_id
   RETURN e.canonical_name, count(c) as mentions
   ORDER BY mentions DESC
   ```

5. **Manual validation**:
   - Read original article
   - Verify 5 random claims against article text
   - Check 3 entities are correctly typed
   - Verify event title makes sense

6. **Document results** in testing log

---

## Testing Log Template

```markdown
## Test: {Article Title}

**URL**: {url}
**Page ID**: {page_id}
**Date**: {test_date}
**Tester**: {name}

### Results

**Extraction** ✅/❌
- Word count: {count}
- Language: {lang}
- Metadata: {score}/1.0

**Semantic** ✅/❌
- Claims: {count}
- Entities: {count}
- Duration: {seconds}s

**Entity Quality** ✅/⚠️/❌
- Precision: {percentage}%
- Recall: {percentage}%
- Enrichment: {count}/{total}

**Claims Quality** ✅/⚠️/❌
- Factual accuracy: {percentage}%
- Attribution: {percentage}%

**Event Formation** ✅/⚠️/❌
- Phases: {count}
- Duplication detected: Yes/No
- Consolidation needed: Yes/No

**Overall Score**: {score}/100

**Issues**:
1.
2.

**Recommendations**:
1.
2.
```

---

## Next Steps

1. **Immediate** (Can evaluate now):
   - Use this framework for all new article tests
   - Document results in testing log
   - Build regression test suite

2. **Short-term fixes** (<1 week):
   - Fix Wikidata threshold (lower to 0.45 for officials)
   - Improve entity recall
   - Add entity profile generation

3. **Critical features** (1-2 weeks):
   - **Implement event deduplication/consolidation**
   - Add event similarity detection
   - Cross-article entity merging
   - Event relationship detection (RELATED_TO, CAUSED_BY, etc.)

4. **Long-term** (>2 weeks):
   - Automated testing suite using this framework
   - Benchmark dataset with ground truth
   - A/B testing for LLM prompts
   - Confidence calibration studies

---

## Version
**Version**: 1.0
**Last Updated**: 2025-12-04
**Status**: Initial framework - Event consolidation not yet implemented
