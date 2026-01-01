# MVP Experiment Suite

This directory contains comprehensive experiments validating all inquiry logic and data flows for the REEE-based Inquiry MVP.

## Running All Experiments

```bash
# Run from inside Docker container
docker exec herenews-app python scripts/experiments/mvp_experiments.py

# Run specific experiment (1-7)
docker exec herenews-app python scripts/experiments/mvp_experiments.py --exp=1
```

## Experiment Files

### Core MVP Logic

| File | Description | Status |
|------|-------------|--------|
| `mvp_experiments.py` | Full suite: schemas, conflicts, resolution, bounties | ✅ PASS |
| `forecast_inquiry.py` | Prediction inquiries with indicators and deadlines | ✅ PASS |
| `scope_homonym_test.py` | Entity disambiguation and scope contamination | ✅ PASS |
| `api_integration_test.py` | REST API endpoints end-to-end | ✅ PASS |

### Real Data Scripts

| File | Description |
|------|-------------|
| `../emulate_real_inquiry.py` | Full lifecycle with real Hong Kong fire claims |
| `../find_fire_claims.py` | Query DB for actual death toll claims |

## Experiments Summary

### 1. Multi-Schema Inquiries ✅
Tests different inquiry types work correctly:
- **Monotone count** (death tolls, sales figures)
- **Boolean** (yes/no questions)
- **Categorical** (verdicts, statuses)

### 2. Conflicting Evidence ✅
Tests belief state behavior with contradictory reports:
- First source reports 12
- Second source reports 25
- Third source corroborates with lower bound ≥23
- MAP correctly converges to 25

### 3. Attribution Chains ✅
Tests tracking of source attribution:
- Blog → Reuters → Original source
- Attribution contribution records chain
- *Gap: weight modification not yet implemented*

### 4. Scope Validation ✅
Tests cross-inquiry contamination prevention:
- Fire A (Wang Fuk Court) vs Fire B (Warehouse)
- Scope entities prevent automatic mixing
- Scope correction contributions tracked

### 5. Resolution Lifecycle ✅
Tests resolution criteria:
- P(MAP) ≥ 95% threshold
- Stability tracking starts
- Blocking tasks prevent resolution
- 24h stability period required

### 6. Bounty Mechanics ✅
Tests stake/bounty handling:
- Multi-user stakes accumulate
- Task bounties computed from pool
- Task generation from meta-claims

### 7. Real DB Integration ✅
Tests with actual claims from Neo4j:
- Loads Wang Fuk Court fire claims
- Extracts death toll values
- MAP converges to 160

### 8. Forecast Inquiries ✅
Tests prediction markets:
- GPT-5 release forecast example
- Indicator types (hiring, product, statement)
- Likelihood ratios for signals
- Deadline-based resolution tasks

### 9. Report-Truth vs World-Truth ✅
Tests epistemological distinction:
- "Did Reuters report X?" → High certainty (link exists)
- "Is X actually true?" → Lower certainty (conflicting evidence)

### 10. Scope Contamination ✅
Tests critical edge cases:
- Two fires in Hong Kong (different buildings)
- Entity homonyms (two Charlie Kirks)
- Overlapping events (shooting vs memorial)

### 11. API Integration ✅
Tests REST endpoints:
- Create inquiry
- List/get inquiries
- Add contributions
- Add stake
- Get trace
- Get tasks

## Known Gaps (for Production)

| Gap | Description | Priority |
|-----|-------------|----------|
| Attribution weight | Attribution doesn't modify evidence credibility | Medium |
| Scope enforcement | Not enforced at claim validation level | High |
| Background resolver | 24h stability requires periodic checker | High |
| Task completion | Claiming/payout needs user system | Medium |
| Deduplication | Same claim not merged across contributions | Medium |
| Surface formation | Claims not clustered during inquiry | Medium |
| Forecast calibration | Track resolved forecasts for reliability | Low |

## Typed Belief Fix Applied

Fixed `_update_monotone_bounds()` in `typed_belief.py` to correctly propagate hard bounds from both LOWER_BOUND and POINT observations for monotone count domains.

**Before fix:** MAP converging to 129 instead of 160
**After fix:** MAP correctly converges to 160 (actual death toll)

## Test Output Example

```
EXPERIMENT RESULTS SUMMARY
  Total: 7
  Passed: 7
  Failed: 0

  ✓ Multi-Schema Inquiries
  ✓ Conflicting Evidence
  ✓ Attribution Chains
  ✓ Scope Validation
  ✓ Resolution Lifecycle
  ✓ Bounty Mechanics
  ✓ Real DB Integration
```
