"""
Calibrated Entropy Formula

Based on epistemic evaluation findings:
1. Our formula has a bug (using wrong relationship direction)
2. LLM assessment suggests different weights
3. Need to account for source independence

This experiment:
1. Fixes the entropy computation
2. Calibrates against LLM assessments
3. Incorporates source independence discount

Run inside container:
    docker exec herenews-app python /app/test_eu/calibrated_entropy.py
"""

import json
import os
import sys
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from scipy.optimize import minimize

sys.path.insert(0, '/app/backend')

from openai import OpenAI
from load_graph import load_snapshot, GraphSnapshot, ClaimData


# =============================================================================
# EMBEDDING SERVICE
# =============================================================================

class EmbeddingService:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.cache: Dict[str, List[float]] = {}
        self.model = "text-embedding-3-small"

    def get_embedding(self, text: str) -> List[float]:
        if text in self.cache:
            return self.cache[text]
        response = self.client.embeddings.create(model=self.model, input=text[:8000])
        embedding = response.data[0].embedding
        self.cache[text] = embedding
        return embedding

    def cosine_similarity(self, emb_a: List[float], emb_b: List[float]) -> float:
        a = np.array(emb_a)
        b = np.array(emb_b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# =============================================================================
# LLM ENTROPY ASSESSOR
# =============================================================================

class LLMAssessor:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    def assess_entropy(self, claim: str, corroborations: List[str], contradictions: List[str]) -> Dict:
        corr_str = "\n".join([f"  - {c[:100]}" for c in corroborations[:5]]) if corroborations else "  (none)"
        contra_str = "\n".join([f"  - {c[:100]}" for c in contradictions[:5]]) if contradictions else "  (none)"

        prompt = f"""Assess epistemic uncertainty for this claim:

CLAIM: {claim}

CORROBORATING CLAIMS ({len(corroborations)} total):
{corr_str}

CONTRADICTING CLAIMS ({len(contradictions)} total):
{contra_str}

Rate uncertainty from 0.0 (certain) to 1.0 (maximum uncertainty).
Consider: number of sources, quality of corroboration, presence of contradictions.

Respond in JSON:
{{
  "entropy": 0.0-1.0,
  "reasoning": "brief explanation"
}}"""

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1,
        )

        try:
            return json.loads(response.choices[0].message.content)
        except:
            return {"entropy": 1.0, "reasoning": "parse error"}


# =============================================================================
# ENTROPY FORMULAS
# =============================================================================

def entropy_v1_original(n_corr: int, n_contra: int) -> float:
    """Original formula (buggy - didn't account for direction)"""
    base = 1.0
    corr_reduction = 0.15 * np.log1p(n_corr) if n_corr > 0 else 0
    contra_addition = 0.1 * np.log1p(n_contra) if n_contra > 0 else 0
    return max(0.0, min(1.0, base - corr_reduction + contra_addition))


def entropy_v2_fixed(n_corr: int, n_contra: int) -> float:
    """Fixed formula with correct relationship direction"""
    base = 1.0

    # Corroboration reduces entropy more aggressively
    if n_corr > 0:
        # Each corroboration reduces by ~20% of remaining uncertainty
        corr_reduction = 0.20 * np.log1p(n_corr)
    else:
        corr_reduction = 0.0

    # Contradiction adds uncertainty
    if n_contra > 0:
        contra_addition = 0.15 * np.log1p(n_contra)
    else:
        contra_addition = 0.0

    return max(0.0, min(1.0, base - corr_reduction + contra_addition))


def entropy_v3_calibrated(n_corr: int, n_contra: int, params: Dict = None) -> float:
    """Calibrated formula with tunable parameters"""
    if params is None:
        params = {
            'base': 1.0,
            'corr_weight': 0.25,
            'contra_weight': 0.12,
            'corr_exponent': 0.8,
            'contra_exponent': 0.6,
        }

    base = params['base']

    # Corroboration with diminishing returns
    if n_corr > 0:
        corr_reduction = params['corr_weight'] * (n_corr ** params['corr_exponent'])
    else:
        corr_reduction = 0.0

    # Contradiction
    if n_contra > 0:
        contra_addition = params['contra_weight'] * (n_contra ** params['contra_exponent'])
    else:
        contra_addition = 0.0

    return max(0.0, min(1.0, base - corr_reduction + contra_addition))


def entropy_v4_independence_aware(
    n_corr: int,
    n_contra: int,
    independence_ratio: float,
    params: Dict = None
) -> float:
    """Formula that accounts for source independence"""
    if params is None:
        params = {
            'base': 1.0,
            'corr_weight': 0.30,
            'contra_weight': 0.15,
            'independence_bonus': 0.5,
        }

    base = params['base']

    # Effective corroboration = raw × independence ratio
    effective_corr = n_corr * independence_ratio

    if effective_corr > 0:
        # More reduction for independent sources
        corr_reduction = params['corr_weight'] * np.log1p(effective_corr)
        # Bonus for high independence
        corr_reduction *= (1 + params['independence_bonus'] * independence_ratio)
    else:
        corr_reduction = 0.0

    if n_contra > 0:
        contra_addition = params['contra_weight'] * np.log1p(n_contra)
    else:
        contra_addition = 0.0

    return max(0.0, min(1.0, base - corr_reduction + contra_addition))


# =============================================================================
# CALIBRATION
# =============================================================================

def collect_training_data(
    snapshot: GraphSnapshot,
    assessor: LLMAssessor,
    sample_size: int = 50
) -> List[Dict]:
    """Collect LLM assessments for calibration"""

    print("Collecting training data...")

    # Sample claims with various levels of corroboration
    all_claims = list(snapshot.claims.values())

    # Stratified sampling
    by_corr_count = defaultdict(list)
    for c in all_claims:
        n = len(c.corroborated_by_ids)
        by_corr_count[n].append(c)

    samples = []
    per_bucket = sample_size // len(by_corr_count) if by_corr_count else sample_size

    for count, claims in sorted(by_corr_count.items()):
        for claim in claims[:per_bucket]:
            samples.append(claim)

    samples = samples[:sample_size]

    training_data = []

    for i, claim in enumerate(samples):
        # Get corroborating claim texts
        corr_texts = []
        for cid in claim.corroborated_by_ids:
            if cid in snapshot.claims:
                corr_texts.append(snapshot.claims[cid].text)

        # Get contradicting claim texts
        contra_texts = []
        for cid in claim.contradicted_by_ids:
            if cid in snapshot.claims:
                contra_texts.append(snapshot.claims[cid].text)

        # Get LLM assessment
        assessment = assessor.assess_entropy(claim.text, corr_texts, contra_texts)

        training_data.append({
            'claim_id': claim.id,
            'claim_text': claim.text[:100],
            'n_corr': len(claim.corroborated_by_ids),
            'n_contra': len(claim.contradicted_by_ids),
            'llm_entropy': assessment.get('entropy', 1.0),
            'reasoning': assessment.get('reasoning', ''),
        })

        if (i + 1) % 10 == 0:
            print(f"  Assessed {i + 1}/{len(samples)} claims...")

    return training_data


def optimize_parameters(training_data: List[Dict]) -> Dict:
    """Optimize entropy formula parameters to match LLM assessments"""

    print("\nOptimizing parameters...")

    def loss_function(params):
        corr_weight, contra_weight, corr_exp, contra_exp = params

        param_dict = {
            'base': 1.0,
            'corr_weight': corr_weight,
            'contra_weight': contra_weight,
            'corr_exponent': corr_exp,
            'contra_exponent': contra_exp,
        }

        errors = []
        for item in training_data:
            computed = entropy_v3_calibrated(
                item['n_corr'],
                item['n_contra'],
                param_dict
            )
            errors.append((computed - item['llm_entropy']) ** 2)

        return np.mean(errors)

    # Initial guess
    x0 = [0.25, 0.12, 0.8, 0.6]

    # Bounds
    bounds = [
        (0.05, 0.5),   # corr_weight
        (0.05, 0.3),   # contra_weight
        (0.3, 1.5),    # corr_exponent
        (0.3, 1.5),    # contra_exponent
    ]

    result = minimize(loss_function, x0, bounds=bounds, method='L-BFGS-B')

    optimized = {
        'base': 1.0,
        'corr_weight': result.x[0],
        'contra_weight': result.x[1],
        'corr_exponent': result.x[2],
        'contra_exponent': result.x[3],
    }

    print(f"Optimized parameters: {optimized}")
    print(f"Final loss: {result.fun:.4f}")

    return optimized


def evaluate_formulas(training_data: List[Dict], optimized_params: Dict) -> Dict:
    """Compare different formula versions"""

    print("\nEvaluating formulas...")

    results = {
        'v1_original': {'errors': [], 'correlation': 0},
        'v2_fixed': {'errors': [], 'correlation': 0},
        'v3_calibrated': {'errors': [], 'correlation': 0},
    }

    for item in training_data:
        n_corr = item['n_corr']
        n_contra = item['n_contra']
        llm = item['llm_entropy']

        # V1: Original
        v1 = entropy_v1_original(n_corr, n_contra)
        results['v1_original']['errors'].append(abs(v1 - llm))

        # V2: Fixed
        v2 = entropy_v2_fixed(n_corr, n_contra)
        results['v2_fixed']['errors'].append(abs(v2 - llm))

        # V3: Calibrated
        v3 = entropy_v3_calibrated(n_corr, n_contra, optimized_params)
        results['v3_calibrated']['errors'].append(abs(v3 - llm))

    # Compute statistics
    for name, data in results.items():
        data['mae'] = np.mean(data['errors'])
        data['max_error'] = np.max(data['errors'])

        # Compute correlation
        computed = []
        llm_values = [item['llm_entropy'] for item in training_data]

        for item in training_data:
            if name == 'v1_original':
                computed.append(entropy_v1_original(item['n_corr'], item['n_contra']))
            elif name == 'v2_fixed':
                computed.append(entropy_v2_fixed(item['n_corr'], item['n_contra']))
            else:
                computed.append(entropy_v3_calibrated(item['n_corr'], item['n_contra'], optimized_params))

        if len(set(computed)) > 1 and len(set(llm_values)) > 1:
            data['correlation'] = np.corrcoef(computed, llm_values)[0, 1]
        else:
            data['correlation'] = 0

    return results


# =============================================================================
# INDEPENDENCE ESTIMATION
# =============================================================================

def estimate_independence(
    snapshot: GraphSnapshot,
    embedding_service: EmbeddingService,
    sample_size: int = 30
) -> Dict:
    """Estimate source independence for claims with multiple corroborations"""

    print("\nEstimating source independence...")

    claims_with_corr = [
        c for c in snapshot.claims.values()
        if len(c.corroborated_by_ids) >= 2
    ][:sample_size]

    independence_estimates = []

    for claim in claims_with_corr:
        corr_claims = [
            snapshot.claims[cid] for cid in claim.corroborated_by_ids
            if cid in snapshot.claims
        ]

        if len(corr_claims) < 2:
            continue

        # Get embeddings
        embeddings = [embedding_service.get_embedding(c.text) for c in corr_claims]

        # Compute pairwise similarity
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = embedding_service.cosine_similarity(embeddings[i], embeddings[j])
                similarities.append(sim)

        # High similarity = copying, low = independent
        avg_similarity = np.mean(similarities) if similarities else 0
        independence_ratio = 1 - avg_similarity  # Inverse of similarity

        independence_estimates.append({
            'claim_id': claim.id,
            'n_corr': len(corr_claims),
            'avg_similarity': avg_similarity,
            'independence_ratio': independence_ratio,
        })

    avg_independence = np.mean([e['independence_ratio'] for e in independence_estimates])
    print(f"Average independence ratio: {avg_independence:.3f}")

    return {
        'estimates': independence_estimates,
        'average': avg_independence,
    }


# =============================================================================
# FINAL FORMULA
# =============================================================================

def create_final_formula(optimized_params: Dict, avg_independence: float) -> callable:
    """Create the final calibrated entropy formula"""

    def final_entropy(
        n_corr: int,
        n_contra: int,
        independence_ratio: float = None
    ) -> float:
        """
        Final calibrated entropy formula.

        Args:
            n_corr: Number of corroborating claims
            n_contra: Number of contradicting claims
            independence_ratio: Ratio of independent sources (0-1), default uses average

        Returns:
            Entropy value (0 = certain, 1 = maximum uncertainty)
        """
        if independence_ratio is None:
            independence_ratio = avg_independence

        base = optimized_params['base']

        # Effective corroboration (discounted by independence)
        effective_corr = n_corr * independence_ratio

        if effective_corr > 0:
            corr_reduction = optimized_params['corr_weight'] * (effective_corr ** optimized_params['corr_exponent'])
        else:
            corr_reduction = 0.0

        if n_contra > 0:
            contra_addition = optimized_params['contra_weight'] * (n_contra ** optimized_params['contra_exponent'])
        else:
            contra_addition = 0.0

        return max(0.0, min(1.0, base - corr_reduction + contra_addition))

    return final_entropy


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("CALIBRATED ENTROPY FORMULA")
    print("=" * 70)

    # Initialize
    embedding_service = EmbeddingService()
    assessor = LLMAssessor()
    snapshot = load_snapshot()
    print(f"Loaded: {len(snapshot.claims)} claims")

    # Step 1: Collect training data
    training_data = collect_training_data(snapshot, assessor, sample_size=50)

    # Step 2: Optimize parameters
    optimized_params = optimize_parameters(training_data)

    # Step 3: Evaluate formulas
    evaluation = evaluate_formulas(training_data, optimized_params)

    print("\n" + "=" * 70)
    print("FORMULA COMPARISON")
    print("=" * 70)

    print("\n{:<20} {:>10} {:>10} {:>12}".format("Formula", "MAE", "Max Err", "Correlation"))
    print("-" * 55)
    for name, data in evaluation.items():
        corr_str = f"{data['correlation']:.3f}" if not np.isnan(data['correlation']) else "N/A"
        print("{:<20} {:>10.3f} {:>10.3f} {:>12}".format(
            name, data['mae'], data['max_error'], corr_str
        ))

    # Step 4: Estimate independence
    independence = estimate_independence(snapshot, embedding_service, sample_size=30)

    # Step 5: Create final formula
    final_entropy = create_final_formula(optimized_params, independence['average'])

    # Test final formula
    print("\n" + "=" * 70)
    print("FINAL FORMULA DEMONSTRATION")
    print("=" * 70)

    test_cases = [
        (0, 0, "No evidence"),
        (1, 0, "1 corroboration"),
        (3, 0, "3 corroborations"),
        (5, 0, "5 corroborations"),
        (10, 0, "10 corroborations"),
        (3, 1, "3 corr, 1 contra"),
        (3, 3, "3 corr, 3 contra"),
        (0, 2, "Only contradictions"),
    ]

    print("\n{:<25} {:>15} {:>15}".format("Case", "Default Indep", "High Indep"))
    print("-" * 55)
    for n_corr, n_contra, description in test_cases:
        default = final_entropy(n_corr, n_contra)
        high_indep = final_entropy(n_corr, n_contra, independence_ratio=0.9)
        print("{:<25} {:>15.3f} {:>15.3f}".format(description, default, high_indep))

    # Summary
    print("\n" + "=" * 70)
    print("CALIBRATION SUMMARY")
    print("=" * 70)

    print(f"""
OPTIMIZED PARAMETERS:
  base: {optimized_params['base']:.3f}
  corr_weight: {optimized_params['corr_weight']:.3f}
  contra_weight: {optimized_params['contra_weight']:.3f}
  corr_exponent: {optimized_params['corr_exponent']:.3f}
  contra_exponent: {optimized_params['contra_exponent']:.3f}

AVERAGE INDEPENDENCE RATIO: {independence['average']:.3f}
  → {(1 - independence['average']) * 100:.1f}% of sources are likely copies

FORMULA PERFORMANCE:
  - Original (v1): MAE = {evaluation['v1_original']['mae']:.3f}
  - Fixed (v2): MAE = {evaluation['v2_fixed']['mae']:.3f}
  - Calibrated (v3): MAE = {evaluation['v3_calibrated']['mae']:.3f}

FINAL FORMULA:
  H = 1.0 - (corr_weight × (n_corr × independence_ratio)^corr_exp) + (contra_weight × n_contra^contra_exp)
""")

    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'optimized_params': optimized_params,
        'independence': {
            'average': independence['average'],
            'sample_size': len(independence['estimates']),
        },
        'evaluation': {k: {kk: vv for kk, vv in v.items() if kk != 'errors'} for k, v in evaluation.items()},
        'training_data_size': len(training_data),
    }

    output_path = Path("/app/test_eu/results/calibrated_entropy_results.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
