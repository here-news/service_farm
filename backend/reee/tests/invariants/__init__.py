"""
Kernel Invariants
=================

Invariants that must always hold for the REEE kernel.

Safety Invariants:
- no_semantic_only_core: Core membership requires structural witness
- no_hub_story_definition: Hub entities cannot define stories
- no_chain_percolation: Chain-only edges never merge cores
- scoped_surface_isolation: (scope_id, question_key) uniquely identifies surfaces

Quantitative Bounds:
- story_count_range: Number of stories within expected bounds
- periphery_rate_range: Periphery attachment rate within bounds
- witness_scarcity_max: Witness scarcity below threshold
"""

from .kernel_invariants import (
    KernelInvariants,
    assert_no_semantic_only_core,
    assert_no_hub_story_definition,
    assert_scoped_surface_isolation,
    assert_core_leak_rate_zero,
)

from .quantitative_bounds import (
    assert_story_count_in_range,
    assert_periphery_rate_in_range,
    assert_witness_scarcity_below,
    assert_max_case_size_below,
)

__all__ = [
    'KernelInvariants',
    'assert_no_semantic_only_core',
    'assert_no_hub_story_definition',
    'assert_scoped_surface_isolation',
    'assert_core_leak_rate_zero',
    'assert_story_count_in_range',
    'assert_periphery_rate_in_range',
    'assert_witness_scarcity_below',
    'assert_max_case_size_below',
]
