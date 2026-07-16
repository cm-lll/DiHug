"""Degree-conditioned typed graphlet energy models.

This package is deliberately independent from the legacy diffusion sampler.
It provides the mathematically checkable gates used before a neural diffusion
model is allowed to participate in graph construction.
"""

from .checkpoint import StructureEnergyCheckpoint
from .features import GraphletFeatureExtractor
from .factorized_edit_denoiser import FactorizedEditDenoiser
from .graph import FamilySpec, TypedGraph, load_typed_graph
from .global_context_joint_edit import GlobalContextJointEditResidual
from .mixed_edit_diffusion import MarginalGraphPrior, MixedKernelSchedule
from .markov_diagnostics import (
    audit_state_summary_aliasing,
    run_forward_coupling_diagnostic,
    run_free_proposal_coverage_diagnostic,
)
from .roles import RolePlan, realize_role_plan, sample_role_plan
from .switch_diffusion import SwitchDenoiserMLP

__all__ = [
    "FamilySpec",
    "FactorizedEditDenoiser",
    "GraphletFeatureExtractor",
    "GlobalContextJointEditResidual",
    "MarginalGraphPrior",
    "MixedKernelSchedule",
    "RolePlan",
    "StructureEnergyCheckpoint",
    "TypedGraph",
    "SwitchDenoiserMLP",
    "load_typed_graph",
    "audit_state_summary_aliasing",
    "realize_role_plan",
    "run_forward_coupling_diagnostic",
    "run_free_proposal_coverage_diagnostic",
    "sample_role_plan",
]
