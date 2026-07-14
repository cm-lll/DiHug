"""Degree-conditioned typed graphlet energy models.

This package is deliberately independent from the legacy diffusion sampler.
It provides the mathematically checkable gates used before a neural diffusion
model is allowed to participate in graph construction.
"""

from .checkpoint import StructureEnergyCheckpoint
from .features import GraphletFeatureExtractor
from .graph import FamilySpec, TypedGraph, load_typed_graph
from .roles import RolePlan, realize_role_plan, sample_role_plan
from .switch_diffusion import SwitchDenoiserMLP

__all__ = [
    "FamilySpec",
    "GraphletFeatureExtractor",
    "RolePlan",
    "StructureEnergyCheckpoint",
    "TypedGraph",
    "SwitchDenoiserMLP",
    "load_typed_graph",
    "realize_role_plan",
    "sample_role_plan",
]
