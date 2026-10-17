"""Controlled visual mutation and deterministic defect analysis."""

from .mutations import MUTATION_TYPES, apply_mutation, revert_mutation

__all__ = ["MUTATION_TYPES", "apply_mutation", "revert_mutation"]
