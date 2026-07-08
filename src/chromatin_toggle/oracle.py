"""Shared program-label helpers.

The project trains only on REAL, experimentally-labeled single-cell data. The
mechanistic KG-propagation oracle that used to GENERATE synthetic bootstrap labels
has been removed -- nothing is trained on synthetic labels. Only the shared label
constants remain (the program class list + the Quiescent baseline name).
"""
from __future__ import annotations

from .kg import KnowledgeGraph

QUIESCENT = "Quiescent"


def all_classes(kg: KnowledgeGraph) -> list[str]:
    return list(kg.program_nodes) + [QUIESCENT]
