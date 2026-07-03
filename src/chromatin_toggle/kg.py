"""Load the literature knowledge graph and expose it as tensors for the GNN
and as plain structures for the mechanistic oracle."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
import yaml

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


@dataclass
class KnowledgeGraph:
    node_ids: list[str]                      # ordered node names
    node_index: dict[str, int]               # name -> index
    node_type: list[str]                     # type per node (by index)
    type_ids: list[str]                      # ordered unique types
    type_index: dict[str, int]
    node_bias: torch.Tensor                  # [N] resting bias (oracle only)
    memory_nodes: list[str]
    gene_map: dict[str, str]                  # node name -> human gene symbol
    program_nodes: list[str]
    program_index: list[int]                 # indices of program nodes
    relations: list[str]                     # ordered relation names
    relation_sign: dict[str, int]
    # edges as (relation -> list of (src_idx, dst_idx, signed_weight))
    edges: list[tuple[int, str, int, float]] = field(default_factory=list)

    @property
    def num_nodes(self) -> int:
        return len(self.node_ids)

    @property
    def num_types(self) -> int:
        return len(self.type_ids)

    @property
    def num_relations(self) -> int:
        return len(self.relations)

    def dense_adjacency(self) -> torch.Tensor:
        """Return [R, N, N] tensor A where A[r, dst, src] = signed weight.

        This carries the oracle's mechanistic edge weights and signs. Use it for
        the mechanistic oracle ONLY -- NOT as GNN input, or the model is handed
        the label-generating function as a constant (see structural_adjacency).
        """
        N, R = self.num_nodes, self.num_relations
        A = torch.zeros(R, N, N)
        rel_idx = {r: i for i, r in enumerate(self.relations)}
        for src, rel, dst, sw in self.edges:
            A[rel_idx[rel], dst, src] += sw
        return A

    def structural_adjacency(self) -> torch.Tensor:
        """Return [R, N, N] binary tensor: 1 where a typed edge exists, else 0.

        This is what the GNN sees -- graph STRUCTURE only. Edge sign and
        magnitude are not exposed; the model must learn them per relation via
        its rel_lin transforms. Keeps the oracle's weights out of the model.
        """
        return (self.dense_adjacency() != 0).to(torch.float32)


def load_kg(path: str | Path | None = None) -> KnowledgeGraph:
    path = Path(path) if path else DATA_DIR / "kg.yaml"
    spec = yaml.safe_load(path.read_text())

    nodes = spec["nodes"]
    node_ids = [n["id"] for n in nodes]
    node_index = {name: i for i, name in enumerate(node_ids)}
    node_type = [n["type"] for n in nodes]
    node_bias = torch.tensor([float(n.get("bias", 0.0)) for n in nodes])

    type_ids = sorted(set(node_type))
    type_index = {t: i for i, t in enumerate(type_ids)}

    relation_sign = dict(spec["relation_signs"])
    relations = sorted(relation_sign.keys())

    edges: list[tuple[int, str, int, float]] = []
    for e in spec["edges"]:
        rel = e["rel"]
        sign = int(e.get("sign", relation_sign[rel]))
        w = float(e.get("w", 1.0)) * sign
        edges.append((node_index[e["src"]], rel, node_index[e["dst"]], w))

    program_nodes = spec["program_nodes"]
    program_index = [node_index[p] for p in program_nodes]

    return KnowledgeGraph(
        node_ids=node_ids,
        node_index=node_index,
        node_type=node_type,
        type_ids=type_ids,
        type_index=type_index,
        node_bias=node_bias,
        memory_nodes=spec["memory_nodes"],
        gene_map=dict(spec.get("gene_map", {})),
        program_nodes=program_nodes,
        program_index=program_index,
        relations=relations,
        relation_sign=relation_sign,
        edges=edges,
    )


def load_contexts(path: str | Path | None = None) -> dict[str, list[str]]:
    path = Path(path) if path else DATA_DIR / "contexts.yaml"
    return yaml.safe_load(path.read_text())["contexts"]
