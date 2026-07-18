"""Prior-constrained latent attractor model for paired multiome time courses.

This module implements the computational claims that can be supported by the
Attractor State manuscript:

1. ATAC accessibility estimates which regulatory regions are open.
2. Motif/occupancy priors constrain which transcription factors can bind at
   gene-linked open regions.
3. A validated TF circuit constrains the mechanistic part of the latent vector
   field.
4. A neural residual captures dynamics not represented by the prior graph.
5. Fixed points and local Jacobian eigenvalues operationalize "attractors."

Important data rule: RNA, cell-type labels, pseudotime, and target-state labels
are supervision/evaluation variables, not encoder inputs. Real transition
claims require longitudinal, lineage-traced, metabolic-labeling, or
perturbation-resolved data. A single snapshot supports cross-modal state
inference but not learned temporal dynamics.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


Tensor = torch.Tensor


@dataclass(frozen=True)
class PriorMatrices:
    """Biological constraints compiled without using held-out outcomes.

    peak_to_gene:
        [num_peaks, num_genes] genomic/regulatory links. Values may be binary
        or confidence weighted.
    motif_tf_gene:
        [num_tfs, num_genes] motif or occupancy support at regions linked to
        each gene. This is the protein-DNA binding feasibility layer.
    circuit_tf_tf:
        [num_tfs, num_tfs] validated directed TF interactions. Positive and
        negative entries encode activating and repressing edges; zero denotes
        no supported edge. Absolute values may encode evidence confidence.
    """

    peak_to_gene: Tensor
    motif_tf_gene: Tensor
    circuit_tf_tf: Tensor

    def validate(self) -> None:
        p2g, mtg, ctt = self.peak_to_gene, self.motif_tf_gene, self.circuit_tf_tf
        if p2g.ndim != 2 or mtg.ndim != 2 or ctt.ndim != 2:
            raise ValueError("All prior matrices must be rank two.")
        num_genes = p2g.shape[1]
        num_tfs = mtg.shape[0]
        if mtg.shape[1] != num_genes:
            raise ValueError("peak_to_gene and motif_tf_gene disagree on genes.")
        if ctt.shape != (num_tfs, num_tfs):
            raise ValueError("circuit_tf_tf must be square with one row per TF.")
        for name, value in (("peak_to_gene", p2g), ("motif_tf_gene", mtg)):
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} contains non-finite values.")
            if (value < 0).any():
                raise ValueError(f"{name} must be non-negative.")
        if not torch.isfinite(ctt).all():
            raise ValueError("circuit_tf_tf contains non-finite values.")


class HybridVectorField(nn.Module):
    """Validated circuit dynamics plus a bounded neural residual."""

    def __init__(self, num_tfs: int, cue_dim: int, circuit_mask: Tensor):
        super().__init__()
        sign = torch.sign(circuit_mask)
        confidence = circuit_mask.abs()
        confidence = confidence / confidence.amax().clamp_min(1.0)
        self.register_buffer("circuit_sign", sign)
        self.register_buffer("circuit_confidence", confidence)
        self.circuit_magnitude_unconstrained = nn.Parameter(
            torch.full((num_tfs, num_tfs), -2.5)
        )

        self.decay_unconstrained = nn.Parameter(torch.zeros(num_tfs))
        self.residual = nn.Sequential(
            nn.Linear(num_tfs + cue_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, num_tfs),
            nn.Tanh(),
        )
        self.residual_scale = nn.Parameter(torch.tensor(-2.0))

    def forward(self, z: Tensor, cues: Optional[Tensor] = None) -> Tensor:
        if cues is None:
            cues = z.new_zeros((z.shape[0], 0))
        circuit_weight = (
            self.circuit_sign
            * self.circuit_confidence
            * F.softplus(self.circuit_magnitude_unconstrained)
        )
        mechanistic = torch.tanh(z @ circuit_weight)
        decay = F.softplus(self.decay_unconstrained) * z
        residual = self.residual(torch.cat([z, cues], dim=-1))
        return mechanistic - decay + torch.sigmoid(self.residual_scale) * residual


def rk4_integrate(
    field: HybridVectorField,
    z0: Tensor,
    cues: Optional[Tensor],
    horizon: float,
    steps: int,
) -> Tuple[Tensor, Tensor]:
    """Differentiable fixed-step RK4 integration."""
    if steps < 1 or horizon <= 0:
        raise ValueError("horizon and steps must be positive.")
    dt = horizon / steps
    z = z0
    path = [z0]
    for _ in range(steps):
        k1 = field(z, cues)
        k2 = field(z + 0.5 * dt * k1, cues)
        k3 = field(z + 0.5 * dt * k2, cues)
        k4 = field(z + dt * k3, cues)
        z = z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        path.append(z)
    return z, torch.stack(path, dim=1)


class PriorConstrainedAttractorModel(nn.Module):
    """Infer TF-aligned latent state from accessibility and evolve it in time.

    The encoder intentionally has no RNA, cell label, pseudotime, cluster ID,
    or target-state input. Those quantities would be direct proxies for the
    state the model is expected to derive.
    """

    def __init__(
        self,
        priors: PriorMatrices,
        cue_dim: int = 0,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        priors.validate()
        num_peaks, num_genes = priors.peak_to_gene.shape
        num_tfs = priors.motif_tf_gene.shape[0]

        self.num_peaks = num_peaks
        self.num_genes = num_genes
        self.num_tfs = num_tfs
        self.cue_dim = cue_dim

        # Confidence weights are retained; row normalization prevents genes
        # with many linked peaks from dominating solely because of annotation
        # density.
        p2g = priors.peak_to_gene.float()
        p2g = p2g / p2g.sum(dim=0, keepdim=True).clamp_min(1.0)
        self.register_buffer("peak_to_gene", p2g)
        self.register_buffer("motif_tf_gene", (priors.motif_tf_gene > 0).float())

        encoder_dim = num_genes + num_tfs + cue_dim
        self.encoder = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_tfs),
        )
        self.vector_field = HybridVectorField(
            num_tfs=num_tfs,
            cue_dim=cue_dim,
            circuit_mask=priors.circuit_tf_tf.float(),
        )

        # The decoder is gated by both motif support and accessibility. Unknown
        # TF-gene edges remain exactly zero; they are not initialized at 0.25.
        self.decoder_weight = nn.Parameter(torch.empty(num_tfs, num_genes))
        nn.init.normal_(self.decoder_weight, std=0.03)
        self.gene_bias = nn.Parameter(torch.zeros(num_genes))

    def accessibility_features(self, atac: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        if atac.ndim != 2 or atac.shape[1] != self.num_peaks:
            raise ValueError("atac must have shape [batch, num_peaks].")
        gene_access = atac @ self.peak_to_gene
        gene_access = gene_access / gene_access.amax(dim=1, keepdim=True).clamp_min(1e-6)
        binding_gate = self.motif_tf_gene.unsqueeze(0) * gene_access.unsqueeze(1)
        tf_binding = binding_gate.sum(dim=-1) / self.motif_tf_gene.sum(dim=-1).clamp_min(1.0)
        return gene_access, tf_binding, binding_gate

    def encode(self, atac: Tensor, cues: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        gene_access, tf_binding, binding_gate = self.accessibility_features(atac)
        if cues is None:
            cues = atac.new_zeros((atac.shape[0], self.cue_dim))
        if cues.shape != (atac.shape[0], self.cue_dim):
            raise ValueError("cues has the wrong shape.")
        z0 = self.encoder(torch.cat([gene_access, tf_binding, cues], dim=-1))
        return z0, binding_gate

    def decode(self, z: Tensor, binding_gate: Tensor) -> Tensor:
        gated_weights = self.decoder_weight.unsqueeze(0) * binding_gate
        log_rna = torch.einsum("bt,btg->bg", z, gated_weights) + self.gene_bias
        return F.softplus(log_rna)

    def forward(
        self,
        atac: Tensor,
        cues: Optional[Tensor] = None,
        horizon: float = 1.0,
        steps: int = 8,
    ) -> Dict[str, Tensor]:
        z0, binding_gate = self.encode(atac, cues)
        zt, latent_path = rk4_integrate(
            self.vector_field, z0, cues, horizon=horizon, steps=steps
        )
        rna_path = torch.stack(
            [self.decode(latent_path[:, i], binding_gate) for i in range(steps + 1)],
            dim=1,
        )
        return {
            "z0": z0,
            "zt": zt,
            "latent_path": latent_path,
            "rna0_pred": rna_path[:, 0],
            "rna_t_pred": rna_path[:, -1],
            "rna_path": rna_path,
            "terminal_velocity": self.vector_field(zt, cues),
            "binding_gate": binding_gate,
        }

    @torch.no_grad()
    def stability_eigenvalues(self, z_star: Tensor, cue: Optional[Tensor] = None) -> Tensor:
        """Return Jacobian eigenvalues for one candidate fixed point.

        A locally asymptotically stable attractor has negative real parts for
        every eigenvalue. This method is diagnostic and is not used to select
        or label held-out states.
        """
        if z_star.shape != (self.num_tfs,):
            raise ValueError("z_star must be a single [num_tfs] state.")
        if cue is None:
            cue = z_star.new_zeros((self.cue_dim,))

        with torch.enable_grad():
            z = z_star.detach().clone().requires_grad_(True)

            def single_field(x: Tensor) -> Tensor:
                return self.vector_field(x.unsqueeze(0), cue.unsqueeze(0)).squeeze(0)

            jac = torch.autograd.functional.jacobian(single_field, z)
        return torch.linalg.eigvals(jac.detach())

    def find_fixed_point(
        self,
        initial_state: Tensor,
        cue: Optional[Tensor] = None,
        iterations: int = 500,
        learning_rate: float = 1e-2,
        tolerance: float = 1e-6,
    ) -> Tuple[Tensor, float]:
        """Refine one candidate state by minimizing the vector-field norm.

        Candidate initialization must come from the training data or a
        predeclared perturbation grid. Do not search held-out outcomes and then
        report the selected state as an unbiased test result.
        """
        if initial_state.shape != (self.num_tfs,):
            raise ValueError("initial_state must be a single [num_tfs] state.")
        if cue is None:
            cue = initial_state.new_zeros((self.cue_dim,))
        z = nn.Parameter(initial_state.detach().clone())
        optimizer = torch.optim.Adam([z], lr=learning_rate)
        residual = float("inf")
        for _ in range(iterations):
            optimizer.zero_grad()
            velocity = self.vector_field(z.unsqueeze(0), cue.unsqueeze(0)).squeeze(0)
            loss = velocity.square().mean()
            loss.backward()
            optimizer.step()
            residual = float(loss.detach().sqrt().cpu())
            if residual <= tolerance:
                break
        return z.detach(), residual


def attractor_objective(
    output: Dict[str, Tensor],
    rna_target: Tensor,
    terminal_mask: Optional[Tensor] = None,
    circuit_l1: float = 1e-4,
    model: Optional[PriorConstrainedAttractorModel] = None,
) -> Dict[str, Tensor]:
    """Training loss with optional endpoint-attractor evidence.

    terminal_mask must identify biologically justified terminal observations
    from the experimental design. Do not create it from clusters inferred on
    the complete dataset.
    """
    reconstruction = F.mse_loss(torch.log1p(output["rna_t_pred"]), torch.log1p(rna_target))
    cosine = 1.0 - F.cosine_similarity(
        torch.log1p(output["rna_t_pred"]), torch.log1p(rna_target), dim=-1
    ).mean()
    if terminal_mask is None or not bool(terminal_mask.any()):
        fixed_point = reconstruction.new_zeros(())
    else:
        fixed_point = output["terminal_velocity"][terminal_mask].square().mean()
    sparsity = reconstruction.new_zeros(())
    if model is not None:
        magnitudes = F.softplus(model.vector_field.circuit_magnitude_unconstrained)
        active = model.vector_field.circuit_confidence > 0
        if bool(active.any()):
            sparsity = circuit_l1 * magnitudes[active].mean()
    total = reconstruction + 0.25 * cosine + 0.1 * fixed_point + sparsity
    return {
        "total": total,
        "reconstruction": reconstruction,
        "cosine": cosine,
        "fixed_point": fixed_point,
        "circuit_l1": sparsity,
    }


def group_holdout_split(
    group_ids: Sequence[str],
    test_fraction: float = 0.2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split by donor/experiment before preprocessing or graph construction."""
    groups = np.asarray(group_ids)
    unique = np.unique(groups)
    if unique.size < 2:
        raise ValueError("Leakage-safe evaluation requires at least two groups.")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique)
    n_test = max(1, int(np.ceil(test_fraction * unique.size)))
    test_groups = set(shuffled[:n_test].tolist())
    test = np.array([g in test_groups for g in groups], dtype=bool)
    train = ~test
    if set(groups[train]).intersection(set(groups[test])):
        raise RuntimeError("Group leakage detected.")
    return train, test


def leakage_audit(
    train_groups: Sequence[str],
    test_groups: Sequence[str],
    encoder_feature_names: Sequence[str],
) -> None:
    """Fail loudly on group overlap or direct state proxies."""
    overlap = set(train_groups).intersection(test_groups)
    if overlap:
        raise ValueError(f"Train/test group overlap: {sorted(overlap)}")
    forbidden_tokens = {
        "rna",
        "scrna",
        "transcriptome",
        "transcriptomic",
        "celltype",
        "cluster",
        "pseudotime",
    }
    forbidden_phrases = {
        "cell_type",
        "gene_expression",
        "expression_profile",
        "target_state",
        "future_state",
    }
    normalized = {
        re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
        for value in encoder_feature_names
    }

    def is_direct_proxy(name: str) -> bool:
        tokens = set(name.split("_"))
        return bool(tokens.intersection(forbidden_tokens)) or any(
            phrase == name
            or name.startswith(phrase + "_")
            or name.endswith("_" + phrase)
            or ("_" + phrase + "_") in name
            for phrase in forbidden_phrases
        )

    bad = sorted(name for name in normalized if is_direct_proxy(name))
    if bad:
        raise ValueError(f"Direct state proxies found in encoder inputs: {bad}")


def _synthetic_smoke_test() -> None:
    torch.manual_seed(7)
    batch, peaks, genes, tfs, cues = 5, 17, 11, 4, 2
    priors = PriorMatrices(
        peak_to_gene=(torch.rand(peaks, genes) > 0.7).float(),
        motif_tf_gene=(torch.rand(tfs, genes) > 0.5).float(),
        circuit_tf_tf=(torch.rand(tfs, tfs) > 0.6).float()
        * torch.where(torch.rand(tfs, tfs) > 0.5, 1.0, -1.0),
    )
    model = PriorConstrainedAttractorModel(priors, cue_dim=cues)
    output = model(torch.rand(batch, peaks), torch.rand(batch, cues))
    assert output["rna_t_pred"].shape == (batch, genes)
    assert output["latent_path"].shape == (batch, 9, tfs)
    assert torch.isfinite(output["rna_t_pred"]).all()
    leakage_audit(["donor_1"], ["donor_2"], ["ATAC_peaks", "external_cue"])
    try:
        leakage_audit(["donor_1"], ["donor_2"], ["RNA_counts"])
    except ValueError:
        pass
    else:
        raise AssertionError("RNA proxy was not rejected by leakage_audit.")
    print("Synthetic smoke test passed.")


if __name__ == "__main__":
    _synthetic_smoke_test()
