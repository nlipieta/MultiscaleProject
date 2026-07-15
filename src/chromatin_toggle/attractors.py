"""Attractor analysis for signed-GRN dynamical models: fixed points, basins, stability.

Dynamics (matches GRNDynamics._step):  x <- (1-eta)x + eta*tanh(A x + b),
A[dst,src] = softplus(mag)*sign (dense signed adjacency). A stable fixed point x* satisfies x*=step(x*)
with Jacobian J = (1-eta)I + eta*diag(1-tanh(A x*+b)^2) A having spectral radius < 1.

These primitives are shared by scripts/attractor_diagnostic.py (is the model multistable?) and
scripts/train_multistable.py (did training MAKE it multistable?).
"""
from __future__ import annotations

import torch


def build_A(m):
    """Dense signed weighted adjacency + params from a GRNDynamics-like model. Detached (analysis)."""
    with torch.no_grad():
        A = torch.zeros(m.N, m.N, device=m.mag.device)
        w = torch.nn.functional.softplus(m.mag) * m.sign
        A.index_put_((m.dst, m.src), w, accumulate=True)
        return A, m.bias.detach(), float(m.in_scale.detach()), m.eta


def step(A, b, eta, x):
    return (1 - eta) * x + eta * torch.tanh(x @ A.T + b)


def settle_converged(A, b, eta, x0, tol=1e-5, max_steps=500):
    """Iterate to a fixed point; return (x*, steps_used, converged_mask)."""
    x = x0; last = None
    for t in range(max_steps):
        xn = step(A, b, eta, x)
        d = (xn - x).abs().amax(dim=1); x = xn
        if float(d.max()) < tol:
            return x, t + 1, d < tol
        last = d
    return x, max_steps, (last < tol) if last is not None else torch.zeros(x.size(0), dtype=torch.bool)


def jacobian(A, b, eta, xstar):
    """Analytic Jacobian of the map at a fixed point [N,N]."""
    g = 1.0 - torch.tanh(xstar @ A.T + b) ** 2
    return (1 - eta) * torch.eye(A.size(0), device=A.device) + eta * (g[:, None] * A)


def spectral_radius(A, b, eta, xstar):
    return float(torch.linalg.eigvals(jacobian(A, b, eta, xstar)).abs().max())


def greedy_cluster(states, eps_rms):
    """Cluster settled states by RMS-per-node distance; returns (labels[long], centroids[K,N])."""
    cents = []; labels = torch.empty(states.size(0), dtype=torch.long)
    for i in range(states.size(0)):
        s = states[i]
        if cents:
            d = ((torch.stack(cents) - s) ** 2).mean(1).sqrt()
            j = int(d.argmin())
            if float(d[j]) < eps_rms:
                labels[i] = j; continue
        labels[i] = len(cents); cents.append(s.clone())
    return labels, torch.stack(cents)


def enumerate_attractors(m, x0, eps_rms=0.08):
    """Settle x0 to convergence, cluster fixed points, return (labels, centroids, rho_per_cluster, x_star)."""
    A, b, in_scale, eta = build_A(m)
    with torch.no_grad():
        xstar, _, _ = settle_converged(A, b, eta, x0 * in_scale)
    labels, cents = greedy_cluster(xstar, eps_rms)
    rho = [spectral_radius(A, b, eta, cents[k].to(A.device)) for k in range(cents.size(0))]
    return labels, cents, rho, xstar
