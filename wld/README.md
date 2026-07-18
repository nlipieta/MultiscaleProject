# Waddington Latent Dynamics (WLD)

WLD is an experimental framework with two deliberately separate tracks. The
PBMC runner evaluates ATAC-only, single-snapshot RNA state reconstruction. The
temporal architecture defines circuit-constrained dynamics for future paired
longitudinal or perturbation data. Snapshot metrics are never reported as
evidence of trajectories or attractors. WLD is a parallel prototype within
`MultiscaleProject`; it does not replace the repository's signed-GRN attractor
model.

The recommended v3 temporal model represents four biological layers explicitly:

1. **Tissue and extracellular cues:** measured inputs enter a signed signaling/PPI graph.
2. **Epigenetic landscape:** peak-level ATAC defines which localized regulatory elements are open.
3. **Binding feasibility:** TF motif or occupancy evidence is intersected with enhancer-to-gene links.
4. **Circuit interactions:** signed TF-to-TF and TF-to-gene relations are the vector field topology.

In v3 the circuit is not handed to a general latent network. Trainable interaction
parameters exist only on supplied edges, edge signs are fixed, and explicit Hill
production, degradation, and slow chromatin kinetics determine the vector field.
There is no neural residual that can bypass the circuit. RNA targets, cell labels,
clusters, pseudotime, and target-state labels are excluded from initialization
because they are direct proxies for the state the model is supposed to derive.
The earlier v2 hybrid dynamics are retained for reproducibility and smoke tests,
but they are not the recommended route for a temporal attractor claim.

## Files

- `wld_attractor_model_v2.py` — reusable PyTorch architecture, RK4 integration, fixed-point search, Jacobian stability diagnostics, grouped splitting, and leakage checks.
- `wld_circuit_dynamics_v3.py` — recommended hard-sparse signaling/TF/chromatin/RNA vector field with enhancer gates, interventions, and attractor diagnostics; no neural bypass.
- `run_wld_v3_validation.py` — structural and numerical contract tests on neutral systems; deliberately not a chromatin-toggle benchmark.
- `wld_temporal_training.py` — grouped temporal trainer with population-level alignment for destructive assays, explicit paired-lineage mode, validation-selected checkpoints, sealed test groups, and circuit controls.
- `run_wld_temporal_smoke.py` — neutral synthetic end-to-end check of the temporal data, training, control, checkpoint, and claim-boundary contracts.
- `run_wld_pbmc_colab.py` — deterministic ATAC-only state-reconstruction runner with mean/ridge baselines, a degree-preserving TF-gene permutation control, a held-out ATAC shuffle, and multi-seed reporting.
- `run_wld_full_validation.py` — environment, syntax, architecture, leakage, output-contract, and claim-boundary checks.
- `wld_next_experiments.py` — audits for the original WLD notebook, including identity and mean baselines, delta metrics, a target-PCA leakage reduction, modality shuffling, prior ablations, and seed sensitivity.
- `docs/legacy_colab_audit.md` — fingerprints, saved outputs, and scientific interpretation of the immutable exploratory notebook.
- `docs/attractor_state_computational_revision.md` — manuscript-ready computational framing and minimum experimental design.
- `docs/wld_v3_circuit_dynamics.md` — v3 data contract, training path, controls, and attractor falsification criteria.
- `docs/wld_temporal_data_contract.md` — exact manifest/NPZ schema for grouped temporal or perturbation cohorts and the commands for real-data training.

## WLD v3 contract check in Colab

Run this small cell first. It uses Colab's existing PyTorch installation and
does not install or replace NumPy, SciPy, Scanpy, or any compiled package. It
checks the hard circuit topology, enhancer gate, intervention path, leakage
contract, signed negative control, and attractor-diagnostic plumbing. It does
not download PBMC data or make a biological attractor claim.

```python
import json
import pathlib
import subprocess
import sys
import urllib.request

branch = "agent/add-wld-attractor-model"
base = f"https://raw.githubusercontent.com/nlipieta/MultiscaleProject/{branch}/wld"
work = pathlib.Path("/content/wld_v3_contract")
work.mkdir(parents=True, exist_ok=True)

for name in ("wld_circuit_dynamics_v3.py", "run_wld_v3_validation.py"):
    urllib.request.urlretrieve(f"{base}/{name}", work / name)

result = subprocess.run(
    [sys.executable, str(work / "run_wld_v3_validation.py")],
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
)
print(result.stdout)
if result.returncode:
    raise RuntimeError(
        f"WLD v3 contract check failed with exit code {result.returncode}; "
        "the complete inner traceback is printed above."
    )
print(json.loads((work / "wld_v3_validation.json").read_text()))
```

## Temporal trainer smoke test in Colab

This cell exercises the next layer: an unpaired destructive single-cell time
course, group-sealed checkpoint selection, held-out test evaluation, and a
no-circuit control. The generated cohort is deliberately small and neutral. A
PASS means that the temporal software contract works; it is not a biological
attractor result.

```python
import pathlib
import subprocess
import sys
import urllib.request

branch = "agent/add-wld-attractor-model"
base = f"https://raw.githubusercontent.com/nlipieta/MultiscaleProject/{branch}/wld"
work = pathlib.Path("/content/wld_temporal_smoke")
work.mkdir(parents=True, exist_ok=True)

for name in (
    "wld_circuit_dynamics_v3.py",
    "wld_temporal_training.py",
    "run_wld_temporal_smoke.py",
):
    urllib.request.urlretrieve(f"{base}/{name}", work / name)

result = subprocess.run(
    [sys.executable, str(work / "run_wld_temporal_smoke.py")],
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
)
print(result.stdout)
if result.returncode:
    raise RuntimeError(
        f"WLD temporal smoke test failed with exit code {result.returncode}; "
        "the complete inner traceback is printed above."
    )
```

## Full repository validation in Colab

Use the single cell below in a fresh Colab runtime. It installs the scientific
stack into a disposable package directory and exposes that directory only to a
separate validation subprocess. The notebook kernel never imports or replaces
those NumPy/SciPy libraries. This prevents the DLPack and missing-OpenBLAS
errors caused by upgrading compiled packages in a running Colab process, and
does not depend on Colab's unavailable Debian `python3-venv` component.

```python
import pathlib
import shutil
import subprocess
import sys
import urllib.request
import os

branch = "agent/add-wld-attractor-model"
base = f"https://raw.githubusercontent.com/nlipieta/MultiscaleProject/{branch}/wld"
work = pathlib.Path("/content/wld_validation")
packages = pathlib.Path("/content/wld_validation_packages")

# These are explicit disposable Colab paths, not Drive paths.
for path in (work, packages):
    if path.exists():
        shutil.rmtree(path)
work.mkdir(parents=True)
packages.mkdir(parents=True)

subprocess.run(
    [
        sys.executable, "-m", "pip", "install", "--no-cache-dir",
        "--target", str(packages), "scanpy==1.12.2", "decoupler==2.1.6",
        "mygene", "threadpoolctl>=3.6",
    ],
    check=True,
)

files = [
    "wld_attractor_model_v2.py",
    "run_wld_pbmc_colab.py",
    "wld_next_experiments.py",
    "wld_circuit_dynamics_v3.py",
    "run_wld_v3_validation.py",
    "wld_temporal_training.py",
    "run_wld_temporal_smoke.py",
    "run_wld_full_validation.py",
]
for name in files:
    urllib.request.urlretrieve(f"{base}/{name}", work / name)

run_env = os.environ.copy()
run_env["PYTHONPATH"] = str(packages)
run_env["PYTHONNOUSERSITE"] = "1"
run_env["MPLBACKEND"] = "Agg"
run_env["WLD_EVAL_SEEDS"] = "42,123,456"
subprocess.run(
    [sys.executable, str(work / "run_wld_full_validation.py")],
    check=True,
    env=run_env,
)
```

The full validation runner downloads and verifies the 10x HDF5 matrix, selects
genes and linked peaks using training cells only, compiles deterministic
CollecTRI regulatory support, executes architecture and leakage tests, trains
ATAC-to-RNA state-reconstruction models across three seeds, and compares the
true TF-gene scaffold against a degree-preserving permutation and shuffled
held-out ATAC. The report retains failed controls rather than printing a
success claim unconditionally.

Expected outputs:

- `wld_v3_validation.json`
- `wld_pbmc_results.json`
- `wld_pbmc_state_model.pt`
- `wld_pbmc_state_results.png`

## Legacy notebook

The original Colab notebook is an immutable exploratory record, not the source
for the corrected runner. Its encoder receives RNA, its pseudotime and pairing
are inferred from RNA before splitting, and its binary target is defined from
test outcomes. Its final degree-matched negative control also performs at least
as well as the true circuit prior. Those results are retained as a negative
finding and are not mixed with the corrected snapshot experiment.

## Claim boundary

The public 10x PBMC dataset is one biological sample at one time point. It can
test development-only held-out cross-modal **state reconstruction**, but it
cannot provide donor-level OOD validation or identify temporal dynamics,
fixed-point stability, basin geometry, or state transitions. Accordingly,
`run_wld_pbmc_colab.py` invokes only the ATAC encoder and constrained decoder;
the temporal vector field is not trained or evaluated.

An attractor claim requires donor- or experiment-grouped longitudinal, lineage-traced, metabolic-labeling, or perturbation-resolved data. Preprocessing and prior compilation must be fit on training groups only. Candidate fixed points must converge from multiple initial conditions, have small vector-field residuals, and have Jacobian eigenvalues with negative real parts on held-out groups.

The included v3 validator uses a neutral stable feed-forward system only to
check fixed-point, Jacobian, and basin-analysis plumbing. A chromatin toggle is
a downstream engineering application and is intentionally not used as evidence
that WLD learned endogenous cell-state dynamics.

No prior-dependence claim is made unless the true TF-gene scaffold beats the
degree-preserving permutation across every configured seed. No ATAC-dependence
claim is made unless paired predictions deteriorate after held-out ATAC profiles
are shuffled. Global Pearson correlation is reported with per-cell, per-gene,
MSE, and R² metrics; it is not evidence that the model learned a transition.
