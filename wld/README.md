# Waddington Latent Dynamics (WLD)

WLD is an experimental, prior-constrained model for deriving a TF-aligned latent cell state from chromatin accessibility. It is a parallel prototype within `MultiscaleProject`; it does not replace the repository's signed-GRN attractor model.

The model represents three biological layers explicitly:

1. **Epigenetic landscape:** peak-to-gene links map open chromatin to accessible gene programs.
2. **Binding feasibility:** TF-to-gene motif or occupancy evidence limits which regulators can act at accessible regions.
3. **Circuit interactions:** a signed, confidence-weighted TF circuit constrains the mechanistic component of the vector field.

The latent dynamics are hybrid: a constrained ODE supplies the interpretable circuit dynamics, while a bounded neural residual can represent missing biology. RNA, cell labels, clusters, pseudotime, and target-state labels are excluded from the encoder because they are direct proxies for the state the model is supposed to derive.

## Files

- `wld_attractor_model_v2.py` — reusable PyTorch architecture, RK4 integration, fixed-point search, Jacobian stability diagnostics, grouped splitting, and leakage checks.
- `run_wld_pbmc_colab.py` — end-to-end Colab runner for the public 10x PBMC multiome snapshot.
- `wld_next_experiments.py` — audits for the original WLD notebook, including identity and mean baselines, delta metrics, a target-PCA leakage reduction, modality shuffling, prior ablations, and seed sensitivity.
- `docs/attractor_state_computational_revision.md` — manuscript-ready computational framing and minimum experimental design.

## Recommended Colab run

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
    "run_wld_full_validation.py",
]
for name in files:
    urllib.request.urlretrieve(f"{base}/{name}", work / name)

run_env = os.environ.copy()
run_env["PYTHONPATH"] = str(packages)
run_env["PYTHONNOUSERSITE"] = "1"
subprocess.run(
    [sys.executable, str(work / "run_wld_full_validation.py")],
    check=True,
    env=run_env,
)
```

The full validation runner downloads the 10x matrix, selects genes and linked
peaks using training cells only, compiles CollecTRI priors, executes the
synthetic architecture and leakage tests, trains an ATAC-to-RNA state
reconstruction model, compares it with training-mean and ridge baselines, and
verifies the saved outputs.

Expected outputs:

- `wld_pbmc_results.json`
- `wld_pbmc_state_model.pt`
- `wld_pbmc_state_results.png`

## Legacy original-notebook audits

Run the original notebook cell first, upload `wld_next_experiments.py`, then execute:

```python
%run -i wld_next_experiments.py
```

These audits are retained only to diagnose the originally reported metrics.
They do not validate temporal dynamics or attractors, because the PBMC dataset
contains no observed transitions. The default audit runs the current-metric and
leakage-reduced checks. Enable the slower audits one at a time before `%run`:

```python
%env WLD_RUN_MODALITY=1
%env WLD_RUN_PRIOR=1
%env WLD_RUN_SEEDS=1
```

## Claim boundary

The public 10x PBMC dataset is one biological sample at one time point. It can test held-out cross-modal **state reconstruction**, but it cannot identify temporal dynamics, fixed-point stability, basin geometry, or state transitions. Accordingly, `run_wld_pbmc_colab.py` freezes the vector field and reports trajectory and attractor metrics as not applicable.

An attractor claim requires donor- or experiment-grouped longitudinal, lineage-traced, metabolic-labeling, or perturbation-resolved data. Preprocessing and prior compilation must be fit on training groups only. Candidate fixed points must converge from multiple initial conditions, have small vector-field residuals, and have Jacobian eigenvalues with negative real parts on held-out groups.

The previously reported final-state Pearson correlation should be read alongside the no-change baseline and delta metrics. A high final-state correlation alone can be dominated by retained cell identity and is not evidence that the model learned a transition.
