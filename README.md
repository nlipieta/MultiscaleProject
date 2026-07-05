# Synthetic Chromatin Toggle — Cell-State Bias GNN

A knowledge-graph Graph Neural Network that predicts which **response program** an
**extrinsic cue** drives in a cell, given that cell's **intrinsic memory** (its
lineage transcription factors and chromatin marks). It implements the
`signal → memory → response` architecture from the report *"Predicting Cell-State
Bias: A Deep Literature Review and Computational Framework for Synthetic Chromatin
Toggles."*

- **Nodes** = KG entities: cues, signaling proteins, TFs, chromatin modifiers,
  marks, plasticity states, and response programs (from Table 5 of the report).
- **GNN** = relation-typed, temporal message passing (R-GCN × GRU over "simulated
  time") → phenotype probabilities over the response programs (+ a *Quiescent* class).
- **Prediction** = e.g. *myoblast + TGF-β → MyogenicDiff*, *epithelial + stiff ECM
  → Fibrosis*, *Xenopus + bioelectric depolarization → Regeneration*.

---

## ⚠️ Read this first — what is and isn't real

The **knowledge graph and the GNN architecture are real** and literature-grounded.
There is **no single-cell dataset in the report**, so out of the box the model
trains on a **bootstrap/wiring harness**: the mechanistic KG oracle (`oracle.py`)
generates labels by propagating cues through the literature edges, and the GNN
learns to reproduce them.

**This proves the pipeline end-to-end and recovers the known mechanisms — it is
not an independent scientific result.** For real predictive claims, train on
measured phenotype labels (e.g. Perturb-seq / scRNA-seq) via the CSV interface
below. The `literature_cases.yaml` anchors (Ostuni 2013, Chang 2018, Backs 2006,
Mullen 2011, Mills 2017, Levin/Tseng) are held out and used to check the trained
model reproduces published biology.

---

## Run it (target Mac needs `uv`, plus internet once)

```bash
cd chromatin-toggle-model
uv sync                 # creates .venv, fetches a compatible Python + torch etc.

uv run chromatin-train  # trains, prints the literature-anchor scorecard, saves artifacts/model.pt

uv run chromatin-predict --context myoblast   --cue TGFbeta
uv run chromatin-predict --context epithelial --cue MechanicalStiffness
uv run chromatin-predict --context xenopus    --cue BioelectricDepolarization
uv run chromatin-predict --list               # show all contexts, cues, levels
```

`uv sync` reads `pyproject.toml`, provisions Python 3.10–3.12 (system Python is
irrelevant), installs PyTorch, and builds the package. The GNN auto-selects the
Apple-Silicon **MPS** GPU and falls back to CPU (`--device cpu` to force it).

## Sending it to another Mac

The whole project is self-contained. Either:

```bash
# zip the source (recipient runs `uv sync` to rebuild the env)
cd .. && zip -r chromatin-toggle-model.zip chromatin-toggle-model \
    -x '*/.venv/*' -x '*/artifacts/*' -x '*/__pycache__/*'
```

…or just copy the folder (drop `.venv/` — it's rebuilt by `uv sync`). On the
other Mac: `cd chromatin-toggle-model && uv sync && uv run chromatin-train`.

## Real single-cell data from CZ CELLxGENE (Census)

Ground the model's **intrinsic-memory layer** in real transcriptomes. Each KG
factor node is mapped to a human gene (`gene_map` in `data/kg.yaml`); the census
backend pulls that gene's expression per cell type from the CELLxGENE Discover
Census, normalizes (CP10K + log1p), and scales it to `[0, 1]`.

```bash
uv sync --extra census            # installs the heavier tiledbsoma/census stack

# build expression-grounded memory vectors -> data/cellxgene_contexts.csv
uv run chromatin-census

# predict using a REAL, data-derived memory state (not hand-set 0/1)
uv run chromatin-predict --real-context ESC     --cue TGFbeta
uv run chromatin-predict --real-context acinar  --cue Caerulein

# optional: build a training set whose MEMORY is real (cells) x cues, then train
uv run chromatin-census --make-training data/real_train.csv
uv run chromatin-train  --data data/real_train.csv
```

A prebuilt `data/cellxgene_contexts.csv` is committed (Census 2025-11-08, 200
cells/type). The lineage-defining TFs land correctly — **PU1 highest in
macrophage, MyoD highest in myoblast** — validating the gene→node grounding.
Caveat: the "embryonic stem cell" label is thin in this Census slice (~34 cells),
so ESC is under-sampled and its markers are unreliable; regenerate with a larger
panel or higher `--max-cells`, or keep the hand-set ESC context for that row.

**What CELLxGENE does and doesn't give you.** It supplies real, data-driven
values for the lineage-TF / signaling *memory* nodes. It does **not** contain an
"applied cue" or a measured *response-program* label — the Census catalogs cell
states, not perturbation→phenotype pairs. So the cue is still the applied
perturbation, and `--make-training` labels with the mechanistic oracle over real
memory vectors (an honest bootstrap). For fully-supervised training, supply
measured phenotype labels (e.g. Perturb-seq) via the generic CSV path below.
Xenopus/planarian contexts have no human-Census equivalent and keep their
hand-set memory.

## Real-label benchmark (non-circular numbers)

`chromatin-train`'s accuracy is high but *circular*: its labels come from the KG
oracle, so the GNN just relearns the rules it was given. `chromatin-realbench`
fixes this — it trains and evaluates on **real CELLxGENE annotations**, split
**by dataset** (test datasets unseen in training → no batch/donor leakage), and
compares the KG-GNN against a majority-class floor and a logistic-regression
baseline.

```bash
uv sync --extra census
uv run chromatin-realbench                       # predict cell_type from KG genes
uv run chromatin-realbench --obs-column disease --classes normal "pulmonary fibrosis"
```

Measured result (Census 2025-11-08, 4 lineages, 400 cells/type, held out by
dataset — 58 train / 25 test datasets, 219 test cells):

| model                | accuracy | macro-F1 |
|----------------------|:--------:|:--------:|
| majority-class       |  0.000   |    –     |
| logistic regression  |  0.712   |  0.382   |
| KG-GNN               |  0.699   |  0.375   |

Honest reading: the KG-gene signature carries **real, generalizable signal**
(~70% cell-type accuracy on unseen datasets vs a 0% majority floor), but the
**GNN only matches the linear baseline** — the graph structure adds nothing for
this *static identity* task. That's expected: the KG dynamics are built for
*cue → program* propagation, not reading cell type off lineage-marker genes. This
benchmark validates the feature/representation layer on real data; testing the
toggle *dynamics* with real labels still needs a perturbation dataset
(Perturb-seq) fed through the generic CSV path below.

## Using any real observations (generic CSV)

Bypass the bootstrap entirely and train on measured data:

```bash
uv run chromatin-train --data path/to/observations.csv
```

CSV schema — one row per observation:

- one column per KG node name (see `data/kg.yaml`), value in `[0, 1]` = that
  node's initial activation for the observation. Map real assays onto nodes, e.g.
  cue columns from the applied perturbation; TF/marker columns from rank-scaled
  scRNA-seq; accessibility marks from scATAC. Missing columns default to 0.
- a `label` column naming the observed program (one of the classes in
  `data/kg.yaml → program_nodes`, or `Quiescent`).

Only the node-init encoding changes; the KG, GNN, and readout are unchanged.

## Real data ingestion (GEO + scPerturb)

Two loaders bridge real public data into the generic CSV interface above. Both
map assay genes onto the KG `gene_map` nodes and write the model's node-column
schema; you still supply the cue/label semantics.

```bash
# NCBI GEO — several deposition formats, one CLI (dataset rules in geo.py):
uv run chromatin-geo --gse GSE21608                       # bulk series matrix (Mullen TGF-b)
uv run chromatin-geo --gse GSE172380 --scrna              # genes x cells CSV (ADM)
uv run chromatin-geo --gse GSE120064 --scrna              # genes x cells CSV (cardiac hypertrophy)
uv run chromatin-geo --gse GSE135893 --mtx                # 10x MTX, streamed (lung fibrosis)
uv run --with anndata chromatin-geo --h5ad-dataset GSE168776   # AnnData h5ad (myogenesis)
uv run chromatin-geo --emtab9702 --min-counts 5000        # SORT-seq plates (trained immunity)

# scPerturb harmonized single-cell pool (Zenodo) -> CSV
uv run --with anndata chromatin-scperturb --list                  # inventory
uv run --with anndata chromatin-scperturb --file <name>.h5ad --map map.json

# pool per-pathway CSVs into ONE cross-pathway training set (+ provenance cols)
uv run chromatin-combine --cap-per-class 3000 --out data/cross_pathway.csv
uv run chromatin-train --data data/cross_pathway.csv --batch-size 256
```

The bundled cross-pathway set spans **6 pathways / 7 program classes**: ADM
(GSE172380), Hypertrophy (GSE120064), Fibrosis (GSE135893), InnateMemory
(E-MTAB-9702 trained immunity), MyogenicDiff (GSE168776), Pluripotency
(GSE21608), and Quiescent baselines from all. Six are at scale (~2–6k cells);
Pluripotency is still n=2 (needs an ESC single-cell arm).

**Cross-pathway pooling.** `chromatin-combine` aligns any number of per-pathway
CSVs onto the shared KG node columns, tags each row with `dataset`/`pathway`/
`assay`, and concatenates so one model learns every program jointly over the
graph. `--cap-per-class` subsamples over-represented (pathway, label) groups so a
single-cell pathway doesn't drown the bulk ones. **Caveat:** the pooled set is
only as balanced as its inputs — single-cell pathways (ADM: thousands of cells)
dwarf bulk pathways (Mullen TGF-beta: n=6), so classes like MyogenicDiff /
Pluripotency are currently too small to train. Genuine cross-pathway training
needs a *single-cell* dataset per pathway; see the feasibility table above.

**What the literature search found (feasibility per pathway).** Real
perturbation->label datasets exist for 4 of the 7 pathways; 2 are weak proxies;
1 has no matching data:

| Pathway | Best public dataset | Verdict |
|---|---|---|
| macrophage + LPS -> InnateMemory | GEO **GSE85246** (Novakovic 2016) + THP1 Perturb-seq | strong |
| ESC / myoblast + TGF-beta (5,6)   | GEO **GSE21608/GSE21621** (Mullen 2011)             | strong but tiny (n=6, two-color ratios) |
| cardiomyocyte + stretch -> Hypertrophy | GEO **GSE186208** (human) / GSE107551 (rat)    | good, bulk only |
| acinar + caerulein -> ADM         | GEO **GSE172380** (mouse scRNA)                     | partial, small |
| epithelial + stiff ECM -> Fibrosis | GSE272029 (prostate proxy)                         | proxy only — wrong phenotype |
| bioelectric -> Regeneration       | none (only mechanical amputation, PMC6986927)       | **no matching data** |

Because the pathways span human/mouse/rat/Xenopus and bulk/single-cell assays
with no shared label space, **per-pathway models are the realistic approach**,
not one CSV training all classes. scPerturb's perturbations are genetic/chemical
in cell lines and do **not** map to these cues/programs — use it only if
reframing the task to general perturbation-response (cf. CellCap / STATE).

## Files

```
data/kg.yaml               the literature knowledge graph (nodes, biases, typed edges)
data/contexts.yaml         cell contexts -> ON intrinsic-memory nodes
data/literature_cases.yaml held-out ground-truth cases from the report's tables
src/chromatin_toggle/
  kg.py         load KG -> tensors                inputs.py   (context, cue) -> node vector
  oracle.py     mechanistic KG propagation        dataset.py  bootstrap + real-CSV loaders
  model.py      the relational temporal GNN        train.py    train + anchor evaluation
  predict.py    inference + mechanistic trace      device.py   MPS/CPU selection
  census.py     CELLxGENE Census -> real memory contexts (optional --extra census)
  geo.py        NCBI GEO series -> CSV schema     scperturb.py  scPerturb pool -> CSV schema
  combine.py    pool per-pathway CSVs -> one cross-pathway training set
  realbench.py  real-label benchmark: KG-GNN vs baselines, held out by dataset
  baselines.py  KG-GNN vs majority/logreg/RF/gboost   ablate.py  mechanism/structure ablation
  perturb.py    in-silico hypertrophy perturbation test
artifacts/                 model.pt + metrics.json (created by training)
```

## Extending the biology

Edit `data/kg.yaml` to add nodes/edges (new cues, modifiers, programs) and
`data/contexts.yaml` to add cell types, then retrain — no code changes needed.
The engineering handles from the report (inducible HDAC nuclear export, an
F-actin-blind ARID1A mutant, bioelectric ion-flux control) can be modeled as new
cue nodes or edge-weight edits and their predicted phenotype read off directly.
