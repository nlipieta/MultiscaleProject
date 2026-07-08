# Curated state-transition dataset candidates — hunt 2026-07-07

Verified by parallel data-scout agents (each accession WebFetch-confirmed live). Bar: real scRNA-seq,
labels mapping to a `cue → cell-state program` vs a Quiescent/progenitor baseline, `geo.py`-ingestable
format, driver genes for the KG cascade. All checked for distinctness from already-ingested programs
(ADM, Hypertrophy, Fibrosis, InnateMemory, MyogenicDiff, Pluripotency, Regeneration, EMT, Senescence,
NeuronalDiff, Osteogenesis, T-cell-exhaustion).

## READ THIS FIRST — label-type is a leakage axis

Datasets fall into two classes. This matters more than the count:

- **PER-CELL labels** (an `obs.cell_type` / metadata column that names the state of each cell
  independent of which sample it came from). These are the safe ones — the program label is not a
  proxy for sample identity. Prioritize these.
- **SAMPLE/CONDITION labels** (program vs Quiescent = which sample/timepoint the cell came from).
  Here "program" can be silently confounded with batch/sample identity — the same failure mode as the
  cue-gating leak. Usable ONLY with grouped-split (hold out whole samples/datasets) AND with the
  understanding that a program present in only one sample cannot be cleanly separated from its batch.
  Do not let sample-of-origin masquerade as a per-cell label.

## PER-CELL-LABELED (ingest first)

| Program | Accession | Repo | Format | ~Size | Label (col → mapping) | Drivers | Note |
|---|---|---|---|---|---|---|---|
| Intestinal differentiation | CELLxGENE `fd89be61-2869-4342-a86e-e1fce3a8f269` | CELLxGENE | h5ad | 266 MB / 17.6k | `cell_type`: enterocyte/goblet/EEC/secretory=program; crypt-stem/TA/progenitor=Quiescent | LGR5,OLFM4,ASCL2,FABP2,MUC2,DEFA5,CHGA | fetal; adult 2nd src GSE185224 (1.3GB) |
| Erythropoiesis + Megakaryopoiesis | CELLxGENE `cd2f23c1-aef1-48ae-8eb4-0bcf124e567d` | CELLxGENE | h5ad | 2.2 GB / 263k | `cell_type`: erythroid-prog/erythroblast=Ery; MkP/megakaryocyte=Mega; HSC/MPP=Quiescent | GATA1,KLF1,GYPA,HBB (Ery); FLI1,PF4,ITGA2B,VWF (Mk) | one atlas → 2 programs; HOLD OUT BY DONOR (integrated across 6 studies) |
| Trophoblast differentiation | CELLxGENE `e2c257e7-6f79-487c-b81c-39451cd4ab3c` (start `primary_trophoblast_organoid`) | CELLxGENE | h5ad | 622 MB / 27k | `cell_type`: syncytiotrophoblast/EVT=program; villous cytotrophoblast=Quiescent | GATA3,KRT7,TP63 (CTB); CGA,CGB,GCM1 (STB); HLA-G (EVT) | dedicated organoid + TSC datasets in collection |
| Beta-cell dedifferentiation | GSE86469 | GEO | CSV | 12 MB | per-cell donor `disease`: T2D beta=program; ND beta=Quiescent | MAFA,NKX6-1,PDX1,UCN3 (loss); ALDH1A3 (up) | small n (human islets, Fluidigm) |
| Cancer drug-tolerant persister | GSE150949 | GEO | CSV+meta | 400 MB | per-cell meta: osimertinib cycling/non-cycling=persister; Day0=parental | NNMT,ATF3,CDKN1A,AXL; MKI67(loss) | PC9, Watermelon lineage barcoding |
| EMT (2nd source) | GSE213753 | GEO | MTX/CSV | 119 MB | per-cell dose barcode: 400–800pM=EMT; 0pM=Quiescent | CDH1/EPCAM vs VIM,ZEB1,CDH2,SNAI2 | ONE multiplexed pool — parse barcode→dose, not by GSM |
| MyogenicDiff (2nd source) | GSE143437 | GEO | txt+meta | 774 MB | `..._metadata.txt.gz` cell-type + timepoint: D0=quiescent MuSC; D2/5/7=activated | Pax7 vs Myod1,Myog,Myf5,Mymk | De Micheli notexin atlas |
| Astrocyte / gliogenesis | GSE245169 | GEO | h5ad | 6–11 GB | timecourse+clusters: D21 GFAP/AQP4=astrocyte; D0 iPSC/NPC=progenitor | GFAP,AQP4,S100B,SLC1A3,SOX9,NFIA | large h5ad; per-cell clusters in obs |

## SAMPLE/CONDITION-LABELED (usable with grouped-split; re-cluster for per-cell state)

| Program | Accession | Repo | Format | ~Size | Label (sample → mapping) | Drivers | Note |
|---|---|---|---|---|---|---|---|
| Adipogenesis | GSE226365 | GEO | MTX | 160 MB | filename timepoint: D0=preadipocyte; D5(mouse)/D8(human)=adipocyte | Pparg,Cebpa,Fabp4,Adipoq,Lpl | BLOCKER RESOLVED (no R needed); mouse mm10 cleanest |
| EndMT | GSE159843 | GEO | 10x .h5 | 170 MB | IL1b+TGFb2 D3/7=EndMT; untreated=Quiescent | SNAI1,ACTA2,CDH2,FN1 vs PECAM1,CDH5 | distinct from EMT |
| Hepatic stellate activation | GSE137720 | GEO | MTX/CSV | 249 MB | CCl4/BDL=myofibroblast; healthy=quiescent HSC | ACTA2,COL1A1,TIMP1,PDGFRB; LRAT,RGS5 (Q) | mouse; distinct from lung/kidney fibrosis |
| Kidney PT injury/maladaptive repair | GSE139107 | GEO | TXT DGE | 90 MB | IRI 4h–6wk=injured/failed-repair; IRIsham=Quiescent | Vcam1,Ccl2,Havcr1,Krt20 vs Hnf4a,Slc34a1 | mouse snRNA |
| Reactive astrogliosis | GSE205511 | GEO | MTX | ~85 MB/samp | A53T transgenic=reactive; wildtype=Quiescent | Gfap,Vim,C3,Serpina3n,Lcn2 | mouse; distinct from astro differentiation |
| Alveolar KRT8+ transitional (ADI) | GSE141259 | GEO | MTX | 40+73 MB | bleomycin d2–54 epi=Krt8+ ADI; PBS/d0=AT2 Quiescent | Krt8,Cldn4,Sfn,Cdkn1a vs Sftpc,Sftpb | epithelial; distinct from lung fibrosis |
| Cardiomyocyte ischemic border-zone | GSE214611 | GEO | MTX/TSV | 2.7 GB | MI border-zone=BZ1/BZ2 stressed; sham=Quiescent | Nppa,Nppb,Ankrd1,Xirp2,Flnc | EXCLUDE TAC/isoproterenol arms (≠ Hypertrophy) |
| Keratinocyte differentiation | GSE147482 | GEO | MTX+TSV | 125 MB | clusters: spinous/granular=diff; basal=Quiescent | KRT5,KRT14,TP63 vs KRT1,KRT10,IVL | homeostatic (no cue); labels from paper supp/re-cluster |
| iPSC reprogramming | GSE106340 | GEO | MTX | 1.2 GB | OSKM timecourse: late iPSC=Pluripotent; D0 MEF=Quiescent | Pou5f1,Sox2,Klf4,Nanog,Sall4 | mouse; intermediates = trajectory |
| Cardiac reprogramming | GSE98567 (mouse) / GSE106888 (human) | GEO | XLSX counts | 173/108 MB | MGT=iCM; control fibroblast=Quiescent | Gata4,Mef2c,Tbx5,Tnnt2,Myh6 | Fluidigm plate (~500-650 cells); XLSX → convert |
| Liver regeneration | GSE158866 | GEO | MTX | 133 MB | PHx 48hr=regenerating; 0hr=Quiescent | Ccnd1,Mki67,Top2a,Afp,Sox9 | mouse |
| Intestinal regeneration (revSC) | GSE117783 | GEO | CSV | 420 MB | irradiated=revival(revSC); normal=Quiescent | Clu,Anxa1,Ly6a,Msln vs Lgr5,Olfm4 | distinct from intestinal DIFFERENTIATION; 2nd model GSE108233 |
| Blastema regeneration | GSE137971 (zebrafish fin) / GSE121737 (axolotl limb) / GSE143888 (mouse digit) | GEO | CSV/TXT/MTX | 63/256/266 MB | post-amputation dpa=blastema; intact=Quiescent | Prrx1,Msx1,Mki67 (+ msx1,mmp9 fish) | cross-species; ortholog map needed |
| Wound keratinocyte | GSE142471 | GEO | MTX+TSV | 166 MB | wounded=activated; unwounded=Quiescent | Krt6a,Krt16,Krt17,Sprr1b vs Krt14 | injury-cued (vs homeostatic GSE147482) |
| Hepatocyte differentiation | GSE159557 | GEO | MTX | 1.3 GB | DM=mature hepatocyte; HM=progenitor organoid | ALB,AFP,HNF4A,APOA1,CYP3A4 | 2nd src E-MTAB-7189 (ArrayExpress, confirm layout) |

## 2ND SOURCES for existing single-source programs
- Senescence: GSE226225 (WI-38; RS/IR/ETO=Senescence, CTRL/ETO-d0=Quiescent; 905 MB MTX; sample-level)
- InnateMemory/trained-immunity: GSE184241 (in-vivo BCG D0 vs 3mo; 4.7 MB; complementary to E-MTAB-9702 in-vitro)
- Pluripotency: GSE75748 (hESC→definitive endoderm; 37 MB CSV; human)
- EMT: GSE213753 (above); MyogenicDiff: GSE143437 (above)

## IMMUNE (from immune scout)

Per-cell-labeled (ingest first):
| Program | Accession | Repo | Format | ~Size | Label | Drivers | Note |
|---|---|---|---|---|---|---|---|
| B-cell → germinal-center/plasma differentiation | CELLxGENE `482954b2-0456-4901-b379-b62f99c0ab2d` (King 2021) | CELLxGENE | h5ad | 214 MB / 25.7k | `cell_type`: centroblast/centrocyte/GC-B/plasmablast=program; naive B=Quiescent | BCL6,AICDA,RGS13 (GC); PRDM1,XBP1,IRF4,SDC1 (plasma) | in-vivo tonsil (cue=GC reaction) |
| T-helper (Tfh) + Treg induction | CELLxGENE `f54647ec-0c03-4775-8dac-5a477c10a3f5` (King 2021) | CELLxGENE | h5ad | 58 MB / 8.8k | `cell_type`: Tfh/Tfr/Treg=program; CD4 helper=Quiescent | BCL6,CXCR5,PDCD1,IL21 (Tfh); FOXP3,IL2RA,CTLA4 (Treg) | one file → 2 programs; baseline softer (not sorted naive) |

Sample/condition-labeled:
| Program | Accession | Repo | Format | ~Size | Label | Drivers | Note |
|---|---|---|---|---|---|---|---|
| Macrophage polarization (M1/M2) | GSE161125 | GEO | 10x MTX | 84 MB | control=Quiescent; LPS+IFNγ=M1; IL-4=M2 (drop co-stim arm) | NOS2,IL1B,TNF,CXCL9 (M1); ARG1,MRC1,RETNLA (M2) | mouse BMDM in-vitro; cleanest immune cue design. 2nd src GSE117176 (IL-4+IL-13) |
| Monocyte → macrophage | GSE221310 | GEO | TXT (BD Rhapsody) | 45 MB | Ly6Chi monocyte=Quiescent; CMDM/resident macrophage=program | Ly6c2,Ccr2,Plac8 (mono); Mrc1,C1qa,Apoe (macro) | PARTIAL — re-cluster from counts (no per-cell annotation file) |

Immune GAPS (NOT verified — leads only): DC maturation (immature vs LPS moDC) and NK activation
(resting vs IL-15/cytokine). Follow-up search surfaced leads but locked none: Shalek 2014 BMDC+LPS
(GSE41265, tiny/Smart-seq — poor fit), blood-DC atlas (Villani GSE94820, steady-state), Fehniger
memory-like NK (IL-12/15/18, accession unconfirmed). Needs a focused search before ingest; treat DC
and NK as open.

## HUNT COMPLETE — all six scouts reported.

## REJECTED (checked, unusable)
GSE160312 (bulk), PRJNA769413 (FASTQ only), GSE163907/GSE112294 (zebrafish, off-target), PNAS-2311685121
(Xenopus), GSE180678 (single infarct, no control), GSE145927 (mislabeled — kidney not lung).

## Honest scope notes
- ~23 new programs + 5 second-sources verified here → would take the pool from 13 to ~35+ programs.
- More classes will LOWER argmax accuracy metrics (harder problem) even as it adds variety; report
  program-AUPRC / balanced-accuracy, not overall accuracy (already the project's convention).
- Format work: cardiac (XLSX) and neural-crest (Seurat .Rds, see memory) need conversion; astrocyte/
  BoneMarrowMap/CM/iPSC-reprog are multi-GB (watch disk — .git already large, keep data/raw gitignored).
- Species mix (mouse/human/zebrafish/axolotl) needs ortholog mapping; cross-species is itself a test.
