"""Paired architecture comparison from two --save-folds CSVs (manuscript 3.7 gap).

Each CSV is written by baselines.py --save-folds and has rows
(model, idx, acc, bal, prog, f1, auprc), where `idx` enumerates (seed, fold) in
the SAME order across runs as long as --seeds/--kfolds/--group-split match. So the
two files' kg_gnn rows are aligned by idx and a paired Wilcoxon signed-rank test is
valid (it controls for per-fold difficulty).

Usage:
    python scripts/compare_archs.py results/resistance_folds.csv results/toggle_folds.csv \
        --labels resistance toggle --model kg_gnn --metric auprc
"""
from __future__ import annotations

import argparse
import csv


def _load(path, model):
    rows = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if r["model"] == model:
                rows.append((int(r["idx"]), float(r["acc"]), float(r["bal"]),
                             float(r["prog"]), float(r["f1"]), float(r["auprc"])))
    rows.sort()                                   # align by (seed,fold) index
    return rows


def main():
    ap = argparse.ArgumentParser(description="Paired arch-vs-arch test from two save-folds CSVs")
    ap.add_argument("csv_a")
    ap.add_argument("csv_b")
    ap.add_argument("--labels", nargs=2, default=["A", "B"])
    ap.add_argument("--model", default="kg_gnn", help="model row to compare (default kg_gnn)")
    ap.add_argument("--metric", default="auprc", choices=["acc", "bal", "prog", "f1", "auprc"])
    args = ap.parse_args()

    col = {"acc": 1, "bal": 2, "prog": 3, "f1": 4, "auprc": 5}[args.metric]
    a, b = _load(args.csv_a, args.model), _load(args.csv_b, args.model)
    la, lb = args.labels

    ia = {r[0]: r[col] for r in a}
    ib = {r[0]: r[col] for r in b}
    shared = sorted(set(ia) & set(ib))
    if not shared:
        raise SystemExit("no overlapping (seed,fold) indices -- were the runs' "
                         "--seeds/--kfolds/--group-split identical?")
    if len(shared) != len(ia) or len(shared) != len(ib):
        print(f"WARNING: {la} has {len(ia)}, {lb} has {len(ib)}, "
              f"{len(shared)} overlap -- comparing the overlap only.")

    va = [ia[i] for i in shared]
    vb = [ib[i] for i in shared]
    mean_a, mean_b = sum(va) / len(va), sum(vb) / len(vb)
    diffs = [x - y for x, y in zip(va, vb)]
    med = sorted(diffs)[len(diffs) // 2]

    print(f"model={args.model}  metric={args.metric}  paired over {len(shared)} (seed,fold)")
    print(f"  {la:<16} mean = {mean_a:.3f}")
    print(f"  {lb:<16} mean = {mean_b:.3f}")
    print(f"  mean diff ({la}-{lb}) = {mean_a - mean_b:+.3f}   median paired diff = {med:+.3f}")

    try:
        from scipy.stats import wilcoxon
        if any(d != 0 for d in diffs):
            p = float(wilcoxon(va, vb).pvalue)
            verdict = "SIGNIFICANT" if p < 0.05 else "not significant"
            print(f"  paired Wilcoxon signed-rank p = {p:.4f}  ({verdict} at 0.05)")
        else:
            print("  all paired diffs are 0 -> no test")
    except ImportError:
        print("  (install scipy for the paired Wilcoxon p-value)")


if __name__ == "__main__":
    main()
