# WLD v5.6 practical-effect audit contract

The completed v5.6 development report used strict comparisons with zero and
one. Consequently, floating-point remnants could satisfy every Boolean gate
even when the fitted model was indistinguishable from persistence. This audit
is a provenance-linked sidecar; it does not rewrite the completed report.

The audit:

- reads only `wld_v56_null_aware_development_report.json`;
- rejects reports that opened a sealed target or made a digital-twin or
  attractor claim;
- verifies the complete reused-target-by-seed grid;
- preserves every raw metric and reports absolute and comparator-relative
  effects;
- uses `1e-6` as a numerical decision tolerance;
- requires, prospectively, at least 2% mean relative improvement over
  persistence and at least 1% over the perturbed-mean, matched-control, and
  frozen-route comparators;
- requires those gains on 75%, 75%, 75%, and 60% of development targets,
  respectively, with the persistence gain positive in every seed;
- validates an exact table of at least ten matched controls and requires the
  true topology to beat every control beyond the `1e-6` numerical tolerance;
- requires at least eight route-supported targets, at least 50% of target/seed
  rows above the predeclared response-amplitude floor, and at least 50% of
  targets above that floor in a majority of seeds;
- computes response gates only above that floor and requires mean/median NRMSE
  at most 0.90/0.95 and mean/median response cosine at least 0.20/0.10; and
- never authorizes opening the sealed test. A successful audit would only
  justify freezing a separate prospective confirmation plan.

These values are development-viability thresholds. They are not p-values,
confidence limits, or universal biological constants. The
reused v5.5 validation targets remain development data, so no inferential claim
is permitted.
