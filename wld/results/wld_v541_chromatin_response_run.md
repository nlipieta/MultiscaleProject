# WLD v5.4.1 verified development run

This record captures the successful response-calibrated chromatin-development
run produced from source commit
`f10ac0584dde0728a2cbf990bd469d8b68564990`.

## Scope

- 73 perturbation targets were used for training.
- 16 complete perturbation targets were used for validation.
- The 16 test targets, muscle subjects J/L, and external test studies remained
  sealed.
- Target identity selected a named intervention after encoding and was not an
  encoder feature.
- The experiment used single-endpoint CRISPR-sciATAC observations. It does not
  identify ODE kinetics, fixed points, basins, or attractors.

## Validation result

| Quantity | Value |
| --- | ---: |
| True-route SWD | 0.024337 |
| Persistence SWD | 0.024389 |
| Gain over persistence | +0.000052 |
| Response NRMSE | 0.999377 |
| Response cosine | 0.024717 |
| Advantage over retrained degree shuffles | +0.000068 |
| Frozen-zero route effect | +0.000052 |

The fitted checkpoint has a small, detectable dependence on its supported
routes. The response NRMSE and cosine do not support a claim of useful
perturbation prediction, biological circuit specificity, or attractor
dynamics.

## Artifact policy

Code, contracts, the exact restart-safe launcher, and this compact run record
belong in Git. Large matrices, compiled priors, checkpoints, and complete logs
remain in Google Drive under
`/content/drive/MyDrive/WLD_Backup/wld_v541_response_calibrated`.
