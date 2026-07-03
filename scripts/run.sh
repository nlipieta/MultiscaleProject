#!/usr/bin/env bash
# One-shot: set up the environment, train, and run a few example predictions.
set -euo pipefail
cd "$(dirname "$0")/.."

uv sync
uv run chromatin-train
echo
uv run chromatin-predict --context myoblast   --cue TGFbeta
uv run chromatin-predict --context epithelial --cue MechanicalStiffness
uv run chromatin-predict --context xenopus    --cue BioelectricDepolarization
