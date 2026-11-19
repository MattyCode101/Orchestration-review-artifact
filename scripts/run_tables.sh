#!/bin/bash
# Regenerate Tables III-VI and every in-text number from the stored results. Run from anywhere; needs pandas and matplotlib.
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/paper_figures"
python final/final_tables.py
python final/final_verify.py
cd "$ROOT"
python paper_figures/make_tables.py
python paper_figures/paper_claims.py
