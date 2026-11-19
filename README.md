# Orchestration framework for PPA-driven FPGA implementation: code and stored results

This repository is the anonymised code release for the paper "Physically Aware Orchestration of Specialized Optimizers for PPA-Driven FPGA Implementation" (double-blind submission). It holds the orchestration framework (the contract between a policy and the optimisation workers, the policies compared in the paper, the Vivado recipes), the stored episode results the paper's tables are computed from, and the scripts that rebuild every table and every number quoted in the text.

The authors' own work is everything in this repository except the third-party material listed under "Third-party material". The timing recipes of LATCH are prior work by others and are used unmodified; they are not part of this repository.

## What can be reproduced without Vivado

Three things run on a laptop with Python only, from the stored results shipped here:

1. the test suite (`python -m pytest`);
2. every table of the paper (Tables III to VI), regenerated from the stored episode results;
3. every number the paper's text states outside a table (percentages, ratios, counts).

Running the optimisation workers (Vivado recipes) and the LLM policies again needs Vivado 2025.1, the contest benchmarks, and model API keys; see "Running the policies again".

## Layout

| Folder | Content |
|---|---|
| `orch/` | the framework: `contract/` (the typed boundary between policy and workers), `engine/` (saved-state tree, journal, validation), `workers/`, `recipes/` (registry loader and adapters), `policies/` (all compared policies), `episode.py`, `episode_r2.py` |
| `orch/policies/` | `v2_*` Rule-Based Orchestration; `v22_*`, `r2_*`, `llm*.py`, `round2_harness.py` LLM Orchestration; `bandit_r2.py`, `bandit_frozen.py` Learned Bandit; `trivial.py`, `random_policy.py`, `training_baselines.py` baselines |
| `recipes/` | `registry.yaml` (78 registered recipes), `tcl/` (every Vivado Tcl recipe, byte-identical to the research copy), `support/env.sh` (environment for Vivado, set from variables), `name_map.csv` (registry id to the recipe names of Table II) |
| `configs/` | design split, worker sets, fitted worker parameters, prompt templates, LLM settings |
| `scripts/` | campaign, episode and analysis drivers, grouped by campaign; `fetch_benchmarks.py` downloads the contest benchmarks |
| `data/` | stored results needed by the tests and by the drivers (about 1,060 small files) |
| `paper_figures/` | the single table pipeline: `inputs/` (stored results the tables are computed from), `final/` (generator and independent verifier), `make_tables.py` (paper layout), `paper_claims.py` (in-text numbers) |
| `tests/` | the test suite (1,038 cases) |
| `tools/llm_bridge/` | the model-API bridge (keys are read from the environment) |
| `notes/` | only the hash pins and records that code and tests read; planning notes are not shipped, so a few docstrings cite notes that are not in this repository |

## Requirements

- Python 3.12 for the framework and the tests: `pip install pydantic pyyaml pandas pytest` (versions pinned in `pyproject.toml` and `uv.lock`).
- The table scripts also use `matplotlib`; they were run with Python 3.9, pandas 2.3.3 and matplotlib 3.9.4.
- To run the optimisation workers: Vivado 2025.1 (device `xcvu3p-ffvc1517-2-e`, the device of the contest benchmarks), the RapidWright tree and validator of the contest, and the published LATCH checkout.
- To run the LLM policies: `GOOGLE_API_KEY` (or `GEMINI_API_KEY`) for Gemini 3.1 Flash-Lite; `LLM_GATEWAY_API_KEY` and `LLM_GATEWAY_BASE_URL` for the two other models evaluated. No key, token or credential is stored in this repository.

## Reproducing the tables and the in-text numbers

From the repository root:

```
cd paper_figures
python final/final_tables.py     # writes the six tables and their values files to paper_figures/out/final/
python final/final_verify.py     # recomputes every displayed cell independently from the 905 stored episodes
cd ..
python paper_figures/make_tables.py    # lays out Tables III to VI of the paper in paper/tables/
python paper_figures/paper_claims.py   # recomputes the numbers quoted in the text: stated value, computed value, agreement
```

`scripts/run_tables.sh` runs the four steps in order. One generator writes every table; the values behind each cell, with the file and column they come from, are in `paper_figures/out/final/values_*.csv`.

| Paper table | Generated file |
|---|---|
| Table III (two workers, summary) | `paper/tables/tab_summary_2worker.tex` |
| Table IV (two workers, per design) | `paper/tables/tab_breakdown_2worker.tex` |
| Table V (three workers, summary) | `paper/tables/tab_summary_3worker.tex` |
| Table VI (three workers, per design) | `paper/tables/tab_breakdown_3worker.tex` |

Notes on the stored results:

- The Learned Bandit timing-first cell of Table V is a post-freeze row: its prior was fitted after the other test rows were known, from training and validation records only. It is computed from `paper_figures/inputs/bandit_w2r_cells.json` (three headroom designs, one deterministic episode each) and marked `d` in the table.
- Exhaustive Search on three workers has no per-run record for the 14-recipe search. Its runs and job-hours in Table V are estimates, scaled from the measured two-step search over ten recipes (82 runs and 17.97 job-hours per design, times (14x15)/(10x11)); both constants are stated in `paper_figures/make_tables.py`, and the measured records behind them are not shipped.
- `paper_claims.py` cannot recompute one statement of the text: that in 10 of the 16 exhaustive search runs on test designs a limit is broken on purpose before a later worker repairs it. The shipped inputs hold the recipe sequence of each exhaustive path but not the metrics of its intermediate states.

## Hash pins

The research code pins the contents of some frozen files by SHA-256. After the anonymisation edits (identifying names and paths removed, wording of comments changed) the pinned files changed, so every pin was recomputed in this release, in this release only. The behaviour is unchanged: the test suite passes, and every table regenerates from the stored inputs with the same values as in the research tree (1,073 cells compared, 0 differences).

## Tests

`python -m pytest` runs 1,038 cases. A few skip with an explicit reason because they need records that are not shipped (the sealed planning notes that declare a freeze, and the large stored search records); the reason is printed with `-rs`.

## Running the policies again

An episode is one policy run on one design and one brief. `scripts/test_r2_2026-11-13/tr_episode.py --key EPISODE_ID` runs or resumes one planned episode of the test campaign (`tr_queue.py` starts one such process per planned episode). It needs Vivado 2025.1 and the contest benchmarks (below). Source `recipes/support/env.sh` after setting `ORCH_VIVADO_ROOT`, `ORCH_RECIPE_TOOLS_DIR`, `ORCH_RAPIDWRIGHT_ROOT` and `ORCH_VALIDATOR_PYTHON`; `ORCH_BENCHMARK_DIR` points to the benchmark folder. LLM policies spend model credit; each campaign writes a spend ledger with a hard stop.

## Third-party material

No benchmark design, checkpoint or third-party tool is redistributed here.

- Contest benchmarks (FPL 2026 FPGA optimisation contest, repository `Xilinx/fpl26_optimization_contest`, Apache-2.0, release v1.3.0): download with `python scripts/fetch_benchmarks.py <target folder>`, which verifies the archive against SHA-256 `14d66b7b32171f20be9aa26eb395fdb75b881555854dfa2fa7b870f30d8b71ac` (the digest GitHub publishes for the release asset).
- Licences of the designs used, as listed in the archive's `LICENSE-BENCHMARKS.txt` files: Apache-2.0: corescore, logicnets; BSD-3-Clause: finn, rosetta spam-filter, rosetta 3d-rendering, rosetta digit-recognition; MIT: vtr mcml, amd mini-isp, the two FIR designs, the two vexriscv designs. Check the files in the archive before redistributing any design.
- cordic (a validation design) is a rebuilt design of unconfirmed licence. Its checkpoint is not included; only its stored results are.
- Start checkpoints of the designs are Vivado default implementations of the benchmarks (derived files); they are not redistributed.
- LATCH (the published timing recipes) is prior work by others, used unmodified and fetched separately.
- Vivado, RapidWright and the model APIs are used as external tools under their own licences.

No licence file or citation block is included in this review copy.
