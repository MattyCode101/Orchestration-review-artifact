"""Settings of the four final results tables (one place; nothing else in final/ spells these)."""
import os

# Which model the LLM-Orchestrator column shows. "gemini" = Gemini 3.1 Flash-Lite everywhere (the rendered version);
# "best" = per setting, the model with the highest pooled % of best-known gain on TEST (computed in final_data.best_models, never shown).
LLM_MODEL = os.environ.get("FINAL_LLM_MODEL", "gemini")

# Cost rows (Vivado job-h, LLM tokens) are not bolded: a policy that gains nothing can be the cheapest.
BOLD_COST_ROWS = False

# Output folder (relative to paper_figures/) and the stem prefix of every file.
OUT_DIR = os.environ.get("FINAL_OUT", "out/final")

# Text sizes in pt (LaTeX \fontsize and the matplotlib preview use the same numbers). The per-benchmark tables never go below 6 pt.
SUMMARY_FONT = (8.0, 7.0)          # (main, second line) of the two summary tables
BREAKDOWN_FONTS = ((8.0, 7.0), (7.5, 6.5), (7.0, 6.0), (6.5, 6.0))   # tried in this order; the first that fits 7.0 in is used (never below 6 pt)
FULL_WIDTH_IN = 7.0
SINGLE_COLUMN_IN = 3.4

assert LLM_MODEL in ("gemini", "best"), LLM_MODEL
