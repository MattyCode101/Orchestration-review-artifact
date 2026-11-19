"""Names the per-benchmark breakdown figure needs and names.py does not hold (short design names, header lines, spanner words).

Kept in a separate module on purpose: names.py and figstyle.py are being edited by another script at the same time, and a second writer
to those files would lose edits. Everything that names.py already defines is imported from it, never repeated. When the two figures are merged,
the dictionaries below can move into names.py unchanged.
"""
from __future__ import annotations

import names as N

# short design names for the row labels (the repository names are the keys); ordered inside a group by start LUT count in the figure
DESIGN_SHORT = {
    "vexriscv_re-place": "vexriscv",
    "rosetta_spam-filter": "spam-filter",
    "rosetta_3d-rendering": "3d-rendering",
    "logicnets_jscl": "logicnets",
    "finn_radioml": "finn",
    "t2_cordic_s": "cordic",
    "amd_mini-isp": "mini-isp",
    "vtr_mcml_v2": "mcml",
    "fir_systolic_transposed": "fir transposed",
    "fir_symmetric_systolic": "fir symmetric",
    "vexriscv_re-place_v2": "vexriscv v2",
    "rosetta_digit-recognition": "digit-recognition",
    "corescore_500_mod": "corescore",
}

# policy columns, in the order of every block: (column key, key into names.POLICY or None, header lines). The Best known column is a reference, not a policy.
POLICY_COLUMNS = (
    ("best", None, ("Best", "known")),
    ("gemini", "V9", ("Gemini 3.1", "Flash-Lite")),
    ("gpt", None, ("GPT-5.6", "Terra")),
    ("qwen", None, ("Qwen3.5-", "397B")),
    ("direct", "GA-R1", ("Direct LLM", "Tool Control")),
    ("rule", "v2", ("Rule-Based", "Policy")),
    ("fixed", "fixed", ("Fixed Worker", "Order")),
    ("bandit", "bandit", ("Learned", "Bandit")),
    ("beam", "beam", ("Beam", "Search")),
    ("grid", "grid", ("Grid", "Search")),
    ("random", "random", ("Random", "Policy")),
)
for _key, _nkey, _lines in POLICY_COLUMNS:           # the header lines must spell the names of names.py
    if _nkey is not None and _key != "gemini":
        assert " ".join(_lines) == N.POLICY[_nkey]["name"], (_key, _lines, N.POLICY[_nkey]["name"])
LLM_COLUMNS = ("gemini", "gpt", "qwen")
LLM_SPANNER = N.POLICY["V9"]["name"]                  # LLM Orchestrator
LLM_MODELS = {"gemini": "Gemini 3.1 Flash-Lite", "gpt": "GPT-5.6 Terra", "qwen": "Qwen3.5-397B"}
BEST_NAME = "Best known"
# The compact variant leaves out the columns that have data on TEST only: Beam Search and Grid Search (TEST, 1 repeat) and the two extra LLM Orchestrator
# models GPT-5.6 Terra and Qwen3.5-397B (TEST only). The LLM Orchestrator column left is the Gemini 3.1 Flash-Lite one, the model with data on all 13 designs.
COMPACT_DROP = ("gpt", "qwen", "beam", "grid")
LLM_SINGLE_HEADER = ("LLM", "Orchestrator")           # header of the LLM Orchestrator column when only one model is shown
assert " ".join(LLM_SINGLE_HEADER) == N.POLICY["V9"]["name"]

START_HEADERS = {"wns": ("WNS", "(ps)"), "luts": ("Packed", "LUTs"), "mw": ("Dynamic", "power (mW)")}
START_SPANNER = "Start values"
BRIEF_UNIT = {"T0": "ps", "W2R": "ps", "A0": "LUTs", "L2R": "LUTs", "P0": "mW"}
BRIEF_COLUMN = {"T0": "g_t", "W2R": "g_t", "A0": "g_a", "L2R": "g_a", "P0": "g_p"}
SECTION_WORDS = {"T0": "Two workers", "A0": "Two workers", "W2R": "Three workers", "L2R": "Three workers", "P0": "Three workers"}


def brief_spanner(b: str) -> str:
    """'Two workers: timing-first, WNS gain (ps)' from the names of names.py."""
    first = N.BRIEF[b][0].replace(" (3 workers)", "").lower()
    col = N.COLUMN[BRIEF_COLUMN[b]][0]
    if not col[:2].isupper():                         # 'Power saved (mW)' -> 'power saved (mW)'; 'WNS gain (ps)' and 'LUTs removed' stay
        col = col[0].lower() + col[1:]
    return f"{SECTION_WORDS[b]}: {first}, {col}"


OVERALL_SPANNER = "Overall: % of best-known gain, pooled over the design's headroom tasks (all five briefs)"
TOTAL_LABEL = {"all": "Total, all designs", "TEST": "Total, TEST"}
NOT_RUN_LINES = tuple(N.NOT_RUN.split())              # ('not', 'run')

TITLE = ("Per-benchmark results: gain of each policy over the starting checkpoint, against the best known result",
         "TRAINING (5 designs), VALIDATION (3 designs) and TEST (5 designs), two and three workers")
