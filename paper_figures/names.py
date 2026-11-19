"""Display names for every DAC 2027 figure (style guide section 1, draft of 16 Nov 12:40).

No figure script spells a name itself: every label a reader sees comes from this module. Internal codes
(T0, V9, goal-repair v2 ...) appear here as dictionary keys only and never in a figure.
"""
from __future__ import annotations

ARROW = " → "

GROUPS = ("TRAINING", "VALIDATION", "TEST")
SETTINGS = ("2worker", "3worker")
SETTING_WORDS = {"2worker": "two workers", "3worker": "three workers"}
SETTING_BRIEFS = {"2worker": ("T0", "A0"), "3worker": ("W2R", "L2R", "P0")}

# internal brief code -> (paper name, short label used in table headers)
BRIEF = {
    "T0": ("Timing-first", "timing"),
    "A0": ("Area-first", "area"),
    "W2R": ("Timing-first (3 workers)", "timing"),
    "L2R": ("Area-first (3 workers)", "area"),
    "P0": ("Power-first", "power"),
    "T": ("Timing-first, relaxed", "timing (relaxed)"),
    "A": ("Area-first, relaxed", "area (relaxed)"),
}
# which quality column a brief fills in a summary table
BRIEF_COLUMN = {"T0": "g_t", "W2R": "g_t", "A0": "g_a", "L2R": "g_a", "P0": "g_p"}
COLUMN_BRIEFS = {"2worker": {"g_t": "T0", "g_a": "A0"}, "3worker": {"g_t": "W2R", "g_a": "L2R", "g_p": "P0"}}

# workers: A is the timing worker, B the area worker, C the power worker
WORKER = {"A": "timing", "B": "area", "C": "power"}


def worker_order(code: str) -> str:
    """'BA' -> 'area -> timing' (orders are always written with arrows)."""
    return ARROW.join(WORKER[c] for c in code)


# policy rows, in the order they appear in every summary table
POLICY_ORDER = ("V9", "v2", "fixed", "bandit", "beam", "grid", "random")
CONTROL_ORDER = ("GA-R1",)
POLICY = {
    "V9": {"name": "LLM Orchestrator", "sub": "Gemini 3.1 Flash-Lite", "llm": True},
    "v2": {"name": "Rule-Based Policy", "sub": "improve, then repair limits", "llm": False},
    "fixed": {"name": "Fixed Worker Order", "sub": None, "llm": False},      # second line = the order, filled from the data
    "bandit": {"name": "Learned Bandit", "sub": "contextual bandit", "llm": False},
    "beam": {"name": "Beam Search", "sub": "width 2, depth 3", "llm": False},
    "grid": {"name": "Grid Search", "sub": None, "llm": False},
    "random": {"name": "Random Policy", "sub": "mean of 3 seeds", "llm": False},
    "GA-R1": {"name": "Direct LLM Tool Control", "sub": "no orchestration", "llm": True},
}
# reference rows (sentence case, italic in the figure); second lines that depend on counts are built from the data
REFERENCE = {
    "ref3": {"name": "Exhaustive search, 3 steps", "sub": None},
    "ref2": {"name": "Exhaustive search, 2 steps", "sub": None},
    "ref2p": {"name": "Exhaustive search, 2 steps, 14 recipes", "sub": None},
    "ceiling": {"name": "Hindsight best, 6 runs", "sub": "best 6-run path, all outcomes known"},
    "bestknown": {"name": "Best known result", "sub": None},
}
DAGGER = "†"


def dagger(setting: str, group: str, row_key: str) -> bool:
    """A row chosen or fitted on the group shown carries a dagger."""
    if row_key == "fixed" and group == "TRAINING":
        return True                                  # the fixed order was chosen on TRAINING
    if row_key == "V9" and setting == "2worker" and group in ("TRAINING", "VALIDATION"):
        return True                                  # the prompt was selected on TRAINING and VALIDATION (never on power briefs)
    return False


# table columns: id -> (header line 1, header line 2 or None)
COLUMN = {
    "g_t": ("WNS gain (ps)", "timing-first"),
    "g_a": ("LUTs removed", "area-first"),
    "g_p": ("Power saved (mW)", "power-first"),
    "runs": ("Tool runs / task", None),
    "jobh": ("Vivado job-h / task", None),
    "tokens": ("LLM tokens / task", None),
    "usd": ("LLM cost / task (US$)", None),
    "pct": ("% of best-known gain", "pooled ± sd"),
}
COST_COLUMNS = ("runs", "jobh", "tokens", "usd")
SPANNER = {
    "quality": "Quality (mean over designs where the best-known result improves beyond noise)",
    "cost": "Cost per task",
    "overall": "Overall",
}
PCT_SUB = {"2worker": "(timing / area)", "3worker": "(timing / area / power)"}

NOT_RUN = "not run"
NOT_APPLICABLE = "—"
NO_VALUE = "n/a"


def title(setting: str, group: str, reference: str) -> str:
    """Figure title: the group, the setting and what the table compares (at most two lines, 40 words)."""
    ref = {"exhaustive": "exhaustive search", "best_known": "the best known result"}[reference]
    return f"{group} designs, {SETTING_WORDS[setting]}: quality and cost of each orchestration policy against {ref}"


def figure_id(setting: str, group: str) -> str:
    return f"summary_{setting}_{group}"


# ======================================================================================================================================================
# FINAL RESULTS TABLES (Mon 16 Nov 2026, "four results tables" prompt). Added below the earlier content, which is unchanged so that the six summary
# figures and the per-benchmark figure still rebuild byte for byte (their hashes are in CODE.sha256). The tables of final/ read only this block.
# ======================================================================================================================================================
FINAL_POLICY = {
    "exh": {"name": "Exhaustive Search"},
    "direct": {"name": "Direct LLM Tool Control", "sub": "no orchestration"},
    "orch": {"name": "LLM-Orchestrator"},                  # second line = the model name (FINAL_LLM_MODELS)
    "rule": {"name": "Rule-Based Policy"},
    "fixed": {"name": "Fixed Worker Order"},               # second line = the order (final_fixed_order)
    "bandit": {"name": "Learned Bandit"},
    "grid": {"name": "Grid Search"},
}
FINAL_LLM_MODELS = {"gemini": "Gemini 3.1 Flash-Lite", "gpt": "GPT-5.6 Terra", "qwen": "Qwen3.5-397B"}
# internal policy key of final/ -> key of the earlier modules
FINAL_TO_EARLIER = {"exh": "ref3", "direct": "GA-R1", "orch": "V9", "rule": "v2", "fixed": "fixed", "bandit": "bandit", "grid": "grid"}

FINAL_DESIGN_SHORT = {                                      # same strings as names_per_benchmark.DESIGN_SHORT (asserted in final_data)
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
REPRESENTATIVE_DROP = ("logicnets_jscl", "fir_symmetric_systolic", "vexriscv_re-place_v2")


def final_fixed_order(setting: str, brief: str) -> str:
    """The order of the Fixed Worker Order policy: two workers area -> timing; three workers timing -> area -> power, power -> area -> timing on power-first."""
    if setting == "2worker":
        return worker_order("BA")
    return worker_order("CBA") if brief == "P0" else worker_order("ABC")


FINAL_BRIEF_WORD = {"T0": "timing-first", "W2R": "timing-first", "A0": "area-first", "L2R": "area-first", "P0": "power-first"}
FINAL_BRIEF_METRIC = {"T0": "WNS gain (ps)", "W2R": "WNS gain (ps)", "A0": "LUTs removed", "L2R": "LUTs removed", "P0": "Power saved (mW)"}
FINAL_NOT_RUN = "not run"
FINAL_NA = "n/a"
FINAL_DASH = "—"
FINAL_PENDING = "pending"
