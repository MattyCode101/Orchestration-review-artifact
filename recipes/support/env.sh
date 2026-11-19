# env.sh: environment for the recipes and the validator. Source it (". recipes/support/env.sh"); do not execute it.
# Set these before sourcing:
#   ORCH_VIVADO_ROOT         Vivado 2025.1 installation (the directory that holds Vivado/)
#   ORCH_RECIPE_TOOLS_DIR    tree with the published timing-recipe checkout (LATCH), its tool environments and the validator
#   ORCH_RAPIDWRIGHT_ROOT    RapidWright tree used by the validator
#   ORCH_VALIDATOR_PYTHON    interpreter of the contest validator environment

export RECIPE_TOOLS="${ORCH_RECIPE_TOOLS_DIR:?set ORCH_RECIPE_TOOLS_DIR}"

# Vivado 2025.1 (older versions cannot open the 2025.1 contest checkpoints).
source "${ORCH_VIVADO_ROOT:?set ORCH_VIVADO_ROOT}/Vivado/settings64.sh"
export VIVADO_EXEC="$ORCH_VIVADO_ROOT/Vivado/bin/vivado"
export VIVADO_BIN="$VIVADO_EXEC"

# Java for RapidWright (needs >= 11): Vivado's bundled JRE 11. The published timing recipes were tested with the bundled Java 21.0.1.
export JAVA_HOME="$ORCH_VIVADO_ROOT/Vivado/tps/lnx64/jre11.0.16_1"
export PATH="$JAVA_HOME/bin:$PATH"

# RapidWright (prebuilt tree, used read-only).
export RAPIDWRIGHT_PATH="${ORCH_RAPIDWRIGHT_ROOT:?set ORCH_RAPIDWRIGHT_ROOT}"
export CLASSPATH="$RAPIDWRIGHT_PATH/bin:$RAPIDWRIGHT_PATH/jars/*"

# Python
export LATCH_ROOT="$RECIPE_TOOLS/LATCH"                    # the published timing-recipe checkout, commit c53a8bf
export LATCH_PY="$RECIPE_TOOLS/venv_latch/bin/python"
export ANALYSIS_PY="$RECIPE_TOOLS/venv_analysis/bin/python"
export PYTHONDONTWRITEBYTECODE=1

# Keep tool caches inside the tool tree
export UV_CACHE_DIR="$RECIPE_TOOLS/.uv_cache"
export MPLCONFIGDIR="$RECIPE_TOOLS/.mplconfig"

# No model is reachable: the recipes run without an LLM
unset OPENROUTER_API_KEY OPENAI_API_KEY OPENAI_BASE_URL ANTHROPIC_API_KEY GEMINI_API_KEY GOOGLE_API_KEY

# Contest validator: upstream contest repository, copied to $RECIPE_TOOLS/validator/.
export VALIDATOR="$RECIPE_TOOLS/validator/validate_dcps.py"
export VALIDATOR_PY="${ORCH_VALIDATOR_PYTHON:?set ORCH_VALIDATOR_PYTHON}"
