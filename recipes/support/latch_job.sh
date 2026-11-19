#!/bin/bash
# latch_job.sh: run ONE named LATCH recipe on ONE checkpoint, then measure the recipe's own output.
# Weekend run 3, Part 2. Extends latch_recipe.sh (Part 1; same LATCH command line) with:
#   * LATCH_JAVA_HOME selects the JRE for LATCH (RapidWright via JPype); recorded in java_used.txt;
#   * the transition output (the recipe's own result, kept even if LATCH rejects it) is re-opened in a
#     separate Vivado batch session and measured with code/measure.tcl (tag "final", reports/ in this dir).
# Needs code/env.sh sourced first (runner3.py does this). Run inside a fresh runs/<run_id>/ directory.
# usage: latch_job.sh <input.dcp> <flow>
# writes: input/<name>.dcp (private copy) + input/sha256.txt, latch_run/, out.dcp (LATCH's selection),
#         transition_outputs.txt, recipe_out.dcp (symlink to the transition output), reports/final_*,
#         java_used.txt, latch_exit.txt, measure_exit.txt
set -u
IN="$1"; FLOW="$2"
export JAVA_HOME="${LATCH_JAVA_HOME:-$JAVA_HOME}"
export PATH="$JAVA_HOME/bin:$PATH"
{ echo "JAVA_HOME=$JAVA_HOME"; "$JAVA_HOME/bin/java" -version 2>&1; } > java_used.txt
mkdir -p input
LOCAL="$PWD/input/$(basename "$IN")"
cp -p "$IN" "$LOCAL"
sha256sum "$IN" "$LOCAL" > input/sha256.txt
"$LATCH_PY" "$LATCH_ROOT/dcp_optimizer.py" "$LOCAL" --no-dotenv --pipeline fixed --plan "$FLOW" --budget-runs 1 \
    --output "$PWD/out.dcp" --run-dir "$PWD/latch_run"
rc=$?
echo "$rc" > latch_exit.txt
find "$PWD/latch_run" -path '*transition_000_*' -name output.dcp > transition_outputs.txt
T=$(head -1 transition_outputs.txt)
if [ -n "$T" ] && [ -f "$T" ]; then
  ln -s "$T" recipe_out.dcp
  cp "$RUN3/code/run1/lib_measure.tcl" "$RUN3/code/lib_run3.tcl" "$RUN3/code/measure2.tcl" .
  # RESET_UNC=1 (nudged-input runs) resets the setup clock uncertainty before measuring
  vivado -mode batch -nojournal -log measure_vivado.log -source measure2.tcl -tclargs "$T" . final 0 "${RESET_UNC:-0}" \
      > measure_stdout.txt 2>&1
  echo $? > measure_exit.txt
else
  echo "no transition output" > measure_exit.txt
fi
exit $rc
