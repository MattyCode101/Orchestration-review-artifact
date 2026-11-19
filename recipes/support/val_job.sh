#!/bin/bash
# val_job.sh: run AMD's validate_dcps.py (unmodified upstream copy in code/validator, @0a58034) on golden vs revised.
# Weekend run 3, Part 2. A WEAK check (run 1: 1 of 12 single-LUT breaks caught at 200 vectors). Keeps the verdict,
# the validator's report and top-level logs in this job dir, then deletes its working folder, which it creates next
# to itself inside the run folder (allowed by the user once the result is recorded).
# usage: val_job.sh <golden.dcp> <revised.dcp> [vectors]   (code/env.sh sourced; run inside runs/val_<target>/)
set -u
G="$1"; RV="$2"; V="${3:-200}"
"$VALIDATOR_PY" -B "$VALIDATOR" "$G" "$RV" --vectors "$V" > validate.log 2>&1
rc=$?
echo "VALIDATOR_RC=$rc" >> validate.log
W=$(grep -m1 -o 'Working directory: .*' validate.log | sed 's/^Working directory: //')
if [ -n "$W" ] && [ -d "$W" ]; then
  case "$W" in
    "$RUN3/code/validator/dcp_validation_"*)
      mkdir -p validator_work
      cp -p "$W"/*.json "$W"/*.log validator_work/ 2>/dev/null
      du -sh "$W" > validator_work/workdir_size.txt
      rm -rf -- "$W"
      echo "deleted validator workdir $W" >> validate.log ;;
  esac
fi
exit $rc
