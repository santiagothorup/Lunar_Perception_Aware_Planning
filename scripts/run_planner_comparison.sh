#!/bin/bash
# Run one or more perception-aware planner variants end-to-end in the LAC sim and capture SLAM RMSE
# + mobility. Each variant is wall-clock-bounded so a stuck one can't stall the suite; the sim is
# cleaned between variants. Results append to output/planner_comparison/results.txt and the labeled
# run dirs are recorded in run_dirs.txt for scripts/compare_runs.py.
#
# Usage: scripts/run_planner_comparison.sh <missions_subset> <timeout_s> <label:config.json> [<label:config.json> ...]
#   e.g. scripts/run_planner_comparison.sh 6 5400 baseline:perception_aware_baseline.json perc_trav:perception_aware_trav.json
cd /home/santiagothorup/Documents/Lunar_Perception_Aware_Planning
PY=/home/santiagothorup/miniconda3/envs/lac/bin/python
export PYTHONNOUSERSITE=1

# If the FINAL_RUNS env var is set, write results.txt + run_dirs.txt to
# LAC_SIM/output/NavAgent/final_runs/ and move each variant's run dir to
# LAC_SIM/output/NavAgent/final_runs/<label>/ for the presentation set.
# Otherwise, default to output/planner_comparison/ as before.
if [ -n "${FINAL_RUNS:-}" ]; then
    OUT=LAC_SIM/output/NavAgent/final_runs
    mkdir -p "$OUT"
else
    OUT=output/planner_comparison
    mkdir -p "$OUT"
fi
R="$OUT/results.txt"

SUBSET=$1; TO=$2; shift 2
echo "================ PLANNER COMPARISON $(date)  subset=$SUBSET timeout=${TO}s ================" >> "$R"

kill_sim() { pkill -9 -f "[L]AC-Linux-Shipping" 2>/dev/null; pkill -9 -f "[l]eaderboard_evaluator.py" 2>/dev/null; rm -f /tmp/.X99-lock; sleep 5; }

for arg in "$@"; do
  label=${arg%%:*}; cfg=${arg##*:}
  kill_sim
  echo "[$label] START $(date)  cfg=$cfg" >> "$R"
  TEAM_AGENT="$PWD/agents/perception_aware_agent.py" TEAM_CONFIG="$PWD/configs/$cfg" MISSIONS_SUBSET="$SUBSET" \
    timeout "$TO" bash launch_headless.sh > "LAC_SIM/logs/cmp_${label}.log" 2>&1
  rc=$?
  # Only consider timestamp-style dirs (avoids picking up final_runs/ itself).
  d=$(ls -dt LAC_SIM/output/NavAgent/[0-9]*/ | head -1)
  # If in FINAL_RUNS mode, rename the run dir from <timestamp>/ to final_runs/<label>/.
  if [ -n "${FINAL_RUNS:-}" ] && [ -n "$d" ]; then
    target="LAC_SIM/output/NavAgent/final_runs/${label}"
    rm -rf "$target"  # if rerunning, clear stale
    mv "${d%/}" "$target" && d="$target/"
  fi
  echo "[$label] END   $(date)  dir=$d  (timeout/exit=$rc)" >> "$R"
  printf '%s\t%s\n' "$label" "$d" >> "$OUT/run_dirs.txt"
  rmse=$(grep "Final RMSE" "LAC_SIM/logs/cmp_${label}.log" | tail -1)
  diag=$($PY scripts/plot_run_diagnostics.py "$d" 2>/dev/null)
  mob=$(echo "$diag" | grep "steps:")
  extra=$(echo "$diag" | grep "anchors:")
  echo "  RESULT [$label]  ${rmse:-(no RMSE - timed out?)}  | ${mob:-(no mobility)}  | ${extra:-(no v1c diagnostics)}" >> "$R"
done

kill_sim
echo "================ COMPARISON COMPLETE $(date) ================" >> "$R"
echo "results -> $R ; overlay -> run scripts/compare_runs.py with run_dirs.txt entries"
