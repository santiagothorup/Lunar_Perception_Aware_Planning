#!/bin/bash
# launch_phase0.sh - collect Phase 0 transect data on preset 2 via the headless launcher.
#
# Drives agents/phase0_collection_agent.py (the phase0_transect raster) instead of nav_agent.py,
# on mission subset 1 (= Moon_Map_01 preset 2, matching the phase0 DEM + sun). Data lands in
# LAC_SIM/output/Phase0CollectionAgent/<timestamp>/.
#
# Prerequisite: LAC_SIM/RunLeaderboard.sh must honor pre-set env vars, i.e. use
#   export TEAM_AGENT="${TEAM_AGENT:-$TEAM_CODE_ROOT/agents/nav_agent.py}"
#   export MISSIONS_SUBSET="${MISSIONS_SUBSET:-1}"
# (one-time edit; the stock script hard-codes these). Then just: ./launch_phase0.sh
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export TEAM_AGENT="$SCRIPT_DIR/agents/phase0_collection_agent.py"
export MISSIONS_SUBSET=1   # mission id 1 = Moon_Map_01 preset 2

exec "$SCRIPT_DIR/launch_headless.sh"
