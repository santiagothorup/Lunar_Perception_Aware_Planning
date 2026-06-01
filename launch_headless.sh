#!/bin/bash
# launch_headless.sh — Start the LAC simulator headlessly and run an agent.
#
# Usage:
#   ./launch_headless.sh              # runs nav_agent.py with configs/config.json (preset 2)
#   MISSIONS_SUBSET=3 ./launch_headless.sh   # different preset
#
# Logs: LAC_SIM/logs/sim.log and LAC_SIM/logs/agent.log
# Results: LAC_SIM/results/

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIM="$SCRIPT_DIR/LAC_SIM"
LOGS="$SIM/logs"
mkdir -p "$LOGS"

# ── 1. Start virtual display ────────────────────────────────────────────────
DISPLAY_NUM=99
if ! pgrep -f "Xvfb :$DISPLAY_NUM" > /dev/null; then
    echo "[launch] Starting Xvfb :$DISPLAY_NUM"
    Xvfb :$DISPLAY_NUM -screen 0 1920x1080x24 &
    XVFB_PID=$!
    sleep 2
    echo "[launch] Xvfb started (PID $XVFB_PID)"
else
    echo "[launch] Xvfb :$DISPLAY_NUM already running"
fi
export DISPLAY=:$DISPLAY_NUM

# Pin UE4 to the NVIDIA GPU (prevents it picking Intel iGPU if both ICDs present)
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json

# Reduce CUDA OOM from fragmentation: the UE4 sim holds ~21 GB of the 24 GB GPU; the agent's
# LightGlue feature matching needs to allocate small buffers as the keyframe DB grows.
# expandable_segments lets PyTorch grow its cache without contiguous-block requirements, avoiding
# the OOM observed at ~step 3000 on the tour reruns. See
# https://pytorch.org/docs/stable/notes/cuda.html#environment-variables
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── 2. Start simulator ───────────────────────────────────────────────────────
echo "[launch] Starting LunarSimulator (logs → $LOGS/sim.log)"
cd "$SIM"
./RunLunarSimulator.sh > "$LOGS/sim.log" 2>&1 &
SIM_PID=$!
echo "[launch] Simulator started (PID $SIM_PID)"

# ── 3. Wait for simulator to be ready ───────────────────────────────────────
echo "[launch] Waiting for simulator to load (~90s)..."
for i in $(seq 1 90); do
    if grep -q "LogCarla: Applying settings" "$LOGS/sim.log" 2>/dev/null || \
       grep -q "LogGameMode:" "$LOGS/sim.log" 2>/dev/null; then
        echo "[launch] Simulator ready after ${i}s ✓"
        break
    fi
    sleep 1
    printf "."
done
echo ""

# ── 4. Run agent ─────────────────────────────────────────────────────────────
echo "[launch] Starting agent (logs → $LOGS/agent.log)"
./RunLeaderboard.sh 2>&1 | tee "$LOGS/agent.log"

echo "[launch] Mission complete. Results in $SIM/results/"
