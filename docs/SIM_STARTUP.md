# LAC Simulator Startup Guide

End-to-end instructions for getting the JHU APL Lunar Autonomy Challenge simulator running. Two paths are documented:

| Path | When to use |
|---|---|
| **Section 0 — Native Linux / SSH Server (PRIMARY)** | Any machine running native Linux + NVIDIA driver. This is the recommended and actively maintained path. |
| **Sections 1–6 — WSL2 / Ubuntu 24.04 (LEGACY)** | Windows 11 + WSL2 + Ubuntu 24.04. **Blocked** on a `dzn` D3D12 fence-timeout bug in UE4 4.26 (see Section 5). Kept for reference only. |

---

## 0. Native Linux / SSH Server Setup (PRIMARY PATH)

Use this section when setting up the simulator on a native Linux machine — including an SSH server accessed remotely. UE4 4.26 on native Linux with a real NVIDIA driver works out-of-the-box; all of the WSL2/dzn complexities disappear.

This section also serves as the **server coding agent playbook** — a server agent should be able to reproduce a complete running environment from these steps alone. Cross-reference with the more detailed 15-step playbook in `docs/PROJECT_TURNOVER.md` Section 9 if additional context is needed.

---

### 0.1 Verify GPU hardware

```bash
nvidia-smi
# Expect: GPU listed, Driver Version >= 525, CUDA >= 12.x, memory >= 8 GB
```

If `nvidia-smi` is not found, install the NVIDIA driver first:
```bash
# Ubuntu 22.04 / 24.04:
sudo apt install -y nvidia-driver-535     # or latest stable
sudo reboot
```

---

### 0.2 System dependencies

```bash
sudo apt update
sudo apt install -y \
    xvfb \
    vulkan-tools mesa-utils libvulkan1 mesa-vulkan-drivers \
    cmake build-essential pkg-config \
    git curl wget
```

What each piece does:
- `xvfb` → headless X11 display server — **required for running UE4 without a physical monitor** on SSH servers.
- `vulkan-tools` → `vulkaninfo`, `vkcube` for diagnostics.
- `libvulkan1` + `mesa-vulkan-drivers` → Vulkan loader + Mesa ICDs (native NVIDIA ICD comes from the NVIDIA driver package, not Mesa).
- `cmake`/`build-essential` → needed for pip wheels that build from source (apriltag).

---

### 0.3 Conda + Python environment

```bash
# Install Miniconda if not present
curl -sLO https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p "$HOME/miniconda3"
"$HOME/miniconda3/bin/conda" init bash
exec bash

# Create the lac env
conda create -y -n lac python=3.10
conda activate lac

# CRITICAL: prevent ~/.local/lib/python3.10/site-packages from shadowing the conda env.
# Skip this and pinned versions are silently ignored; wrong packages imported without warning.
conda env config vars set PYTHONNOUSERSITE=1 -n lac
conda deactivate && conda activate lac
```

---

### 0.4 Clone the repo

```bash
# On vader.stanford.edu the repo lives at:
cd ~/Documents
git clone <YOUR_FORK_URL> Lunar_Perception_Aware_Planning
cd Lunar_Perception_Aware_Planning
```

> **vader.stanford.edu note**: Large gitignored assets (`LAC_SIM/`, `data/Example_Implementations/`, `models/`) must live on the `/data/santiago/` partition. Create directories there and symlink them into the repo after cloning:
> ```bash
> mkdir -p /data/santiago/Lunar_Perception_Aware_Planning/{LAC_SIM,Example_Implementations,models}
> ln -s /data/santiago/Lunar_Perception_Aware_Planning/LAC_SIM ~/Documents/Lunar_Perception_Aware_Planning/LAC_SIM
> ln -s /data/santiago/Lunar_Perception_Aware_Planning/Example_Implementations \
>        ~/Documents/Lunar_Perception_Aware_Planning/data/Example_Implementations
> ln -s /data/santiago/Lunar_Perception_Aware_Planning/models ~/Documents/Lunar_Perception_Aware_Planning/models
> ```

---

### 0.5 Python dependencies

```bash
conda activate lac
cd ~/Documents/Lunar_Perception_Aware_Planning

# apriltag 0.0.16's CMakeLists.txt uses syntax that CMake 4.x rejects. Pin to 3.x first.
pip install "cmake<4"

# apriltag must be installed with --no-build-isolation BEFORE requirements.txt,
# otherwise pip's build isolation hides our cmake and the build fails.
pip install --no-build-isolation apriltag==0.0.16

# PyTorch with CUDA 12.1 — install BEFORE requirements.txt so it doesn't
# get overridden by the default CPU wheel from PyPI.
pip install torch==2.4.1 torchvision==0.19.1 \
    --index-url https://download.pytorch.org/whl/cu121

# Main deps (gtsam, symforce, transformers, ...)
pip install -r requirements.txt

# Make the lac package importable
pip install -e .

# Four undocumented deps used by the agent path but missing from requirements.txt:
pip install imageio munch segmentation-models-pytorch opt_einsum

# Two more undocumented deps needed by mission_weather.py (sun position computation):
# NOT in requirements.txt — verified missing on vader.stanford.edu.
pip install astropy==5.2.2 lunarsky==0.2.1
```

---

### 0.6 LightGlue

```bash
mkdir -p ~/opt && cd ~/opt
git clone https://github.com/cvg/LightGlue.git
cd LightGlue
pip install -e .
```

---

### 0.7 Transfer LAC_SIM/ to the server

`LAC_SIM/` is gitignored (~12 GB). Transfer it from your development machine.

**Option A — rsync** (works when direct SSH is available):
```bash
# From your source machine (replace server-ip and paths):
rsync -avz --progress \
    ~/Documents/Lunar_Perception_Aware_Planning/LAC_SIM/ \
    santiagothorup@vader.stanford.edu:~/Documents/Lunar_Perception_Aware_Planning/LAC_SIM/
```

**Option B — Google Drive + rclone** (required when rsync/scp is blocked by VPN):

> **vader.stanford.edu note**: Direct SCP/rsync is blocked by the Stanford VPN. The method that worked was: zip the folder → upload to Google Drive from Windows → download via rclone on the server.

```bash
# 1. Zip and upload to Google Drive from Windows/WSL2 (do this on your laptop):
#    - Compress LAC_SIM/ to LunarAutonomyChallenge.zip
#    - Upload to Google Drive (browser or rclone)

# 2. On the server — install rclone and configure Google Drive remote:
curl https://rclone.org/install.sh | sudo bash
rclone config   # create remote named "gdrive" pointing to your Google Drive

# 3. Download and extract:
cd /data/santiago/Lunar_Perception_Aware_Planning
rclone copy gdrive:LunarAutonomyChallenge.zip . --progress
unzip LunarAutonomyChallenge.zip

# Fix nesting if the zip extracted with an extra subdirectory:
ls  # if you see LunarAutonomyChallenge/ directory:
mv LunarAutonomyChallenge/* LAC_SIM/
rmdir LunarAutonomyChallenge
```

> **Storage note**: On `vader.stanford.edu`, large gitignored assets should live under `/data/santiago/Lunar_Perception_Aware_Planning/` (the data partition). Create symlinks from the repo so code paths remain unchanged:
> ```bash
> cd ~/Documents/Lunar_Perception_Aware_Planning
> ln -s /data/santiago/Lunar_Perception_Aware_Planning/LAC_SIM LAC_SIM
> ```

After transfer, verify the structure:
```
LAC_SIM/
├── RunLunarSimulator.sh, RunLeaderboard.sh
├── requirements.txt
├── agents/
├── Leaderboard/
├── LunarSimulator/
│   ├── LAC.sh
│   └── LAC/Binaries/Linux/LAC-Linux-Shipping
├── results/
└── wheelhouse/
    └── carla-0.9.15-cp310-cp310-manylinux_2_27_x86_64.whl
```

---

### 0.8 Install Carla + sim deps

```bash
conda activate lac
SIM="$HOME/Stanford/AA278/Lunar_Perception_Aware_Planning/LAC_SIM"
pip install "$SIM/wheelhouse/carla-0.9.15-cp310-cp310-manylinux_2_27_x86_64.whl"
pip install dictor==0.1.12 tabulate==0.9.0 pygame==2.5.2
python -c "import carla, dictor, tabulate, pygame; print('carla 0.9.15 OK')"
```

---

### 0.9 Transfer model weights + DEM data

```bash
# From your source machine (if rsync/scp is available):
rsync -avz models/unet_v2.pth \
    santiagothorup@vader.stanford.edu:~/Documents/Lunar_Perception_Aware_Planning/models/

rsync -avz data/DEMs/ \
    santiagothorup@vader.stanford.edu:~/Documents/Lunar_Perception_Aware_Planning/data/DEMs/
```

> **If rsync/scp is blocked by VPN** (as on vader with Stanford VPN): use Google Drive + rclone, same method as Section 0.7 Option B. Upload models/ and data/DEMs/ as separate zips; download and extract on the server.

---

### 0.10 Configure `RunLunarSimulator.sh` for native Linux

The file in the transferred `LAC_SIM/` may contain WSL2-specific env vars. Replace its contents with:

```bash
#!/bin/bash
# Native Linux / SSH server launch wrapper.
# No WSL2 or dzn environment variables needed — NVIDIA driver handles everything directly.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SIMULATOR_ROOT="$SCRIPT_DIR/LunarSimulator"
bash "$SIMULATOR_ROOT/LAC.sh" "$@"
```

The file may be root-owned (set during initial sim setup). Edit with:
```bash
# Send to user to run if needed:
sudo nano LAC_SIM/RunLunarSimulator.sh
# or:
sudo tee LAC_SIM/RunLunarSimulator.sh << 'EOF'
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SIMULATOR_ROOT="$SCRIPT_DIR/LunarSimulator"
bash "$SIMULATOR_ROOT/LAC.sh" "$@"
EOF
```

---

### 0.11 Configure `RunLeaderboard.sh`

Verify/patch `RunLeaderboard.sh` to match the server. On `vader.stanford.edu` the current working version is:

```bash
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# CRITICAL: prepend conda env bin so 'rerun' binary is found.
# Without this, nav_agent.py fails with "Failed to find Rerun Viewer executable in PATH".
export PATH="/home/santiagothorup/miniconda3/envs/lac/bin:$PATH"

export LAC_BASE_PATH="$SCRIPT_DIR"
export LEADERBOARD_ROOT="$SCRIPT_DIR/Leaderboard"
export TEAM_CODE_ROOT="/home/santiagothorup/Documents/Lunar_Perception_Aware_Planning"

export PYTHONPATH="$LEADERBOARD_ROOT:$TEAM_CODE_ROOT:$PYTHONPATH"

export TEAM_AGENT="$TEAM_CODE_ROOT/agents/nav_agent.py"
export TEAM_CONFIG="$TEAM_CODE_ROOT/configs/config.json"

export MISSIONS="$LEADERBOARD_ROOT/data/missions_training.xml"
export MISSIONS_SUBSET="1"   # index 1 = preset 2 (Moon_Map_01_2_rep0.dat)
# ...rest of file unchanged...
```

Key things to patch when adapting to a new server:
1. `PATH` prepend — conda env bin must contain `rerun`.
2. `TEAM_CODE_ROOT` — absolute path to the repo root.
3. `MISSIONS_SUBSET` — **index** into the missions XML, not the preset number. Index 0 = preset 1, index 1 = preset 2.

**DEM naming**: `nav_agent.py` reads `results/Moon_Map_01_{MISSIONS_SUBSET}_rep0.dat` at startup (using the index, not the preset number). You must copy the DEM file to match:
```bash
# For MISSIONS_SUBSET=1 (preset 2):
cp data/DEMs/Moon_Map_01_2_rep0.dat LAC_SIM/results/Moon_Map_01_1_rep0.dat
# For MISSIONS_SUBSET=0 (preset 1):
cp data/DEMs/Moon_Map_01_0_rep0.dat LAC_SIM/results/Moon_Map_01_0_rep0.dat
```
Without this, nav_agent.py crashes immediately with `FileNotFoundError`.

---

### 0.12 Make scripts executable

```bash
SIM="$HOME/Stanford/AA278/Lunar_Perception_Aware_Planning/LAC_SIM"
chmod +x "$SIM/RunLunarSimulator.sh" "$SIM/RunLeaderboard.sh" \
         "$SIM/LunarSimulator/LAC.sh" \
         "$SIM/LunarSimulator/LAC/Binaries/Linux/LAC-Linux-Shipping"
mkdir -p "$SIM/output"
```

---

### 0.13 Launch headless with Xvfb

UE4 requires a display even when rendering to off-screen textures. On a headless SSH server, create a virtual display with Xvfb.

#### Recommended: use `launch_headless.sh`

`launch_headless.sh` at the repo root handles everything — Xvfb, VK_ICD selection, sim + agent launch with `nohup setsid` for persistence after SSH shell exit, and port-2000 polling to sequence the launches correctly:

```bash
cd ~/Documents/Lunar_Perception_Aware_Planning
bash launch_headless.sh
tail -f logs/agent.log   # watch for "Step: 1, 2, ..."
```

#### Manual launch (two terminals)

If you need more control:

**Terminal A — start virtual display + simulator:**
```bash
# Remove stale Xvfb lock if it exists from a previous session:
rm -f /tmp/.X99-lock

Xvfb :99 -screen 0 1920x1080x24 &
export DISPLAY=:99

# On multi-GPU servers (e.g., vader.stanford.edu), force NVIDIA ICD explicitly.
# Without this, Mesa may be selected as the Vulkan provider → software rendering.
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json

conda activate lac
cd ~/Documents/Lunar_Perception_Aware_Planning/LAC_SIM
./RunLunarSimulator.sh
```

Wait ~60 seconds for UE4 to fully load. Watch for the Carla server log output indicating the world loaded (the process will not terminate — it waits for a client connection).

**Terminal B — run the agent:**
```bash
export DISPLAY=:99
conda activate lac
cd ~/Documents/Lunar_Perception_Aware_Planning/LAC_SIM
./RunLeaderboard.sh
```

Watch for `Step: 1`, `Step: 2`, ... — the agent is running.

#### Non-interactive environments (important for SSH sessions)

Standard `&` backgrounding does **not** survive shell exit — the SSH shell kills all child processes on logout. Use `nohup setsid` to make processes persist:

```bash
# Start sim in background, surviving shell exit:
nohup setsid bash -c "
  export DISPLAY=:99
  export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
  cd ~/Documents/Lunar_Perception_Aware_Planning/LAC_SIM
  ./RunLunarSimulator.sh
" > logs/sim.log 2>&1 &

# Wait for port 2000 to open (sim ready):
while ! ss -tlnp | grep -q ':2000'; do sleep 3; done

# Start agent in background:
nohup setsid bash -c "
  export PATH='/home/santiagothorup/miniconda3/envs/lac/bin:$PATH'
  export DISPLAY=:99
  cd ~/Documents/Lunar_Perception_Aware_Planning/LAC_SIM
  ./RunLeaderboard.sh
" > logs/agent.log 2>&1 &
```

> **Headless image quality note:** `SceneCaptureComponent2D` in Carla renders to GPU textures independently of the display window. Sensor images (front camera, stereo pair, depth) are produced at full fidelity even in headless mode. Xvfb is only needed so UE4 can create its internal window object — it has no effect on sensor output.

---

### 0.14 Optional: VNC for visual monitoring

If you want to observe the UE4 window remotely:

```bash
# On server:
sudo apt install -y tigervnc-standalone-server
vncserver :1 -geometry 1920x1080 -depth 24

# Start sim on :1 instead of :99:
export DISPLAY=:1
./RunLunarSimulator.sh

# From laptop (replace server-ip):
# Connect VNC client to server-ip:5901
```

---

### 0.15 Native Linux sanity checks

```bash
conda activate lac
export DISPLAY=:99   # if headless

REPO="$HOME/Documents/Lunar_Perception_Aware_Planning"
echo "Python   : $(python --version)"
echo "Torch    : $(python -c 'import torch; print(torch.__version__, "cuda=", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")')"
echo "Carla    : $(python -c 'import carla; print("0.9.15 OK")')"
echo "astropy  : $(python -c 'import astropy; print(astropy.__version__)')"
echo "lunarsky : $(python -c 'import lunarsky; print("OK")')"
echo "Vulkan   : $(vulkaninfo --summary 2>/dev/null | grep deviceName | head -2)"
echo "GPU      : $(nvidia-smi --query-gpu=name,memory.free --format=csv,noheader)"
echo "LAC_SIM  : $(test -d $REPO/LAC_SIM && echo present || echo MISSING)"
echo "DEM      : $(test -f $REPO/data/DEMs/Moon_Map_01_2_rep0.dat && echo present || echo MISSING)"
echo "DEM(idx) : $(test -f $REPO/LAC_SIM/results/Moon_Map_01_1_rep0.dat && echo present || echo MISSING — run DEM naming fix)"
echo "UNet     : $(test -f $REPO/models/unet_v2.pth && echo present || echo MISSING)"
```

Expected:
- Python 3.10.x, Torch 2.4.1+cu121, `cuda= True`
- Carla 0.9.15 OK, astropy 5.2.2, lunarsky OK
- Vulkan device name contains "NVIDIA" (native NVIDIA ICD, not Mesa/dzn)
- All `present` — the `DEM(idx)` check verifies the DEM naming fix was applied

---

### 0.16 Collect Phase 0 transect data (CURRENT TASK)

Drives `agents/phase0_collection_agent.py` over the full-map `phase0_transect` raster on preset 2 and logs FrontLeft+FrontRight + ground-truth poses in the exact format `scripts/phase0_validation.py` reads. This is the gating Phase 0 re-run (the earlier preset-2 run was WEAK because the rover only covered a 7×15 m box). Decision logic and escalation are in `PROJECT_TURNOVER.md` §9.A; the mechanics:

**One-time — make `RunLeaderboard.sh` env-overridable** so `launch_phase0.sh` can swap in the collection agent (the stock script hard-codes these with bare `export`):
```bash
export TEAM_AGENT="${TEAM_AGENT:-$TEAM_CODE_ROOT/agents/nav_agent.py}"
export TEAM_CONFIG="${TEAM_CONFIG:-$TEAM_CODE_ROOT/configs/config.json}"
export MISSIONS_SUBSET="${MISSIONS_SUBSET:-1}"
```

**Collect** (~20 min sim time; serpentine raster, lander-avoiding, no obstacle-avoidance needed):
```bash
cd ~/Documents/Lunar_Perception_Aware_Planning
git pull
pkill -f "LAC-Linux-Shipping" 2>/dev/null; rm -f /tmp/.X99-lock
./launch_phase0.sh
tail -f LAC_SIM/logs/agent.log      # watch "Step:" / "Waypoint i/N", clean finalize, no early Out-of-power
```
`launch_phase0.sh` sets `TEAM_AGENT=agents/phase0_collection_agent.py` + `MISSIONS_SUBSET=1` and execs `launch_headless.sh`. Output → `LAC_SIM/output/Phase0CollectionAgent/<timestamp>/` (`data_log.json` + `FrontLeft/` + `FrontRight/`; the two image folders should have equal counts).

**Re-validate on the collected data:**
```bash
RUN=$(ls -dt LAC_SIM/output/Phase0CollectionAgent/*/ | head -1)
PHASE0_DATA_DIR="$PWD/$RUN" PHASE0_OUT_DIR="output/phase0_transect" \
  PYTHONNOUSERSITE=1 python scripts/phase0_validation.py
```
Read `output/phase0_transect/response_comparison.png` + the stdout verdict. Matched |H4b Pearson_iid| ≥ 0.2 (ideally `n_matched_temporal`) green-lights the planner build; otherwise escalate to a richer preset (the agent is preset-agnostic — change `MISSIONS_SUBSET`, obtain that preset's DEM from `results/`, recompute the sun azimuth).

**Troubleshooting:** every transect leg is < 20 m, so `WAYPOINT_TIMEOUT` (2000 steps = 100 s) force-advances a stuck waypoint before the 300 s "blocked" termination — the mission won't deadlock. If the run is mistakenly in `testing` mode (30 s mission cap) the data will be near-empty; the default `RunLeaderboard.sh` (empty `--qualifier`, no `--testing`) gives the 24 h budget, so don't add those flags.

---

## 1. [LEGACY — WSL2] Host setup (Ubuntu 24.04)

> **⚠ WSL2 path status: BLOCKED (2026-05-26).**
> Even after completing all steps in this section, UE4 4.26 hangs indefinitely during PSO initialization due to a `dzn` D3D12 fence-timeout bug. See the troubleshooting entry at the bottom of Section 5 ("`./RunLunarSimulator.sh` hangs after dzn/Vulkan init"). **Use Section 0 (native Linux) instead.**

The steps below are preserved for reference. They were verified working up to and including Vulkan ICD setup (`vkcube` rendering at >30 fps). The simulator itself cannot proceed past UE4 startup on WSL2.

Authored against:

| Component | Verified value |
|---|---|
| Host OS | Windows 11 |
| WSL | WSL2 with WSLg enabled |
| Distro | Ubuntu 24.04 LTS (noble) |
| GPU | NVIDIA RTX 4060 Laptop GPU (driver 581.95 on Windows; CUDA 13.0) |
| iGPU | Intel Iris Xe (visible to WSLg as default D3D12 adapter) |

### 1.1 Conda

```bash
cd /tmp
curl -sLO https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p "$HOME/miniconda3"
"$HOME/miniconda3/bin/conda" init bash
exec bash

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

conda create -y -n lac python=3.10
conda activate lac

conda env config vars set PYTHONNOUSERSITE=1 -n lac
conda deactivate && conda activate lac
```

### 1.2 Graphics + GPU passthrough apt packages

```bash
sudo apt update
sudo apt install -y \
    vulkan-tools mesa-utils libvulkan1 mesa-vulkan-drivers \
    xvfb cmake build-essential pkg-config
```

### 1.3 Vulkan path — kisak-mesa PPA gives you dzn on noble

**Verified working on 2026-05-26: Ubuntu 24.04's stock `mesa-vulkan-drivers 25.2.8` does NOT include `dzn_icd.x86_64.json`.** Without dzn, `vulkaninfo` only shows `llvmpipe` and UE4 will time out.

**Fix: add kisak-mesa PPA (noble / 24.04 supported), which ships Mesa 26.x with dzn.**

```bash
sudo add-apt-repository -y ppa:kisak/kisak-mesa
sudo apt update
sudo apt full-upgrade -y
sudo apt install -y mesa-vulkan-drivers
```

After upgrade, verify:
```bash
vulkaninfo --summary | grep -B 1 -A 20 "Devices:"
```

Expected:
```
GPU0:  Microsoft Direct3D12 (NVIDIA GeForce RTX 4060 Laptop GPU)
       driverID = DRIVER_ID_MESA_DOZEN  •  Mesa 26.1.x
GPU1:  Microsoft Direct3D12 (Intel Iris Xe Graphics)
GPU2:  llvmpipe   (software fallback)
```

Confirm the GPU actually renders:
```bash
vkcube      # spinning textured cube; close window to exit
```

> **WARNING:** Even with `vkcube` rendering at >30 fps via dzn, UE4 4.26 will hang during PSO initialization. This is a fundamental `dzn` D3D12 translation bug — see Section 5 troubleshooting entry. `vkcube` success does **not** mean UE4 will work.

#### Troubleshooting

- **`add-apt-repository` says PPA unavailable**: PPA is `ppa:kisak/kisak-mesa` (lowercase, hyphens). Noble is actively supported.
- **`vulkaninfo` still only shows llvmpipe after upgrade**: verify `/usr/share/vulkan/icd.d/dzn_icd.x86_64.json` exists. If not, `sudo apt install --reinstall mesa-vulkan-drivers`.
- **`vkcube` crashes**: `wsl --shutdown` from PowerShell and try again.

### 1.4 OpenGL through WSLg (verify only)

```bash
MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA glxinfo -B | grep -E "renderer|Vendor|Video memory|Accelerated"
# Expect: "D3D12 (NVIDIA GeForce RTX 4060 Laptop GPU)", "Accelerated: yes"
```

---

## 2. [LEGACY — WSL2] Project code setup

### 2.1 Clone the repo

```bash
mkdir -p ~/Stanford/AA278 && cd ~/Stanford/AA278
git clone <FORK_URL> Lunar_Perception_Aware_Planning
cd Lunar_Perception_Aware_Planning
```

### 2.2 Python dependencies for the lac env

```bash
conda activate lac
cd ~/Stanford/AA278/Lunar_Perception_Aware_Planning

pip install "cmake<4"
pip install --no-build-isolation apriltag==0.0.16
pip install -r requirements.txt
pip install -e .
pip install imageio munch segmentation-models-pytorch opt_einsum
```

**Two gotchas:**
1. Pip transactions are atomic — one failing build aborts everything. Verify with `python -c "import torch; print(torch.__version__)"` afterwards.
2. `PYTHONNOUSERSITE=1` is non-negotiable. See Section 1.1.

### 2.3 LightGlue

```bash
mkdir -p ~/opt && cd ~/opt
git clone https://github.com/cvg/LightGlue.git
cd LightGlue
pip install -e .
```

The JHU install guide mentions patching `lightglue.py:24` — no longer needed; upstream already uses the new form (verified 2026-05).

### 2.4 LangSAM (deprecated — skip)

Not in `requirements.txt`. Agent code does not import it.

### 2.5 Pretrained UNet segmentation weights

Required by `lac/perception/segmentation.py`. Download from the JHU portal's "Model weights" link.

```bash
mkdir -p models
# Drop unet_v2.pth here (~100 MB)
ls -la models/unet_v2.pth
```

### 2.6 Verify agent-path imports

```bash
conda activate lac
SIM="$HOME/Stanford/AA278/Lunar_Perception_Aware_Planning/LAC_SIM"
PYTHONPATH="$SIM/Leaderboard:$HOME/Stanford/AA278/Lunar_Perception_Aware_Planning:$PYTHONPATH" \
python - <<'PY'
import warnings; warnings.filterwarnings('ignore')
import numpy as np, torch
from lac.slam.frontend import Frontend
from lac.slam.backend import Backend
from lac.planning.arc_planner import ArcPlanner
from lac.planning.waypoint_planner import WaypointPlanner

pose = np.eye(4); pose[:2, 3] = [1.0, 1.0]
ap = ArcPlanner()
wp = WaypointPlanner(pose, trajectory_type='five_loops')
print('torch        :', torch.__version__, 'cuda=', torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')
print('ArcPlanner   :', len(ap.np_candidate_arcs), 'arcs')
print('WaypointPlnr :', len(wp.waypoints), 'waypoints (first=', wp.waypoints[0].tolist(), ')')
print('PASS: agent-path imports green.')
PY
```

---

## 3. [LEGACY — WSL2] LAC simulator (JHU APL zip)

### 3.1 Acquire the zip

Simulator zip from JHU APL, ~12 GB unzipped.

**Don't unzip via Windows Explorer** — 260-char path limit silently skips some assets.

```bash
cd ~/Stanford/AA278/Lunar_Perception_Aware_Planning
# Extract via WSL unzip, rename to LAC_SIM/ (single level, not double-nested):
unzip /mnt/c/Users/<you>/Downloads/LunarAutonomyChallenge.zip
mv LunarAutonomyChallenge LAC_SIM
```

### 3.2 Install Carla + sim deps

Don't use `--force-reinstall` on the sim's `requirements.txt` (it downgrades numpy/matplotlib). Install only missing pieces:

```bash
SIM="$HOME/Stanford/AA278/Lunar_Perception_Aware_Planning/LAC_SIM"
pip install "$SIM/wheelhouse/carla-0.9.15-cp310-cp310-manylinux_2_27_x86_64.whl"
pip install dictor==0.1.12 tabulate==0.9.0 pygame==2.5.2
python -c "import carla, dictor, tabulate, pygame; print('carla 0.9.15 OK')"
```

### 3.3 Make scripts executable + create output/

```bash
SIM="$HOME/Stanford/AA278/Lunar_Perception_Aware_Planning/LAC_SIM"
chmod +x "$SIM/RunLunarSimulator.sh" "$SIM/RunLeaderboard.sh" \
         "$SIM/LunarSimulator/LAC.sh" \
         "$SIM/LunarSimulator/LAC/Binaries/Linux/LAC-Linux-Shipping"
mkdir -p "$SIM/output"
```

### 3.4 `RunLunarSimulator.sh` — WSL2 version (LEGACY)

> **This configuration is for WSL2 only and does NOT result in a working simulator (dzn fence timeout).** For native Linux, see Section 0.10.

Current file contents (WSL2 config — do not use on native Linux):

```bash
#!/bin/bash
export MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
export VK_LAYER_PATH="/tmp"
export VK_INSTANCE_LAYERS="VK_LAYER_lac_compat"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SIMULATOR_ROOT="$SCRIPT_DIR/LunarSimulator"
bash "$SIMULATOR_ROOT/LAC.sh" "$@"
```

The file is root-owned. If editing is needed: `sudo nano LAC_SIM/RunLunarSimulator.sh`.

### 3.5 `RunLeaderboard.sh` — already patched

`TEAM_CODE_ROOT` already points at this repo. The patched top section:

```bash
export LAC_BASE_PATH="$SCRIPT_DIR"
export LEADERBOARD_ROOT="$SCRIPT_DIR/Leaderboard"
export TEAM_CODE_ROOT="$HOME/Stanford/AA278/Lunar_Perception_Aware_Planning"
export PYTHONPATH="$LEADERBOARD_ROOT:$TEAM_CODE_ROOT:$PYTHONPATH"
export TEAM_AGENT="$TEAM_CODE_ROOT/agents/nav_agent.py"
export TEAM_CONFIG="$TEAM_CODE_ROOT/configs/config.json"
```

`MISSIONS_SUBSET=0` = preset 1; `MISSIONS_SUBSET=1` = preset 2 (our DEM: `data/DEMs/Moon_Map_01_2_rep0.dat`).

### 3.6 Pre-commit hooks (optional)

```bash
pip install pre-commit
pre-commit install
```

---

## 4. [LEGACY — WSL2] Running the simulator

> **This section assumes WSL2 + dzn. The simulator hangs at startup on WSL2. For native Linux headless launch, see Section 0.13.**

Two terminals, both inside `LAC_SIM/`.

### 4.1 Terminal A — base simulator

```bash
conda activate lac
./RunLunarSimulator.sh
```

**Watch for:** UE4 window opens, lunar terrain renders at >10 fps, GPU panel shows NVIDIA RTX 4060. Close with the X button — not Ctrl-C.

### 4.2 Terminal B — agent / leaderboard

```bash
conda activate lac
./RunLeaderboard.sh
```

**Watch for:** `> Loading the world`, then `Step: 1`, `Step: 2`, ...

Ground-truth heightmap written to `LAC_SIM/results/Moon_Map_01_<PRESET>_rep0.dat` after each run. Copy to `data/DEMs/`.

---

## 5. Common failures and fixes

| Symptom | Cause | Fix |
|---|---|---|
| Sim window opens, immediately crashes with "OpenGL no longer supported" + GameThread timeout | UE4 4.26 tried Vulkan, got llvmpipe | Section 1.3 — fix Vulkan ICDs |
| `vulkaninfo` shows only llvmpipe even on 24.04 | `dzn_icd.x86_64.json` not in `/usr/share/vulkan/icd.d/` | See Section 1.3 troubleshooting |
| Sim runs but slow (<10 fps) | WSLg routed to Intel iGPU | `export MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA` (already in `RunLunarSimulator.sh`) |
| `torch.cuda.is_available()` → False | Wrong torch wheel or `/usr/lib/wsl/lib/libcuda.so` not visible | `pip install --force-reinstall torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121`. If needed: `export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH`. |
| `pip install -r requirements.txt` fails on apriltag | CMake 4.x rejects apriltag's old CMakeLists | `pip install "cmake<4"` then `pip install --no-build-isolation apriltag==0.0.16` BEFORE requirements install |
| `python -c "import torch"` returns wrong version | User-site shadowing | Section 1.1 — `PYTHONNOUSERSITE=1` on the lac env |
| RunLeaderboard.sh "ModuleNotFoundError: leaderboard" | PYTHONPATH not set | Always run via `./RunLeaderboard.sh`, not directly |
| Ctrl-C leaves Carla orphaned | UE4 quirk | Close with X. Clean up with `pkill -f 'LAC-Linux-Shipping'`. |
| Sim window doesn't open at all (WSL2) | WSLg not running / no DISPLAY | `echo $DISPLAY` should be `:0`; if empty, `wsl --shutdown` from PowerShell. |
| UE4 hangs — no window, no crash; process stuck in `poll(/dev/dxg)` | **[WSL2 ONLY] `dzn` D3D12 fence timeout in PSO initialization** | **This is a fundamental bug in Mesa `dzn` with UE4 4.26. Not fixable via Vulkan layer, Mesa version, or any env var.** Root cause: UE4 submits PSO (pipeline state object) compilation work via D3D12; `dzn` mistranslates the D3D12 synchronization primitives; the GPU fence submitted to `/dev/dxg` never signals. Process hangs indefinitely. **Resolution: use native Linux (Section 0).** |
| Xvfb fails with "cannot open display :99" | Xvfb not installed or not running | `sudo apt install xvfb` then `Xvfb :99 -screen 0 1920x1080x24 &` before launching sim |
| Xvfb fails with "Server is already active for display 99" | Stale lock file from previous session | `rm -f /tmp/.X99-lock` then restart Xvfb |
| `./RunLunarSimulator.sh` exits immediately on SSH server | No display set | `export DISPLAY=:99` before running (or launch Xvfb first — see Section 0.13) |
| Native Linux: `vulkaninfo` shows Mesa software renderer only | NVIDIA driver not installed or not active | `nvidia-smi` to verify; re-install NVIDIA driver if missing |
| On multi-GPU server: Vulkan uses Mesa/software instead of NVIDIA | Multiple Vulkan ICDs; Mesa selected first | `export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json` before launching |
| `ModuleNotFoundError: No module named 'astropy'` | `astropy`/`lunarsky` not in requirements.txt | `pip install astropy==5.2.2 lunarsky==0.2.1` |
| `RuntimeError: Failed to find Rerun Viewer executable in PATH` | `rerun` binary in conda env not on PATH | Prepend `export PATH="/home/santiagothorup/miniconda3/envs/lac/bin:$PATH"` in `RunLeaderboard.sh` |
| `FileNotFoundError: results/Moon_Map_01_1_rep0.dat` | nav_agent.py reads DEM by MISSIONS_SUBSET **index**, not preset number | `cp data/DEMs/Moon_Map_01_2_rep0.dat LAC_SIM/results/Moon_Map_01_1_rep0.dat` (index 1 = preset 2) |
| Background processes die when SSH shell exits | Standard `&` backgrounding killed on shell exit | Use `nohup setsid bash -c "..." > logfile 2>&1 &` — see Section 0.13 non-interactive pattern |
| rsync/scp hangs silently or VS Code drag-and-drop fails | VPN blocks direct SSH data transfer | Use Google Drive + rclone as intermediary — see Section 0.7 Option B |
| Sim crash "bind: Address already in use" on retry | Previous sim instance still bound to port 2000 | `pkill -f "LAC-Linux-Shipping"` then wait 5 s before relaunch |

---

## 6. [LEGACY — WSL2] Quick sanity-check command sequence

> For native Linux sanity checks, use Section 0.15 instead.

```bash
conda activate lac
echo "Python    : $(python --version)"
echo "Torch     : $(python -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")')"
echo "Carla     : $(python -c 'import carla; print("0.9.15 importable")' 2>&1)"
echo "Vulkan    : $(vulkaninfo --summary 2>/dev/null | grep -A 1 'deviceName' | head -2)"
echo "OpenGL    : $(MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA glxinfo -B 2>/dev/null | grep 'OpenGL renderer' | head -1)"
echo "CUDA      : $(nvidia-smi --query-gpu=name,driver_version,memory.free --format=csv,noheader,nounits)"
echo "Sim dir   : $(test -d ~/Stanford/AA278/Lunar_Perception_Aware_Planning/LAC_SIM && echo 'present' || echo 'MISSING')"
echo "DEM       : $(test -f ~/Stanford/AA278/Lunar_Perception_Aware_Planning/data/DEMs/Moon_Map_01_2_rep0.dat && echo 'present' || echo 'MISSING')"
echo "UNet      : $(test -f ~/Stanford/AA278/Lunar_Perception_Aware_Planning/models/unet_v2.pth && echo 'present' || echo 'MISSING')"
```

---

## 7. End-state checklist

When all of the following are true, you're ready for Phase 0 re-run and planner build.

**Native Linux / SSH server (Section 0 path) — vader.stanford.edu status as of 2026-05-26:**
- [x] `nvidia-smi` shows GPU with ≥8 GB free VRAM ✅ (DONE on vader)
- [x] `python -c "import torch; print(torch.cuda.is_available())"` → `True` ✅
- [x] `python -c "import carla; print('OK')"` → `OK` ✅
- [x] `python -c "import astropy, lunarsky; print('OK')"` → `OK` ✅
- [x] `Xvfb :99 ...` starts without error; `export DISPLAY=:99` in the shell ✅
- [x] `VK_ICD_FILENAMES` set to NVIDIA ICD on multi-GPU server ✅
- [x] `./RunLunarSimulator.sh` starts and stays running (UE4 logs visible, no immediate crash) ✅
- [x] `./RunLeaderboard.sh` in a second terminal prints `Step: 1, 2, 3, ...` and the rover moves ✅
- [x] After a run, `results/Moon_Map_01_<SUBSET>_rep0.dat` appears in the sim folder ✅
- [x] DEM naming fix applied (`results/Moon_Map_01_1_rep0.dat` present) ✅
- [x] `python scripts/phase0_validation.py` runs end-to-end (matched-features v2 on preset 2 — WEAK 0.18) ✅

**Phase 0 transect collection (Section 0.16) — CURRENT TASK:**
- [ ] `RunLeaderboard.sh` made env-overridable (TEAM_AGENT/TEAM_CONFIG/MISSIONS_SUBSET)
- [ ] `./launch_phase0.sh` completes with a clean finalize; `LAC_SIM/output/Phase0CollectionAgent/<run>/` has data_log.json + FrontLeft/ + FrontRight/ (equal counts)
- [ ] `PHASE0_DATA_DIR=<run> PHASE0_OUT_DIR=output/phase0_transect python scripts/phase0_validation.py` → verdict read; |H4b Pearson_iid| ≥ 0.2 green-lights the planner

**WSL2 (legacy — do not expect the sim to run):**
- [ ] `vkcube` opens and renders at >30 fps
- [ ] `vulkaninfo --summary` shows NVIDIA / Direct3D12 device (not only llvmpipe)
- [ ] `python -c "import torch; print(torch.cuda.is_available())"` → `True`
- ~~`./RunLunarSimulator.sh` opens a window~~ — **BLOCKED** (dzn fence timeout, see Section 5)

The next planning/research step after this is documented in `docs/PROJECT_TURNOVER.md`.
