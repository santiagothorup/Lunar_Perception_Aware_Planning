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
mkdir -p ~/Stanford/AA278 && cd ~/Stanford/AA278
git clone <YOUR_FORK_URL> Lunar_Perception_Aware_Planning
cd Lunar_Perception_Aware_Planning
```

---

### 0.5 Python dependencies

```bash
conda activate lac
cd ~/Stanford/AA278/Lunar_Perception_Aware_Planning

# apriltag 0.0.16's CMakeLists.txt uses syntax that CMake 4.x rejects. Pin to 3.x first.
pip install "cmake<4"

# apriltag must be installed with --no-build-isolation BEFORE requirements.txt,
# otherwise pip's build isolation hides our cmake and the build fails.
pip install --no-build-isolation apriltag==0.0.16

# Main deps (torch 2.4.1+cu121, gtsam, symforce, transformers, ...)
pip install -r requirements.txt

# Make the lac package importable
pip install -e .

# Four undocumented deps used by the agent path but missing from requirements.txt:
pip install imageio munch segmentation-models-pytorch opt_einsum
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

`LAC_SIM/` is gitignored (~12 GB). Transfer it from your development machine:

```bash
# From your source machine (replace server-ip and paths):
rsync -avz --progress \
    /home/sthorup/Stanford/AA278/Lunar_Perception_Aware_Planning/LAC_SIM/ \
    user@server-ip:~/Stanford/AA278/Lunar_Perception_Aware_Planning/LAC_SIM/
```

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
# From your source machine:
rsync -avz models/unet_v2.pth \
    user@server-ip:~/Stanford/AA278/Lunar_Perception_Aware_Planning/models/

rsync -avz data/DEMs/ \
    user@server-ip:~/Stanford/AA278/Lunar_Perception_Aware_Planning/data/DEMs/
```

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

Verify/patch `TEAM_CODE_ROOT` in `LAC_SIM/RunLeaderboard.sh` to point at the repo on this server:

```bash
# The patched top section should look like:
# export TEAM_CODE_ROOT="$HOME/Stanford/AA278/Lunar_Perception_Aware_Planning"
# export TEAM_AGENT="$TEAM_CODE_ROOT/agents/nav_agent.py"
# export TEAM_CONFIG="$TEAM_CODE_ROOT/configs/config.json"

# MISSIONS_SUBSET=0 → preset 1
# MISSIONS_SUBSET=1 → preset 2 (use this; our DEM is Moon_Map_01_2_rep0.dat)
```

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

UE4 requires a display even when rendering to off-screen textures. On a headless SSH server, create a virtual display with Xvfb:

**Terminal A — start virtual display + simulator:**
```bash
Xvfb :99 -screen 0 1920x1080x24 &
export DISPLAY=:99
conda activate lac
cd ~/Stanford/AA278/Lunar_Perception_Aware_Planning/LAC_SIM
./RunLunarSimulator.sh
```

Wait ~60 seconds for UE4 to fully load. Watch for the Carla server log output indicating the world loaded (the process will not terminate — it waits for a client connection).

**Terminal B — run the agent:**
```bash
export DISPLAY=:99
conda activate lac
cd ~/Stanford/AA278/Lunar_Perception_Aware_Planning/LAC_SIM
./RunLeaderboard.sh
```

Watch for `Step: 1`, `Step: 2`, ... — the agent is running.

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

echo "Python   : $(python --version)"
echo "Torch    : $(python -c 'import torch; print(torch.__version__, "cuda=", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")')"
echo "Carla    : $(python -c 'import carla; print("0.9.15 OK")')"
echo "Vulkan   : $(vulkaninfo --summary 2>/dev/null | grep deviceName | head -2)"
echo "GPU      : $(nvidia-smi --query-gpu=name,memory.free --format=csv,noheader)"
echo "LAC_SIM  : $(test -d ~/Stanford/AA278/Lunar_Perception_Aware_Planning/LAC_SIM && echo present || echo MISSING)"
echo "DEM      : $(test -f ~/Stanford/AA278/Lunar_Perception_Aware_Planning/data/DEMs/Moon_Map_01_2_rep0.dat && echo present || echo MISSING)"
echo "UNet     : $(test -f ~/Stanford/AA278/Lunar_Perception_Aware_Planning/models/unet_v2.pth && echo present || echo MISSING)"
```

Expected:
- Python 3.10.x, Torch 2.4.1+cu121, `cuda= True`
- Carla 0.9.15 OK
- Vulkan device name contains "NVIDIA" (native NVIDIA ICD, not Mesa/dzn)
- All `present`

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
| `./RunLunarSimulator.sh` exits immediately on SSH server | No display set | `export DISPLAY=:99` before running (or launch Xvfb first — see Section 0.13) |
| Native Linux: `vulkaninfo` shows Mesa software renderer only | NVIDIA driver not installed or not active | `nvidia-smi` to verify; re-install NVIDIA driver if missing |

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

**Native Linux / SSH server (Section 0 path):**
- [ ] `nvidia-smi` shows GPU with ≥8 GB free VRAM
- [ ] `python -c "import torch; print(torch.cuda.is_available())"` → `True`
- [ ] `python -c "import carla; print('OK')"` → `OK`
- [ ] `Xvfb :99 ...` starts without error; `export DISPLAY=:99` in the shell
- [ ] `./RunLunarSimulator.sh` starts and stays running (UE4 logs visible, no immediate crash)
- [ ] `./RunLeaderboard.sh` in a second terminal prints `Step: 1, 2, 3, ...` and the rover moves
- [ ] After a run, `results/Moon_Map_01_<PRESET>_rep0.dat` appears in the sim folder
- [ ] `python scripts/phase0_validation.py` runs end-to-end on the preset 2 DEM

**WSL2 (legacy — do not expect the sim to run):**
- [ ] `vkcube` opens and renders at >30 fps
- [ ] `vulkaninfo --summary` shows NVIDIA / Direct3D12 device (not only llvmpipe)
- [ ] `python -c "import torch; print(torch.cuda.is_available())"` → `True`
- ~~`./RunLunarSimulator.sh` opens a window~~ — **BLOCKED** (dzn fence timeout, see Section 5)

The next planning/research step after this is documented in `docs/PROJECT_TURNOVER.md`.
