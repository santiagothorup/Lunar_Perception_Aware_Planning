# LAC Simulator Startup Guide (WSL2 + Ubuntu 24.04 + NVIDIA dGPU)

End-to-end instructions for getting the JHU APL Lunar Autonomy Challenge simulator running on Windows 11 + WSL2 + Ubuntu 24.04 + an NVIDIA discrete GPU. Authored against this exact configuration:

| Component | Verified value |
|---|---|
| Host OS | Windows 11 |
| WSL | WSL2 with WSLg enabled (`/mnt/wslg` present, `DISPLAY=:0`, `WAYLAND_DISPLAY=wayland-0`) |
| Distro | Ubuntu 24.04 LTS (noble) |
| GPU | NVIDIA RTX 4060 Laptop GPU (driver 581.95 on Windows; reports CUDA 13.0 via `nvidia-smi`) |
| iGPU | Intel Iris Xe (visible to WSLg as the default D3D12 adapter — must be overridden) |
| Repo location | `~/Stanford/AA278/Lunar_Perception_Aware_Planning` (referred to as `$REPO`) |

## Why Ubuntu 24.04 (not 22.04)

We initially set this up on Ubuntu 22.04. The LAC simulator is built on **Unreal Engine 4.26**, which **dropped OpenGL support for desktop Linux** and is Vulkan-only. WSL2 Vulkan requires either:

1. **Mesa's `dzn` driver** (Microsoft Direct3D12 Vulkan implementation, translates Vulkan → D3D12 → NVIDIA Windows driver), OR
2. A WSL-aware **NVIDIA Linux Vulkan ICD** with a loader that can dlopen Windows DLLs through `/dev/dxg`.

Ubuntu 22.04's stock Mesa is too old to ship `dzn`, and the kisak-mesa PPA was discontinued for jammy in 2025. Ubuntu 24.04 ships Mesa 25.x which IS the right Mesa version — but the `dzn` ICD may still need targeted installation. **The Vulkan-on-WSL setup step (Section 1.3) is the highest-risk step in this guide.**

> **TL;DR rendering plan:** Get `dzn` (or any non-llvmpipe GPU Vulkan device) showing in `vulkaninfo --summary`. Then UE4 will render natively via Vulkan → D3D12 → NVIDIA, no flag tricks needed.

---

## 1. Host setup (one-time, on a fresh Ubuntu 24.04 WSL distro)

### 1.1 Conda

```bash
cd /tmp
curl -sLO https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p "$HOME/miniconda3"
"$HOME/miniconda3/bin/conda" init bash
exec bash    # reload PATH so `conda` is available

# Conda now requires explicit TOS acceptance for the default channels:
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Create the project env.
conda create -y -n lac python=3.10
conda activate lac

# CRITICAL: prevent ~/.local/lib/python3.10/site-packages from shadowing the conda env.
# Without this, `pip install` silently skips packages already in user-site, AND Python
# imports the user-site version regardless of what's installed in the env. The result is
# that pinned versions in requirements.txt are ignored without warning. This setting is
# persistent per-env. Bit me hard during the 22.04 setup.
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

What each piece does:
- `vulkan-tools` → `vulkaninfo`, `vkcube` for diagnostics.
- `mesa-utils` → `glxinfo` for OpenGL diagnostics.
- `libvulkan1` + `mesa-vulkan-drivers` → Vulkan loader + Mesa's GPU drivers (intel, radeon, nouveau, lvp/llvmpipe; hopefully dzn too).
- `xvfb` → headless X server, optional for SSH/CI.
- `cmake`/`build-essential` → needed when pip builds any C-extension wheels from source.

### 1.3 Vulkan path — kisak-mesa PPA gives you dzn on noble

**Verified working on 2026-05-26: Ubuntu 24.04's stock `mesa-vulkan-drivers 25.2.8` does NOT include the `dzn_icd.x86_64.json` ICD file** (the Microsoft Direct3D12 Vulkan driver that bridges to WSL's NVIDIA Windows driver). Without dzn, `vulkaninfo` only shows `llvmpipe` (CPU software rasterizer) and UE4 will time out.

**Fix: add the kisak-mesa PPA, which IS supported on noble (24.04) and ships Mesa 26.x with dzn bundled.**

```bash
sudo add-apt-repository -y ppa:kisak/kisak-mesa
sudo apt update
sudo apt full-upgrade -y
sudo apt install -y mesa-vulkan-drivers
```

After upgrading (Mesa goes 25.2 → 26.1+), `dzn_icd.x86_64.json` appears in `/usr/share/vulkan/icd.d/`. Verify:

```bash
vulkaninfo --summary | grep -B 1 -A 20 "Devices:"
```

**Expected:**
```
GPU0:  Microsoft Direct3D12 (NVIDIA GeForce RTX 4060 Laptop GPU)
       vendorID = 0x10de  •  deviceType = PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
       driverID = DRIVER_ID_MESA_DOZEN  •  Mesa 26.1.x
GPU1:  Microsoft Direct3D12 (Intel(R) Iris(R) Xe Graphics)
       deviceType = PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU
GPU2:  llvmpipe   (software fallback, harmless)
```

The `WARNING: dzn is not a conformant Vulkan implementation` line is informational, not an error.

Confirm the GPU actually renders:
```bash
vkcube      # spinning textured cube; close window to exit
```

If the cube spins at >30 fps on the NVIDIA device (check task manager on Windows side), you're done with Section 1. Proceed to Section 2.

#### Troubleshooting

- **`add-apt-repository` says PPA is unavailable**: re-check the URL — the PPA is `ppa:kisak/kisak-mesa` (lowercase, hyphens). Per the PPA's own page, noble (24.04) is the actively supported target as of 2026.
- **`vulkaninfo` still only shows llvmpipe after the upgrade**: verify `/usr/share/vulkan/icd.d/dzn_icd.x86_64.json` exists. If not, `apt show mesa-vulkan-drivers` should show version ≥ 26.x; if it's still 25.2, the upgrade didn't take — try `sudo apt install --reinstall mesa-vulkan-drivers` and recheck.
- **`vulkaninfo` shows dzn but `vkcube` crashes**: this would indicate a more fundamental WSL graphics issue. Restart WSL from PowerShell (`wsl --shutdown`) and try again.

### 1.4 OpenGL through WSLg (already works, just verify)

```bash
MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA glxinfo -B | grep -E "renderer|Vendor|Video memory|Accelerated"
# Expect: "D3D12 (NVIDIA GeForce RTX 4060 Laptop GPU)", "Accelerated: yes", ~24 GB
```

`MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA` is required if your laptop also has an Intel iGPU — WSLg otherwise picks the iGPU for D3D12 by default.

---

## 2. Project code setup

### 2.1 Clone the repo

```bash
mkdir -p ~/Stanford/AA278 && cd ~/Stanford/AA278
git clone <FORK_URL> Lunar_Perception_Aware_Planning
cd Lunar_Perception_Aware_Planning
```

(If migrating from another distro, you can also copy the repo via `cp -r /mnt/c/Users/<you>/...` or via `\\wsl$\Ubuntu-22.04\home\sthorup\Stanford\AA278\...` from Windows. Don't forget hidden files like `.git/`.)

### 2.2 Python dependencies for the lac env

```bash
conda activate lac
cd ~/Stanford/AA278/Lunar_Perception_Aware_Planning

# apriltag 0.0.16's CMakeLists.txt uses `cmake_minimum_required` syntax that CMake 4.x
# rejects. Pin to 3.x.
pip install "cmake<4"

# apriltag needs cmake on PATH at build time. Pip's default build isolation hides our
# env's cmake from the build subprocess. Install apriltag separately with
# --no-build-isolation BEFORE running `pip install -r requirements.txt`, otherwise
# apriltag fails and pip rolls back the entire `-r requirements.txt` transaction.
pip install --no-build-isolation apriltag==0.0.16

pip install -r requirements.txt   # torch==2.4.1+cu121, gtsam, symforce, transformers, ...
pip install -e .                  # makes the `lac` package importable

# Four undocumented deps the agent path uses but requirements.txt doesn't list:
#   imageio  -> lac/utils/plotting.py
#   munch    -> lac/perception/depth.py
#   segmentation-models-pytorch -> lac/perception/segmentation.py
#   opt_einsum -> thirdparty/raft_stereo
pip install imageio munch segmentation-models-pytorch opt_einsum
```

**Two gotchas worth flagging again:**

1. **Pip transactions are atomic.** A single failing build (e.g. apriltag without cmake) aborts and rolls back *everything*. `Successfully installed lac-0.1.0` at the end can come from a later `pip install -e .` and look like success — verify with `python -c "import torch; print(torch.__version__)"` afterwards.
2. **PYTHONNOUSERSITE is non-negotiable.** See Section 1.1.

### 2.3 LightGlue

```bash
mkdir -p ~/opt && cd ~/opt
git clone https://github.com/cvg/LightGlue.git
cd LightGlue
pip install -e .
```

The JHU install guide says to manually patch `LightGlue/lightglue/lightglue.py:24` to `@torch.amp.custom_fwd(..., device_type='cuda')`. **No longer needed** — upstream LightGlue already uses the new form with a fallback (verified 2026-05).

### 2.4 LangSAM (deprecated — skip)

Not in our `requirements.txt`. The agent-path code does not import it.

### 2.5 Pretrained UNet segmentation weights

Required by `lac/perception/segmentation.py:UnetSegmentation` (`models/unet_v2.pth` at the repo root). Download from the JHU portal's "Model weights" link.

```bash
mkdir -p models
# Drop unet_v2.pth from JHU portal into here. ~100 MB file.
ls -la models/unet_v2.pth   # should exist; ~100 MB
```

`models/` is gitignored.

### 2.6 Verify the full agent-path import chain

```bash
conda activate lac
SIM="$HOME/Stanford/AA278/Lunar_Perception_Aware_Planning/LunarAutonomyChallenge/LunarAutonomyChallenge"
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

Expected last line: `PASS: agent-path imports green.`

---

## 3. LAC simulator (the JHU APL zip)

### 3.1 Acquire the zip

The simulator zip is distributed by JHU APL after challenge approval. It's ~12 GB unzipped.

**Don't unzip via Windows Explorer** — Windows hits 260-char path limits on some bundled Carla city assets (e.g. `CommercialStyle1_01`) and skips them. The lunar maps (`Moon_Map_01.uexp`, `Moon_Map_02.uexp`) are not affected, but cleaner to extract via WSL `unzip` to avoid the silent skips.

```bash
cd ~/Stanford/AA278/Lunar_Perception_Aware_Planning
# Either: copy the existing extracted folder from another distro:
#   cp -r /mnt/c/Users/<you>/LunarAutonomyChallenge ./LunarAutonomyChallenge
# Or: unzip the original zip via WSL:
#   unzip /mnt/c/Users/<you>/Downloads/LunarAutonomyChallenge.zip -d ./LunarAutonomyChallenge
```

The resulting layout is **double-nested** (the zip already contains a `LunarAutonomyChallenge/` top-level folder):

```
$REPO/LunarAutonomyChallenge/LunarAutonomyChallenge/    <-- the sim
├── RunLunarSimulator.sh, RunLeaderboard.sh
├── requirements.txt           (top-level — has the carla wheel)
├── agents/                    (JHU example agents: dummy/human/opencv)
├── Leaderboard/
│   ├── data/missions_training.xml   (mission_id ↔ preset mapping)
│   ├── leaderboard/leaderboard_evaluator.py
│   └── requirements.txt       (leaderboard-only subset)
├── LunarSimulator/
│   ├── LAC.sh
│   └── LAC/Binaries/Linux/LAC-Linux-Shipping   (UE4 binary)
└── wheelhouse/
    └── carla-0.9.15-cp310-cp310-manylinux_2_27_x86_64.whl
```

The entire `LunarAutonomyChallenge/` directory is gitignored in this repo.

### 3.2 Install Carla + missing sim deps (selective, do NOT use --force-reinstall)

The sim's `requirements.txt` would downgrade `numpy 1.26 → 1.21.4` and `matplotlib 3.10 → 3.7.5` via `--force-reinstall`. **Don't do that** — numpy 1.26.4 is still numpy 1.x (ABI-compatible with Carla 0.9.15) and our SLAM code is tested with it. Install only the missing pieces:

```bash
SIM="$HOME/Stanford/AA278/Lunar_Perception_Aware_Planning/LunarAutonomyChallenge/LunarAutonomyChallenge"
pip install "$SIM/wheelhouse/carla-0.9.15-cp310-cp310-manylinux_2_27_x86_64.whl"
pip install dictor==0.1.12 tabulate==0.9.0 pygame==2.5.2
# astropy + lunarsky are pulled in by our repo's requirements.txt already.
```

Verify:
```bash
python -c "import carla, dictor, tabulate, pygame; print('carla 0.9.15 OK')"
```

### 3.3 Make scripts executable + create output/

```bash
SIM="$HOME/Stanford/AA278/Lunar_Perception_Aware_Planning/LunarAutonomyChallenge/LunarAutonomyChallenge"
chmod +x "$SIM/RunLunarSimulator.sh" "$SIM/RunLeaderboard.sh" \
         "$SIM/LunarSimulator/LAC.sh" \
         "$SIM/LunarSimulator/LAC/Binaries/Linux/LAC-Linux-Shipping"
mkdir -p "$SIM/output"
```

### 3.4 Patch `RunLunarSimulator.sh` for WSL graphics

The bundled script doesn't forward args and doesn't set the WSL graphics env vars. Replace its contents with:

```bash
#!/bin/bash
# WSL2 + NVIDIA dGPU launch wrapper.
# - Route OpenGL via D3D12 to the NVIDIA dGPU (WSLg defaults to the Intel iGPU otherwise).
# - On Ubuntu 22.04 we disabled all Vulkan ICDs because only llvmpipe existed; on 24.04
#   with dzn working we want UE4 to use Vulkan, so leave the loader alone.
export MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SIMULATOR_ROOT="$SCRIPT_DIR/LunarSimulator"

# Forward any caller args to LAC.sh -> UE4 binary.
bash "$SIMULATOR_ROOT/LAC.sh" "$@"
```

(If 24.04's Vulkan ICD setup is shaky, you can also `export VK_LOADER_DRIVERS_DISABLE='*'` and try `-opengl` as a desperation flag — but UE4 4.26 will likely ignore it and crash anyway. Real fix is Section 1.3.)

### 3.5 Patch `RunLeaderboard.sh` to point at this repo

Replace the top of the bundled `RunLeaderboard.sh`:

```bash
#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Point at OUR forked repo instead of the sim's bundled example agents.
export LAC_BASE_PATH="$SCRIPT_DIR"
export LEADERBOARD_ROOT="$SCRIPT_DIR/Leaderboard"
export TEAM_CODE_ROOT="$HOME/Stanford/AA278/Lunar_Perception_Aware_Planning"

export PYTHONPATH="$LEADERBOARD_ROOT:$TEAM_CODE_ROOT:$PYTHONPATH"

export TEAM_AGENT="$TEAM_CODE_ROOT/agents/nav_agent.py"
export TEAM_CONFIG="$TEAM_CODE_ROOT/configs/config.json"
```

(Leave the rest of the file — MISSIONS, MISSIONS_SUBSET, CHECKPOINT_ENDPOINT, REPETITIONS, RECORD, SEED, and the python invocation — unchanged.)

The default `MISSIONS_SUBSET=0` selects mission id 0 = preset 1 (per `missions_training.xml`). Change to `MISSIONS_SUBSET=1` to run preset 2 (which has our matched DEM at `data/DEMs/Moon_Map_01_2_rep0.dat`).

### 3.6 Pre-commit hooks (optional)

```bash
cd ~/Stanford/AA278/Lunar_Perception_Aware_Planning
pip install pre-commit
pre-commit install
```

---

## 4. Running the simulator

Two terminals, both inside `~/Stanford/AA278/Lunar_Perception_Aware_Planning/LunarAutonomyChallenge/LunarAutonomyChallenge`.

### 4.1 Terminal A — base simulator (keep open between agent runs)

```bash
conda activate lac
./RunLunarSimulator.sh
```

**Watch for:** an Unreal Engine window opens, lunar terrain renders at >10 fps, GPU panel shows NVIDIA RTX 4060 (not Intel Iris Xe). In-window controls: WASD = horizontal translate, mouse-drag = aim, Q/E = down/up. **Close with the X button, not Ctrl-C** — Ctrl-C may leave the Carla server zombied.

### 4.2 Terminal B — agent / leaderboard

```bash
conda activate lac
./RunLeaderboard.sh
```

**Watch for:** `> Loading the world`, `> Setting up the simulation`, then `Step: 1`, `Step: 2`, ... in the console. The agent processes images every other step starting from step 80 (the ARM_RAISE_WAIT_FRAMES delay).

Ground-truth heightmap for the active preset is dumped to `LunarAutonomyChallenge/LunarAutonomyChallenge/results/Moon_Map_01_<PRESET>_rep0.dat` at the end of the run. Copy it to `data/DEMs/` in this repo for downstream Phase 0 / planner use.

---

## 5. Common WSL-specific failures and fixes

| Symptom | Cause | Fix |
|---|---|---|
| Sim window opens, immediately crashes with "OpenGL no longer supported" + GameThread timeout | UE4 4.26 tried Vulkan, got llvmpipe | Section 1.3 — fix Vulkan ICDs |
| `vulkaninfo` shows only llvmpipe even on 24.04 | `dzn_icd.x86_64.json` not in `/usr/share/vulkan/icd.d/` | See Section 1.3 troubleshooting |
| Sim runs but feels slow (<10 fps when nothing is happening) | WSLg routed graphics to Intel iGPU | `export MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA`. Already set in `RunLunarSimulator.sh`. |
| `torch.cuda.is_available()` returns False | Wrong torch wheel OR `/usr/lib/wsl/lib/libcuda.so` not visible | `pip install --force-reinstall torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121`. If needed, `export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH`. |
| `pip install -r requirements.txt` fails on apriltag | CMake 4.x rejects apriltag's old CMakeLists | `pip install "cmake<4"` then `pip install --no-build-isolation apriltag==0.0.16` BEFORE the requirements install |
| `python -c "import torch"` returns a version not in `requirements.txt` | User-site shadowing | Section 1.1 — `PYTHONNOUSERSITE=1` on the lac env |
| RunLeaderboard.sh "ModuleNotFoundError: leaderboard" | PYTHONPATH not set | Don't run `nav_agent.py` directly — always via `./RunLeaderboard.sh` which sets PYTHONPATH |
| Ctrl-C leaves Carla orphaned | UE4 quirk | Close with X. Clean up with `pkill -f 'LAC-Linux-Shipping'`. |
| Sim window doesn't open at all | WSLg not running / no DISPLAY | `echo $DISPLAY` should be `:0`; if empty, `wsl --shutdown` from PowerShell and start fresh. May also need `wsl --update`. |

---

## 6. Quick sanity-check command sequence

After a fresh shell, this verifies everything is ready without launching the sim:

```bash
conda activate lac
echo "Python    : $(python --version)"
echo "Torch     : $(python -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"\")')"
echo "Carla     : $(python -c 'import carla; print(\"0.9.15 importable\")' 2>&1)"
echo "Vulkan    : $(vulkaninfo --summary 2>/dev/null | grep -A 1 'deviceName' | head -2)"
echo "OpenGL    : $(MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA glxinfo -B 2>/dev/null | grep 'OpenGL renderer' | head -1)"
echo "CUDA      : $(nvidia-smi --query-gpu=name,driver_version,memory.free --format=csv,noheader,nounits)"
echo "Sim dir   : $(test -d ~/Stanford/AA278/Lunar_Perception_Aware_Planning/LunarAutonomyChallenge/LunarAutonomyChallenge && echo 'present' || echo 'MISSING')"
echo "DEM       : $(test -f ~/Stanford/AA278/Lunar_Perception_Aware_Planning/data/DEMs/Moon_Map_01_2_rep0.dat && echo 'present' || echo 'MISSING')"
echo "UNet      : $(test -f ~/Stanford/AA278/Lunar_Perception_Aware_Planning/models/unet_v2.pth && echo 'present' || echo 'MISSING')"
```

Expected:
- Python 3.10.20, Torch 2.4.1+cu121 + CUDA True
- Carla 0.9.15 importable
- Vulkan device line containing "NVIDIA" or "Direct3D12" (NOT just llvmpipe)
- OpenGL renderer line containing "NVIDIA"
- All `present` for sim/DEM/UNet

---

## 7. End-state checklist

When all of the following are true, you're ready for Phase 0 re-run and planner build:

- [ ] `vulkaninfo --summary` shows NVIDIA / D3D12 device, NOT only llvmpipe
- [ ] `vkcube` opens and renders a spinning cube at >30 fps
- [ ] `./RunLunarSimulator.sh` opens a window and renders a lunar scene
- [ ] `./RunLeaderboard.sh` in a second terminal prints `Step: 1, 2, 3, ...` and the rover moves
- [ ] After a run, a `results/Moon_Map_01_<PRESET>_rep0.dat` file appears in the sim folder
- [ ] `python scripts/phase0_validation.py` runs end-to-end on the preset 2 DEM + frames (already proven working on 22.04; just re-verify in 24.04)

The next planning/research step after this is documented in `docs/PROJECT_TURNOVER.md`.
