"""Phase 0 validation: does DEM-derived height roughness predict SuperPoint feature density?

Tests the foundational hypothesis of the perception-aware path planner against a matched
LAC dataset: preset 2 DEM (`data/DEMs/Moon_Map_01_2_rep0.dat`) + 2000 stereo frames with
ground-truth poses from the same preset (`data/Example_Implementations/HW3_Final/data/lac_data/`).

For each frame, this script samples DEM roughness and ground-truth rock density at the rover
position and at several look-ahead distances along the rover's heading, then extracts SuperPoint
features (same configuration as the SLAM uses) and correlates the two. A weak correlation here
forces a pivot in the planner design before any planner code is written.

Outputs are written under `output/phase0_validation/` (gitignored). The plan file is
`/home/sthorup/.claude/plans/to-answer-your-questions-resilient-seal.md`.

NOTE on the data: the 1:1 mapping between this `data_log.json` (2000 frames) and the FrontLeft
PNGs holds because the source was a `data_collection_agent` variant. A generic `nav_agent` run
would log every step (see `agents/nav_agent.py:365`), so a future regeneration may need an even-
step filter to reproduce the alignment.

Phase 0.5 caveats (shadow-aware ρ_full extension):
- Sun direction in the LAC world frame is HYPOTHESIZED from astropy + lunarsky at the preset 2
  initial epoch (az=263.575° CCW from world +X, alt=1.488°). The mapping from astropy's lunar-
  topocentric convention to LAC's world-frame XY convention is unverified; visual verification
  against shadow directions in `frame_examples.png` is required. If shadows are wrong, sweep
  SUN_AZIMUTH_DEG over the candidates documented next to the constant.
- The `sun_factor` term from the original ρ formula (Project_Background.md:462-474) is dropped:
  at alt=1.488° it evaluates to a constant clip(sin(0.026)*10, 0.1, 1.0) = 0.26 and so cannot
  affect rank-based correlations on a single mission.
- Rover-cast shadow on its own camera frustum is unmodeled and inflates n_features in some
  frames (a known confound on the predictor's signal).
- The lander is a fixed, texture-rich object at the world origin in every preset; its keypoints
  are uncorrelated with terrain ρ_full. They are dropped via the UNet LANDER class (MASK_LANDER,
  on by default), and the pre-mask count is retained as n_features_raw to quantify the confound.
- The dataset is overridable via PHASE0_DATA_DIR. The default lac_data set is a ~7 m × 15 m,
  2000-frame preset-2 trajectory (too little terrain variation -> weak ρ_full correlation does NOT
  falsify the predictor); the phase0_transect run covers the full ±11 m map for a stronger test.
"""

from __future__ import annotations

import json
import os
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from lightglue import LightGlue, SuperPoint
from lightglue.utils import rbd
from PIL import Image
from scipy.ndimage import uniform_filter
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

from lac.util import grayscale_to_3ch_tensor
from lac.perception.segmentation import UnetSegmentation, SemanticClasses

# The DEM field math (roughness, shadow ray-cast, rho_full, world<->grid) is shared with the
# perception-aware planner -- lac/planning/perception_map.py is the canonical home, so this script
# and the planner are guaranteed to use identical, validated code. load_lac_dem is the same loader
# previously named load_dem here.
from lac.planning.perception_map import (
    load_lac_dem as load_dem,
    compute_roughness_field,
    compute_rock_density_field,
    compute_shadow_mask,
    compute_rho_full,
    world_to_grid,
    _test_shadow_pole,
)

# ============================================================================
# CONFIG
# ============================================================================

REPO_ROOT = Path(__file__).resolve().parent.parent
# DEM is overridable so the same script validates any preset. The sun is FIXED across all presets
# (MissionWeather hardcodes lat=-90, lon=0, date 2023-01-15 in leaderboard/missionmanager/
# mission_weather.py), so SUN_AZIMUTH_DEG/SUN_ALTITUDE_DEG below stay valid for every preset and
# only the DEM needs to change. A preset's ground-truth DEM is written to
# LAC_SIM/results/Moon_Map_01_<MISSIONS_SUBSET>_rep0.dat by any completed run (statistics_manager).
DEM_PATH = Path(
    os.environ.get("PHASE0_DEM_PATH", REPO_ROOT / "data" / "DEMs" / "Moon_Map_01_2_rep0.dat")
)

# Input dataset. Override with PHASE0_DATA_DIR to analyze a new collection run (e.g. the
# phase0_transect output); the directory must contain data_log.json + FrontLeft/ + FrontRight/.
# Defaults to the original preset-2 lac_data set.
_DEFAULT_DATA_DIR = (
    REPO_ROOT / "data" / "Example_Implementations" / "HW3_Final" / "data" / "lac_data"
)
DATA_DIR = Path(os.environ.get("PHASE0_DATA_DIR", _DEFAULT_DATA_DIR))
LOG_PATH = DATA_DIR / "data_log.json"
IMG_DIR = DATA_DIR / "FrontLeft"
IMG_DIR_RIGHT = DATA_DIR / "FrontRight"
OUT_DIR = Path(
    os.environ.get("PHASE0_OUT_DIR", REPO_ROOT / "output" / "phase0_validation_matched_az263")
)

# Map params (mirror lac/params.py:55-57)
MAP_SIZE = 180
MAP_EXTENT = 13.5
CELL_WIDTH = 0.15

# Algorithm knobs
ROUGHNESS_WINDOW_CELLS = 5
LOOKAHEAD_DISTANCES_M = (1.0, 1.5, 2.0, 3.0, 4.0)
SUPERPOINT_MAX_KP = 2048  # SLAM uses 512; raised here after 91.9% saturation in the first run.
                          # Original 512-cap results are preserved in output/phase0_validation_kp512/.
SATURATION_PCT_WARN = 10.0

# Sun direction for preset 2 (Phase 0.5).
# Computed offline via `get_sun().transform_to(LunarTopo(MoonLocation(lat=-90, lon=0)))` at
# Time("2023-01-15 00:00:00") (matches `mission_weather.py:106-114`). Drift across the 200-s
# preset-2 trajectory ≤ 0.03° in azimuth -> treat as constant.
# CONVENTION: az is degrees CCW from world +X axis in the world XY plane; alt is degrees above
# horizon. If shadows fall in the WRONG direction in dem_overlays.png (visual verification),
# sweep SUN_AZIMUTH_DEG over: {83.575, 96.425, 186.425, 0, 90, 180, 270} until shadows align.
SUN_AZIMUTH_DEG = 263.575  # astropy value (CCW from world +X). The 180°-flipped A/B value 83.575°
                           # was rejected: it gave H4b ≈ −0.36 (worse direction). See output/
                           # phase0_validation_az263_575/ (kept) vs phase0_validation/ (the 83.575° run).
SUN_ALTITUDE_DEG = 1.488
SHADOW_RAY_EPS = 1e-3  # vertical lift (m) preventing self-shadow on flat terrain

# Feature matching (Phase 0.5 v2): matched features are the SLAM-usable subset of SuperPoint
# keypoints. We count both temporal (FrontLeft[i-1]<->FrontLeft[i]) and stereo
# (FrontLeft[i]<->FrontRight[i]) matches via LightGlue, mirroring lac/slam/feature_tracker.py.
MATCH_MIN_SCORE = 0.5          # loop-closure-grade confidence (cf. configs/*.json loop_closure.min_score)
MAX_TEMPORAL_GAP_STEPS = 2     # nominal image cadence (every other sim step); larger gap -> NaN temporal

# Lander masking (preset-agnostic confound removal). The lander is a fixed, man-made, texture-rich
# object at the world origin in EVERY preset, so SuperPoint keypoints landing on it inflate feature
# counts in a way uncorrelated with terrain roughness/shadow (it is also not movable by preset
# choice -- see missions_training.xml + params.LANDER_GLOBAL). We drop keypoints that fall on the
# UNet LANDER class (lac/perception/segmentation.py:SemanticClasses.LANDER) before counting and
# matching, so all three response variables exclude the lander. The unmasked count is retained as
# `n_features_raw` to quantify the confound. Disable with PHASE0_MASK_LANDER=0.
MASK_LANDER = os.environ.get("PHASE0_MASK_LANDER", "1").lower() not in ("0", "false", "no")

# Statistics
SUBSAMPLE_MIN_DIST_M = 1.0
BOOTSTRAP_BLOCK_SIZE = 50
BOOTSTRAP_N = 1000
BOOTSTRAP_CI_PCT = 95.0

# Plotting
EXAMPLE_FRAME_COUNT = 3
EXAMPLE_MIN_STEP_GAP = 50
EXAMPLE_PATCH_MARGIN_M = 1.5  # heatmap patch extends best_d + this beyond the rover (m), so the
                             # look-ahead marker always lands on the heatmap, not on black space


# ============================================================================
# Field computation (DEM, roughness, rock density)
# ============================================================================


@dataclass
class DEMFields:
    z: np.ndarray  # (180, 180) heightmap
    rock: np.ndarray  # (180, 180) ground-truth rock indicator
    roughness: np.ndarray  # (180, 180) window-std of z
    rock_density: np.ndarray  # (180, 180) window-mean of rock
    shadow_mask: np.ndarray  # (180, 180) bool, True = cell is in sun-shadow
    rho_full: np.ndarray  # (180, 180) roughness * (1 - shadow_mask); NaN propagates from roughness


# DEM field functions (load_dem, compute_roughness_field, compute_rock_density_field,
# compute_shadow_mask, compute_rho_full, _test_shadow_pole) now live in
# lac/planning/perception_map.py and are imported at the top of this file.


# ============================================================================
# Frame / pose handling
# ============================================================================


def load_frames(json_path: Path) -> list[dict]:
    """Load `data_log.json` and return the `frames` list directly."""
    with json_path.open() as f:
        log = json.load(f)
    return log["frames"]


def lookahead_xy(pose: np.ndarray, d: float) -> tuple[float, float]:
    """Point d meters ahead of the rover along its heading, projected to the XY plane.

    Rover +X is forward (per `lac/utils/frames.py:11`). Renormalizing the XY projection
    keeps a pitched-down rover producing a level look-ahead.
    """
    h_world = pose[:3, :3] @ np.array([1.0, 0.0, 0.0])
    hx, hy = h_world[0], h_world[1]
    n = np.hypot(hx, hy)
    if n < 1e-9:
        return float(pose[0, 3]), float(pose[1, 3])
    return float(pose[0, 3] + d * hx / n), float(pose[1, 3] + d * hy / n)


def sample_field(field: np.ndarray, ij: tuple[int, int] | None) -> float:
    """Look up a scalar field at grid cell (i, j); NaN if cell is None or value is non-finite."""
    if ij is None:
        return float("nan")
    v = float(field[ij[0], ij[1]])
    return v if np.isfinite(v) else float("nan")


# ============================================================================
# SuperPoint
# ============================================================================


def init_superpoint(max_kp: int) -> SuperPoint:
    """Instantiate the same SuperPoint configuration the SLAM uses."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available; SuperPoint needs GPU for this script")
    return SuperPoint(max_num_keypoints=max_kp).eval().cuda()


def init_matcher() -> LightGlue:
    """Instantiate the LightGlue matcher (same config as lac/slam/feature_tracker.py:59)."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available; LightGlue needs GPU for this script")
    return LightGlue(features="superpoint").eval().cuda()


def extract_features(
    extractor: SuperPoint, img: np.ndarray
) -> tuple[int, float, np.ndarray, np.ndarray, dict]:
    """Run SuperPoint on a (H, W) uint8 grayscale image.

    Returns (n_kpts, mean_score, kpts_xy (N,2), scores (N,), feats). `kpts_xy` are pixel coords in
    the EXTRACTED image (SuperPoint resizes internally; see lightglue/utils.py:142), rescaled back
    to original-image pixels for downstream plotting. `feats` is the raw batched SuperPoint output
    dict (cuda tensors, keypoints in EXTRACTED coords) suitable for feeding straight to LightGlue;
    the rescale above only mutates the CPU numpy copy, so `feats` is left untouched.
    """
    h0, w0 = img.shape
    tensor = grayscale_to_3ch_tensor(img).cuda()
    feats = extractor.extract(tensor)
    kpts = feats["keypoints"][0].cpu().numpy()
    scores = feats["keypoint_scores"][0].cpu().numpy()
    # Rescale kpts to original image coords using SuperPoint's reported image_size (W, H).
    sp_w, sp_h = feats["image_size"][0].cpu().numpy()
    kpts[:, 0] *= w0 / float(sp_w)
    kpts[:, 1] *= h0 / float(sp_h)
    n = int(len(kpts))
    mean_score = float(scores.mean()) if n > 0 else float("nan")
    return n, mean_score, kpts, scores, feats


def count_matches(matcher: LightGlue, feats_a: dict, feats_b: dict, min_score: float) -> int:
    """Count LightGlue matches between two raw SuperPoint feature dicts that exceed `min_score`."""
    # Lander masking can prune a frame down to 0 keypoints; LightGlue errors on an empty set.
    if feats_a["keypoints"].shape[1] == 0 or feats_b["keypoints"].shape[1] == 0:
        return 0
    m = rbd(matcher({"image0": feats_a, "image1": feats_b}))
    return int((m["scores"] > min_score).sum())


# ============================================================================
# Lander masking (preset-agnostic confound removal)
# ============================================================================


def init_segmentation() -> UnetSegmentation | None:
    """UNet++ segmentation model used for lander masking, or None when masking is disabled.

    Loads models/unet_v2.pth onto the GPU (the same model + grayscale camera feed the SLAM frontend
    uses in lac/slam/frontend.py). Returns None if PHASE0_MASK_LANDER is off.
    """
    if not MASK_LANDER:
        return None
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available; UNet segmentation needs GPU for lander masking")
    return UnetSegmentation()


def lander_mask_for_image(seg: UnetSegmentation, img: np.ndarray) -> np.ndarray:
    """(H, W) bool mask, True where the UNet predicts LANDER. `img` is (H, W) uint8 grayscale.

    UnetSegmentation.predict expects a 3-channel image (it runs cvtColor BGR2RGB internally), so we
    replicate the grayscale frame to 3 channels first. The returned class map is in original-image
    pixel coords (predict resizes its output back), aligning with extract_features' rescaled kpts.
    """
    img3 = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    pred = seg.predict(img3)
    return pred == SemanticClasses.LANDER.value


def _prune_feats(feats: dict, keep_idx: np.ndarray) -> dict:
    """Filter a SuperPoint feats dict to `keep_idx` keypoints along the keypoint axis.

    Mirrors lac/slam/feature_tracker.py:prune_features; `image_size` is left untouched so LightGlue's
    positional encoding stays correct. Keeps the batch dim.
    """
    idx = torch.as_tensor(keep_idx, dtype=torch.long, device=feats["keypoints"].device)
    return {k: (v if k == "image_size" else v[:, idx]) for k, v in feats.items()}


def extract_features_masked(
    extractor: SuperPoint, img: np.ndarray, seg: UnetSegmentation | None
) -> tuple[int, float, np.ndarray, np.ndarray, dict, int, float]:
    """extract_features, then drop keypoints on LANDER-class pixels.

    Returns (n, mean_score, kpts, scores, feats, n_raw, lander_frac). `feats` is pruned to the kept
    keypoints so downstream LightGlue matching also excludes the lander. With seg=None this reduces
    to extract_features with n_raw=n and lander_frac=0.0. `kpts`/`feats` keypoints share an index
    order (extract_features only rescales a CPU copy), so the keep-set computed from `kpts` prunes
    `feats` correctly.
    """
    n, mean_score, kpts, scores, feats = extract_features(extractor, img)
    n_raw = n
    lander_frac = 0.0
    if seg is not None and n > 0:
        lander = lander_mask_for_image(seg, img)
        h, w = lander.shape
        xi = np.clip(np.rint(kpts[:, 0]).astype(int), 0, w - 1)
        yi = np.clip(np.rint(kpts[:, 1]).astype(int), 0, h - 1)
        on_lander = lander[yi, xi]
        lander_frac = float(on_lander.mean())
        keep = np.flatnonzero(~on_lander)
        kpts, scores = kpts[keep], scores[keep]
        feats = _prune_feats(feats, keep)
        n = int(keep.size)
        mean_score = float(scores.mean()) if n > 0 else float("nan")
    return n, mean_score, kpts, scores, feats, n_raw, lander_frac


# ============================================================================
# Per-frame processing loop
# ============================================================================


def process_all_frames(
    frames: list[dict],
    extractor: SuperPoint,
    matcher: LightGlue,
    fields: DEMFields,
    distances: tuple[float, ...],
    seg: UnetSegmentation | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Iterate frames, extract features + matched-feature counts, sample DEM fields.

    Response variables per frame (all lander-masked when `seg` is provided):
      - n_features          : SuperPoint keypoint count on FrontLeft (auxiliary baseline).
      - n_matched_temporal  : LightGlue matches FrontLeft[i-1]<->FrontLeft[i] above MATCH_MIN_SCORE.
                              NaN on the first frame and across any image-cadence gap.
      - n_matched_stereo    : LightGlue matches FrontLeft[i]<->FrontRight[i]. NaN if right is missing.
    Plus diagnostics: n_features_raw (pre-mask count) and lander_kp_frac (fraction dropped).
    """
    rows: list[dict] = []
    skipped_no_image = 0
    skipped_no_image_right = 0
    skipped_oob_pose = 0
    # Pre-cast shadow once (bool -> float64) so sample_field can NaN-check via np.isfinite.
    shadow_float = fields.shadow_mask.astype(np.float64)

    prev_feats: dict | None = None  # FrontLeft feats of the previously processed frame
    prev_step: int | None = None

    for idx, frame in enumerate(tqdm(frames, desc="frames", unit="f")):
        step = int(frame["step"])
        pose = np.array(frame["pose"], dtype=np.float64)
        x, y = float(pose[0, 3]), float(pose[1, 3])
        heading_deg = float(np.degrees(np.arctan2(pose[1, 0], pose[0, 0])))

        ij_pose = world_to_grid(x, y)
        if ij_pose is None:
            skipped_oob_pose += 1
            # We still want a row for completeness; in-bounds flag handles exclusion later.

        img_path = IMG_DIR / f"{step:06d}.png"
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            skipped_no_image += 1
            warnings.warn(f"image not loadable: {img_path}")
            continue  # prev_feats/prev_step intentionally NOT updated -> next frame's temporal = NaN

        n_feats, mean_score, _, _, feats_l, n_feats_raw, lander_frac = extract_features_masked(
            extractor, img, seg
        )

        # Temporal match: FrontLeft[i-1] <-> FrontLeft[i], only if frames are cadence-adjacent.
        if prev_feats is not None and (step - prev_step) <= MAX_TEMPORAL_GAP_STEPS:
            n_matched_temporal = count_matches(matcher, prev_feats, feats_l, MATCH_MIN_SCORE)
        else:
            n_matched_temporal = float("nan")

        # Stereo match: FrontLeft[i] <-> FrontRight[i].
        img_r = cv2.imread(str(IMG_DIR_RIGHT / f"{step:06d}.png"), cv2.IMREAD_GRAYSCALE)
        if img_r is None:
            skipped_no_image_right += 1
            n_matched_stereo = float("nan")
        else:
            _, _, _, _, feats_r, _, _ = extract_features_masked(extractor, img_r, seg)
            n_matched_stereo = count_matches(matcher, feats_l, feats_r, MATCH_MIN_SCORE)

        prev_feats, prev_step = feats_l, step

        row: dict = {
            "frame_idx": idx,
            "step": step,
            "image_filename": img_path.name,
            "x": x,
            "y": y,
            "heading_deg": heading_deg,
            "n_features": n_feats,
            "n_features_raw": n_feats_raw,
            "lander_kp_frac": lander_frac,
            "n_matched_temporal": n_matched_temporal,
            "n_matched_stereo": n_matched_stereo,
            "mean_score": mean_score,
            "roughness_at_pose": sample_field(fields.roughness, ij_pose),
            "rock_density_at_pose": sample_field(fields.rock_density, ij_pose),
            "shadow_at_pose": sample_field(shadow_float, ij_pose),
            "rho_full_at_pose": sample_field(fields.rho_full, ij_pose),
            "in_bounds_pose": ij_pose is not None,
        }
        for d in distances:
            xl, yl = lookahead_xy(pose, d)
            ij_la = world_to_grid(xl, yl)
            row[f"roughness_la_{d}"] = sample_field(fields.roughness, ij_la)
            row[f"rock_density_la_{d}"] = sample_field(fields.rock_density, ij_la)
            row[f"shadow_la_{d}"] = sample_field(shadow_float, ij_la)
            row[f"rho_full_la_{d}"] = sample_field(fields.rho_full, ij_la)
            row[f"in_bounds_la_{d}"] = ij_la is not None
        rows.append(row)

        if (idx + 1) % 200 == 0:
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    counters = {
        "n_frames_input": len(frames),
        "n_frames_processed": len(df),
        "n_skipped_no_image": skipped_no_image,
        "n_skipped_no_image_right": skipped_no_image_right,
        "n_skipped_oob_pose": skipped_oob_pose,
        "mask_lander": bool(seg is not None),
        "mean_lander_kp_frac": float(df["lander_kp_frac"].mean()) if len(df) else 0.0,
        "mean_n_features_raw": float(df["n_features_raw"].mean()) if len(df) else 0.0,
        "mean_n_features_masked": float(df["n_features"].mean()) if len(df) else 0.0,
    }
    return df, counters


# ============================================================================
# Statistics
# ============================================================================


def subsample_iid(df: pd.DataFrame, min_dist_m: float) -> np.ndarray:
    """Greedy IID subsample: keep a frame iff its displacement from the last kept frame is >= min_dist_m."""
    xs = df["x"].to_numpy()
    ys = df["y"].to_numpy()
    kept: list[int] = [0]
    last_x, last_y = xs[0], ys[0]
    for i in range(1, len(df)):
        if np.hypot(xs[i] - last_x, ys[i] - last_y) >= min_dist_m:
            kept.append(i)
            last_x, last_y = xs[i], ys[i]
    return np.asarray(kept, dtype=np.int64)


def block_bootstrap_r(
    x: np.ndarray, y: np.ndarray, block_size: int, n_resamples: int, ci_pct: float, rng: np.random.Generator
) -> tuple[float, float, float]:
    """Block bootstrap CI on Pearson r for a time-correlated series.

    Returns (mean_r_across_resamples, ci_low, ci_high).
    """
    n = len(x)
    if n < 2 * block_size:
        return float("nan"), float("nan"), float("nan")
    n_blocks_needed = int(np.ceil(n / block_size))
    max_start = n - block_size
    rs = np.empty(n_resamples, dtype=np.float64)
    for k in range(n_resamples):
        starts = rng.integers(0, max_start + 1, size=n_blocks_needed)
        idx = np.concatenate([np.arange(s, s + block_size) for s in starts])[:n]
        rs[k] = np.corrcoef(x[idx], y[idx])[0, 1]
    alpha = (100.0 - ci_pct) / 2.0
    return float(np.nanmean(rs)), float(np.nanpercentile(rs, alpha)), float(np.nanpercentile(rs, 100 - alpha))


def _pair_dropna(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return paired arrays with rows containing NaN removed."""
    mask = np.isfinite(a) & np.isfinite(b)
    return a[mask], b[mask]


def correlate(
    x_raw: np.ndarray,
    y_raw: np.ndarray,
    iid_indices: np.ndarray,
    block_size: int,
    n_boot: int,
    ci_pct: float,
    rng: np.random.Generator,
) -> dict:
    """Compute raw Pearson, IID-subsampled Pearson, Spearman, and block-bootstrap CI."""
    x_full, y_full = _pair_dropna(x_raw, y_raw)
    pearson_raw = float(pearsonr(x_full, y_full)[0]) if len(x_full) >= 2 else float("nan")
    spearman = float(spearmanr(x_full, y_full).statistic) if len(x_full) >= 2 else float("nan")
    iid_mask = np.isin(np.arange(len(x_raw)), iid_indices)
    x_iid, y_iid = _pair_dropna(x_raw[iid_mask], y_raw[iid_mask])
    pearson_iid = float(pearsonr(x_iid, y_iid)[0]) if len(x_iid) >= 2 else float("nan")
    _, ci_low, ci_high = block_bootstrap_r(x_full, y_full, block_size, n_boot, ci_pct, rng)
    return {
        "pearson_raw": pearson_raw,
        "pearson_iid": pearson_iid,
        "spearman": spearman,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_raw": int(len(x_full)),
        "n_iid": int(len(x_iid)),
    }


def compute_all_correlations(
    df: pd.DataFrame,
    distances: tuple[float, ...],
    rng: np.random.Generator,
    response_col: str = "n_features",
) -> dict:
    """Compute H1a, H1b/H2/H3 (per d), H4a, H4b (per d) for `response_col`.

    H2/H3 are response-independent (rocks vs roughness) but are recomputed per call for a
    self-contained result. best_d is driven by H4b (PRIMARY). The IID subsample is spatial, so it
    is identical across response variables; NaN rows in the response drop out via _pair_dropna.
    """
    iid = subsample_iid(df, SUBSAMPLE_MIN_DIST_M)
    nf = df[response_col].to_numpy(dtype=np.float64)
    args = (iid, BOOTSTRAP_BLOCK_SIZE, BOOTSTRAP_N, BOOTSTRAP_CI_PCT, rng)

    h1a = correlate(df["roughness_at_pose"].to_numpy(), nf, *args)
    h4a = correlate(df["rho_full_at_pose"].to_numpy(), nf, *args)

    per_d: dict[float, dict] = {}
    for d in distances:
        rough_la = df[f"roughness_la_{d}"].to_numpy()
        rock_la = df[f"rock_density_la_{d}"].to_numpy()
        rho_la = df[f"rho_full_la_{d}"].to_numpy()
        per_d[d] = {
            "h1b": correlate(rough_la, nf, *args),
            "h2": correlate(rock_la, nf, *args),
            "h3": correlate(rough_la, rock_la, *args),
            "h4b": correlate(rho_la, nf, *args),
        }

    def _argmax_pearson(metric_key: str) -> float:
        return max(
            distances,
            key=lambda d: (per_d[d][metric_key]["pearson_raw"] if np.isfinite(per_d[d][metric_key]["pearson_raw"]) else -np.inf),
        )

    best_d = _argmax_pearson("h4b")
    best_d_h1b = _argmax_pearson("h1b")
    return {
        "response_col": response_col,
        "h1a": h1a, "h4a": h4a, "per_d": per_d,
        "best_d": best_d, "best_d_h1b": best_d_h1b,
        "n_iid_total": int(len(iid)),
    }


# ============================================================================
# Example frame selection
# ============================================================================


def select_example_frames(df: pd.DataFrame, best_d: float, n: int, min_gap: int) -> list[int]:
    """Pick frames spanning the (rho_full, features) range at the H4b-optimal lookahead distance."""
    rho = df[f"rho_full_la_{best_d}"].to_numpy()
    nf = df["n_features"].to_numpy(dtype=np.float64)
    valid = np.isfinite(rho) & np.isfinite(nf)
    if valid.sum() < n:
        return df.index[:n].tolist()

    score_high = rho * nf  # high-rho, high-feature
    score_low = -(rho + 1e-3) - (nf + 1e-3)  # low-rho, low-feature

    # Residual from OLS fit -> off-diagonal example.
    r, n_ = rho[valid], nf[valid]
    slope = np.cov(r, n_, bias=True)[0, 1] / max(np.var(r), 1e-12)
    intercept = n_.mean() - slope * r.mean()
    residual_all = np.full_like(rho, -np.inf)
    residual_all[valid] = np.abs(nf[valid] - (slope * rho[valid] + intercept))

    candidates = [int(np.nanargmax(score_high)), int(np.nanargmax(score_low)), int(np.nanargmax(residual_all))]

    # Enforce min-gap on step index, preferring higher-scoring candidates.
    chosen: list[int] = []
    steps = df["step"].to_numpy()
    for c in candidates:
        if all(abs(steps[c] - steps[k]) >= min_gap for k in chosen):
            chosen.append(c)
        if len(chosen) >= n:
            break
    # Pad with random valid frames if too few.
    while len(chosen) < n:
        pool = [i for i in df.index if i not in chosen]
        chosen.append(pool[len(pool) // 2])
    return chosen[:n]


# ============================================================================
# Plotting
# ============================================================================


def _imshow_field(ax, field: np.ndarray, cmap: str, title: str, label: str) -> None:
    """imshow a (180,180) field in world coordinates with x right, y up."""
    im = ax.imshow(
        field.T,
        extent=[-MAP_EXTENT, MAP_EXTENT, -MAP_EXTENT, MAP_EXTENT],
        origin="lower",
        cmap=cmap,
        aspect="equal",
    )
    ax.set_title(title)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    plt.colorbar(im, ax=ax, label=label, fraction=0.046, pad=0.04)


def plot_dem_overlays(fields: DEMFields, traj_xy: np.ndarray, out_path: Path) -> None:
    """2x3 DEM-overlay panel: elevation/roughness/rocks (top), shadow/rho_full/lit-rocks (bottom).

    The "lit-rock density" panel (rock_density gated by NOT-shadow) directly visualizes where the
    rover SHOULD be able to detect rock-anchored features. Use it to cross-check that the shadow
    direction is correctly oriented.
    """
    shadow_f = fields.shadow_mask.astype(np.float64)
    lit_rock = fields.rock_density * (1.0 - shadow_f)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    _imshow_field(axes[0, 0], fields.z, "terrain", "Elevation", "height (m)")
    _imshow_field(axes[0, 1], fields.roughness, "viridis", "Roughness (5x5 std)", "std (m)")
    _imshow_field(axes[0, 2], fields.rock_density, "Reds", "Rock density (5x5 mean)", "density")
    _imshow_field(
        axes[1, 0], shadow_f, "gray_r",
        f"Shadow mask (az={SUN_AZIMUTH_DEG:.1f}°, alt={SUN_ALTITUDE_DEG:.2f}°)", "1 = in shadow",
    )
    _imshow_field(axes[1, 1], fields.rho_full, "viridis", "rho_full = roughness * (1 - shadow)", "rho_full")
    _imshow_field(axes[1, 2], lit_rock, "Reds", "Lit-rock density (rock * (1 - shadow))", "density")

    for ax in axes.flat:
        ax.plot(traj_xy[:, 0], traj_xy[:, 1], color="black", lw=0.8, alpha=0.8)
        ax.plot(traj_xy[0, 0], traj_xy[0, 1], "o", color="lime", ms=8, mec="black", label="start")
        ax.plot(traj_xy[-1, 0], traj_xy[-1, 1], "s", color="red", ms=8, mec="black", label="end")
    axes[0, 0].legend(loc="lower right", fontsize=8)

    # Sun-direction arrow on the shadow panel. Shadows extend in the direction OPPOSITE the arrow.
    ax = axes[1, 0]
    arrow_len = MAP_EXTENT * 0.6
    az_rad = np.radians(SUN_AZIMUTH_DEG)
    ax.annotate(
        "", xy=(arrow_len * np.cos(az_rad), arrow_len * np.sin(az_rad)), xytext=(0, 0),
        arrowprops=dict(arrowstyle="->", color="orange", lw=2.5),
    )
    ax.text(0.5, 1.0, "→ sun", color="orange", fontsize=9, fontweight="bold")

    fig.suptitle(f"{DEM_PATH.name} with rover trajectory (Phase 0.5: shadow-aware)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_frame_examples(
    df: pd.DataFrame,
    example_idxs: list[int],
    extractor: SuperPoint,
    fields: DEMFields,
    best_d: float,
    out_path: Path,
    seg: UnetSegmentation | None = None,
) -> None:
    """For each example frame: image + keypoints, roughness/shadow overlay patch, rho_full patch.

    When `seg` is provided, the lander mask is overlaid (translucent red) and only the KEPT
    (non-lander) keypoints are scattered, so this doubles as the visual confirmation that masking
    matches the per-frame counts.
    """
    n = len(example_idxs)
    fig, axes = plt.subplots(n, 3, figsize=(15, 5 * n))
    if n == 1:
        axes = axes[None, :]
    # Size the patch so the rover AND its best_d look-ahead point both sit on the heatmap.
    H = int(np.ceil((best_d + EXAMPLE_PATCH_MARGIN_M) / CELL_WIDTH))

    for row_i, frame_idx in enumerate(example_idxs):
        row = df.iloc[frame_idx]
        img = cv2.imread(str(IMG_DIR / row["image_filename"]), cv2.IMREAD_GRAYSCALE)
        n_kp, _, kpts, scores, _, n_raw, lander_frac = extract_features_masked(extractor, img, seg)

        ax_img = axes[row_i, 0]
        ax_img.imshow(img, cmap="gray")
        if seg is not None:
            lander = lander_mask_for_image(seg, img)
            ax_img.imshow(np.ma.masked_where(~lander, lander), cmap="autumn", alpha=0.35)
        if n_kp > 0:
            sizes = 4 + 30 * (scores / (scores.max() + 1e-9))
            ax_img.scatter(kpts[:, 0], kpts[:, 1], s=sizes, c="red", alpha=0.6, edgecolors="none")
        ax_img.set_title(
            f"step={row['step']}, n_feats={n_kp} (raw {n_raw}, lander {lander_frac * 100:.0f}%), "
            f"match_t={row['n_matched_temporal']:.0f}, match_s={row['n_matched_stereo']:.0f}\n"
            f"rho_full@LA={row[f'rho_full_la_{best_d}']:.4f}, "
            f"shadow@LA={row[f'shadow_la_{best_d}']:.0f}"
        )
        ax_img.set_axis_off()

        # Patch coords centered on rover.
        ij_pose = world_to_grid(row["x"], row["y"])
        if ij_pose is None:
            continue
        ic, jc = ij_pose
        i0, i1 = max(0, ic - H), min(MAP_SIZE, ic + H + 1)
        j0, j1 = max(0, jc - H), min(MAP_SIZE, jc + H + 1)
        x_lo, x_hi = -MAP_EXTENT + i0 * CELL_WIDTH, -MAP_EXTENT + i1 * CELL_WIDTH
        y_lo, y_hi = -MAP_EXTENT + j0 * CELL_WIDTH, -MAP_EXTENT + j1 * CELL_WIDTH
        xl, yl = lookahead_xy(np.array(_frame_pose(row), dtype=np.float64), best_d)

        for col_i, (field, cmap, label) in enumerate(
            [(fields.rho_full, "viridis", f"rho_full @ best_d={best_d:.1f}m"),
             (fields.rock_density * (1.0 - fields.shadow_mask.astype(np.float64)), "Reds",
              f"lit-rock density @ best_d={best_d:.1f}m")],
            start=1,
        ):
            ax = axes[row_i, col_i]
            ax.imshow(field[i0:i1, j0:j1].T, extent=[x_lo, x_hi, y_lo, y_hi], origin="lower", cmap=cmap)
            ax.plot(row["x"], row["y"], "x", color="black", ms=12, mew=2, label="rover")
            ax.plot(xl, yl, "+", color="black", ms=12, mew=2, label=f"lookahead {best_d:.1f}m")
            ax.annotate(
                "", xy=(xl, yl), xytext=(row["x"], row["y"]),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
            )
            ax.set_title(
                f"{label}\nrho_full@LA={row[f'rho_full_la_{best_d}']:.4f}, "
                f"rock@LA={row[f'rock_density_la_{best_d}']:.3f}, "
                f"shadow@LA={row[f'shadow_la_{best_d}']:.0f}"
            )
            ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
            if col_i == 1:
                ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("Example frames: image features vs DEM-derived signals")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _frame_pose(row: pd.Series) -> list:
    """Rebuild a 4x4 pose-like list from a DataFrame row's x, y, heading_deg.

    We only used pose[:3,:3] for heading and pose[0:2,3] for position; reconstructing the rotation
    around z from heading_deg is sufficient for the look-ahead recomputation here.
    """
    h = np.radians(row["heading_deg"])
    c, s = np.cos(h), np.sin(h)
    return [[c, -s, 0.0, row["x"]], [s, c, 0.0, row["y"]], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]


def _scatter_with_fit(ax, x: np.ndarray, y: np.ndarray, title: str) -> None:
    """Scatter + OLS line + r-squared annotation. NaNs dropped internally."""
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    ax.scatter(x, y, s=8, alpha=0.4, edgecolors="none")
    if len(x) >= 2:
        slope, intercept = np.polyfit(x, y, 1)
        xs = np.array([x.min(), x.max()])
        ax.plot(xs, slope * xs + intercept, "r-", lw=1.5)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)


def _corr_label(stats: dict, name: str, d: float | None = None) -> str:
    """Short human-readable correlation label for plot titles."""
    d_str = f" @ d={d:.1f}m" if d is not None else ""
    return (
        f"{name}{d_str}\n"
        f"Pearson_raw={stats['pearson_raw']:.3f} [{stats['ci_low']:.3f},{stats['ci_high']:.3f}]  "
        f"Pearson_iid={stats['pearson_iid']:.3f} (n_iid={stats['n_iid']})  "
        f"Spearman={stats['spearman']:.3f}"
    )


def plot_hypothesis_scatter(df: pd.DataFrame, corr: dict, out_path: Path) -> None:
    """3x2 grid of hypothesis scatter plots at the H4b-optimal lookahead distance.

    Row 0: H1a (roughness@pose) / H1b (roughness@lookahead).
    Row 1: H2 (rocks@lookahead) / H3 (roughness vs rocks).
    Row 2: H4a (rho_full@pose) / H4b PRIMARY (rho_full@lookahead).

    The response variable (n_features / n_matched_temporal / n_matched_stereo) is read from
    `corr["response_col"]`.
    """
    best_d = corr["best_d"]
    resp = corr["response_col"]
    nf = df[resp].to_numpy(dtype=np.float64)
    fig, axes = plt.subplots(3, 2, figsize=(13, 15))

    _scatter_with_fit(axes[0, 0], df["roughness_at_pose"].to_numpy(), nf,
                      _corr_label(corr["h1a"], f"H1a: roughness@pose -> {resp}"))
    axes[0, 0].set_xlabel("roughness @ rover position (m)"); axes[0, 0].set_ylabel(resp)

    _scatter_with_fit(axes[0, 1], df[f"roughness_la_{best_d}"].to_numpy(), nf,
                      _corr_label(corr["per_d"][best_d]["h1b"], f"H1b: roughness@lookahead -> {resp}", best_d))
    axes[0, 1].set_xlabel(f"roughness @ lookahead {best_d:.1f}m (m)"); axes[0, 1].set_ylabel(resp)

    _scatter_with_fit(axes[1, 0], df[f"rock_density_la_{best_d}"].to_numpy(), nf,
                      _corr_label(corr["per_d"][best_d]["h2"], f"H2: rock_density@lookahead -> {resp}", best_d))
    axes[1, 0].set_xlabel(f"rock density @ lookahead {best_d:.1f}m"); axes[1, 0].set_ylabel(resp)

    _scatter_with_fit(axes[1, 1], df[f"roughness_la_{best_d}"].to_numpy(),
                      df[f"rock_density_la_{best_d}"].to_numpy(),
                      _corr_label(corr["per_d"][best_d]["h3"], "H3: roughness -> rocks (proxy quality)", best_d))
    axes[1, 1].set_xlabel(f"roughness @ lookahead {best_d:.1f}m (m)")
    axes[1, 1].set_ylabel(f"rock density @ lookahead {best_d:.1f}m")

    _scatter_with_fit(axes[2, 0], df["rho_full_at_pose"].to_numpy(), nf,
                      _corr_label(corr["h4a"], f"H4a: rho_full@pose -> {resp}"))
    axes[2, 0].set_xlabel("rho_full @ rover position"); axes[2, 0].set_ylabel(resp)

    _scatter_with_fit(axes[2, 1], df[f"rho_full_la_{best_d}"].to_numpy(), nf,
                      _corr_label(corr["per_d"][best_d]["h4b"], f"H4b PRIMARY: rho_full@lookahead -> {resp}", best_d))
    axes[2, 1].set_xlabel(f"rho_full @ lookahead {best_d:.1f}m"); axes[2, 1].set_ylabel(resp)

    fig.suptitle(f"Phase 0.5 hypothesis scatter [{resp}] (best d={best_d:.1f}m driven by H4b)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_r_vs_lookahead(corr: dict, out_path: Path) -> None:
    """Line plot of correlation strength vs lookahead distance, comparing H1b and H4b predictors."""
    resp = corr["response_col"]
    ds = sorted(corr["per_d"].keys())

    def _series(metric: str, field: str) -> list[float]:
        return [corr["per_d"][d][metric][field] for d in ds]

    fig, ax = plt.subplots(figsize=(10, 6))

    # x=0 sanity: rover-position correlations for both predictors.
    ax.scatter([0], [corr["h1a"]["pearson_raw"]], color="C0", marker="x", s=80, label="H1a Pearson_raw @ pose")
    ax.scatter([0], [corr["h4a"]["pearson_raw"]], color="C3", marker="x", s=80, label="H4a Pearson_raw @ pose")

    # H1b family (roughness alone).
    ax.plot(ds, _series("h1b", "pearson_raw"), "o-", color="C0", label="H1b Pearson_raw (roughness@LA)")
    ax.plot(ds, _series("h1b", "pearson_iid"), "s--", color="C1", label="H1b Pearson_iid")
    ax.plot(ds, _series("h1b", "spearman"), "^-", color="C2", label="H1b Spearman")
    ax.fill_between(ds, _series("h1b", "ci_low"), _series("h1b", "ci_high"),
                    alpha=0.20, color="C0", label=f"H1b {int(BOOTSTRAP_CI_PCT)}% bootstrap CI")

    # H4b family (rho_full).
    ax.plot(ds, _series("h4b", "pearson_raw"), "o-", color="C3", label="H4b Pearson_raw (rho_full@LA)")
    ax.plot(ds, _series("h4b", "pearson_iid"), "s--", color="C4", label="H4b Pearson_iid")
    ax.plot(ds, _series("h4b", "spearman"), "^-", color="C5", label="H4b Spearman")
    ax.fill_between(ds, _series("h4b", "ci_low"), _series("h4b", "ci_high"),
                    alpha=0.15, color="C3", label=f"H4b {int(BOOTSTRAP_CI_PCT)}% bootstrap CI")

    ax.axvline(corr["best_d"], ls=":", color="black", label=f"best d (H4b) = {corr['best_d']:.1f}m")
    if corr.get("best_d_h1b", corr["best_d"]) != corr["best_d"]:
        ax.axvline(corr["best_d_h1b"], ls=":", color="gray", label=f"best d (H1b) = {corr['best_d_h1b']:.1f}m")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlabel("Lookahead distance (m)  -- x=0 is rover position")
    ax.set_ylabel(f"Correlation with {resp}")
    ax.set_title(f"H1b vs H4b [{resp}]: roughness alone vs rho_full predictor")
    ax.grid(alpha=0.3); ax.legend(fontsize=7, loc="best", ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_roughness_quartile_bars(
    df: pd.DataFrame, best_d: float, out_path: Path, response_col: str = "n_features"
) -> None:
    """Mean `response_col` per quartile of roughness@best_d. Monotonic increase = signal present."""
    rough = df[f"roughness_la_{best_d}"].to_numpy()
    nf = df[response_col].to_numpy(dtype=np.float64)
    mask = np.isfinite(rough) & np.isfinite(nf)
    rough, nf = rough[mask], nf[mask]
    q = np.quantile(rough, [0.0, 0.25, 0.5, 0.75, 1.0])
    bins = np.digitize(rough, q[1:-1], right=False)  # 0..3
    means = [nf[bins == k].mean() if (bins == k).any() else np.nan for k in range(4)]
    stds = [nf[bins == k].std() if (bins == k).any() else np.nan for k in range(4)]
    labels = [f"Q{k+1}\n[{q[k]:.3f},{q[k+1]:.3f}]" for k in range(4)]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, means, yerr=stds, capsize=4, color="steelblue", edgecolor="black")
    ax.set_ylabel(f"{response_col} (mean ± std)")
    ax.set_xlabel(f"Quartile of roughness @ lookahead {best_d:.1f}m (m)")
    ax.set_title(f"Mean {response_col} per roughness quartile (best d={best_d:.1f}m)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_timeline(
    df: pd.DataFrame, best_d: float, out_path: Path, response_col: str = "n_features"
) -> None:
    """Per-frame timeline view -- useful for debugging stationary segments or detector instability."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(df["frame_idx"], df[response_col], lw=0.8); axes[0].set_ylabel(response_col); axes[0].grid(alpha=0.3)
    axes[1].plot(df["frame_idx"], df[f"roughness_la_{best_d}"], lw=0.8, color="C2")
    axes[1].set_ylabel(f"roughness@LA d={best_d:.1f}m"); axes[1].grid(alpha=0.3)
    axes[2].plot(df["frame_idx"], df["heading_deg"], lw=0.8, color="C3")
    axes[2].set_ylabel("heading (deg)"); axes[2].set_xlabel("frame index"); axes[2].grid(alpha=0.3)
    fig.suptitle(f"Per-frame timeline [{response_col}] (debug)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_response_comparison(corr_by_resp: dict[str, dict], out_path: Path) -> None:
    """The money plot: H4b (rho_full@lookahead) correlation vs distance, one line per response var.

    Directly answers "did matched features help?": if n_matched_* lines sit above n_features, the
    SLAM-usable subset of keypoints tracks rho_full better than the raw count. Solid = Pearson_iid
    (the verdict statistic), dashed = Pearson_raw, X at x=0 = H4a (rho_full @ pose).
    """
    fig, ax = plt.subplots(figsize=(11, 6.5))
    colors = {"n_features": "C7", "n_matched_temporal": "C0", "n_matched_stereo": "C3",
              "n_features_raw": "C1"}  # raw (unmasked) shown vs masked n_features to expose the lander confound

    for resp, corr in corr_by_resp.items():
        ds = sorted(corr["per_d"].keys())
        c = colors.get(resp, None)
        iid = [corr["per_d"][d]["h4b"]["pearson_iid"] for d in ds]
        raw = [corr["per_d"][d]["h4b"]["pearson_raw"] for d in ds]
        ax.plot(ds, iid, "s-", color=c, label=f"{resp} Pearson_iid")
        ax.plot(ds, raw, "o--", color=c, alpha=0.5, label=f"{resp} Pearson_raw")
        ax.scatter([0], [corr["h4a"]["pearson_raw"]], color=c, marker="x", s=70)
        ax.axvline(corr["best_d"], ls=":", color=c, alpha=0.4)

    for thr in (0.2, 0.3, 0.5):  # verdict thresholds (|r|): WEAK/MODERATE/STRONG boundaries
        ax.axhline(thr, color="green", lw=0.6, ls=":", alpha=0.5)
        ax.axhline(-thr, color="green", lw=0.6, ls=":", alpha=0.5)
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlabel("Lookahead distance (m)  -- x=0 (X marks) is rho_full @ pose (H4a)")
    ax.set_ylabel("H4b correlation: rho_full@LA -> response")
    ax.set_title("Response-variable comparison: does matched-feature counting sharpen the rho_full signal?")
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="best", ncol=3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Summary / IO
# ============================================================================


def write_summary(
    df: pd.DataFrame,
    corr_by_resp: dict[str, dict],
    counters: dict,
    saturation_pct: float,
    runtime_s: float,
    out_dir: Path,
) -> Path:
    """Write summary.json and per_frame_metrics.csv. Returns the JSON path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "per_frame_metrics.csv"
    df.to_csv(csv_path, index=False)

    config = {
        "MAP_SIZE": MAP_SIZE, "MAP_EXTENT": MAP_EXTENT, "CELL_WIDTH": CELL_WIDTH,
        "ROUGHNESS_WINDOW_CELLS": ROUGHNESS_WINDOW_CELLS,
        "LOOKAHEAD_DISTANCES_M": list(LOOKAHEAD_DISTANCES_M),
        "SUPERPOINT_MAX_KP": SUPERPOINT_MAX_KP,
        "MATCH_MIN_SCORE": MATCH_MIN_SCORE,
        "MAX_TEMPORAL_GAP_STEPS": MAX_TEMPORAL_GAP_STEPS,
        "SUN_AZIMUTH_DEG": SUN_AZIMUTH_DEG,
        "SUN_ALTITUDE_DEG": SUN_ALTITUDE_DEG,
        "SHADOW_RAY_EPS": SHADOW_RAY_EPS,
        "SUBSAMPLE_MIN_DIST_M": SUBSAMPLE_MIN_DIST_M,
        "BOOTSTRAP_BLOCK_SIZE": BOOTSTRAP_BLOCK_SIZE,
        "BOOTSTRAP_N": BOOTSTRAP_N,
        "BOOTSTRAP_CI_PCT": BOOTSTRAP_CI_PCT,
        "MASK_LANDER": bool(counters.get("mask_lander", False)),
        "response_variables": {
            "n_features": "SuperPoint keypoint count on FrontLeft, lander-masked when MASK_LANDER (auxiliary baseline)",
            "n_matched_temporal": "LightGlue matches FrontLeft[i-1]<->FrontLeft[i] above MATCH_MIN_SCORE (lander-masked)",
            "n_matched_stereo": "LightGlue matches FrontLeft[i]<->FrontRight[i] above MATCH_MIN_SCORE (lander-masked)",
            "n_features_raw": "SuperPoint keypoint count on FrontLeft BEFORE lander masking (confound baseline)",
        },
    }

    def _corr_block(corr: dict) -> dict:
        return {
            "best_d": corr["best_d"],
            "best_d_h1b": corr.get("best_d_h1b", corr["best_d"]),
            "n_iid_subsample": corr["n_iid_total"],
            "h1a_roughness_at_pose": corr["h1a"],
            "h4a_rho_full_at_pose": corr["h4a"],
            "per_lookahead_distance": {
                str(d): corr["per_d"][d] for d in sorted(corr["per_d"].keys())
            },
        }

    summary = {
        "config": config,
        "counters": counters,
        "saturation_pct": saturation_pct,
        "runtime_s": runtime_s,
        "correlations_by_response": {resp: _corr_block(corr) for resp, corr in corr_by_resp.items()},
    }
    json_path = out_dir / "summary.json"
    with json_path.open("w") as f:
        json.dump(summary, f, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else o)
    return json_path


def _verdict(iid_abs: float) -> str:
    """Map |H4b Pearson_iid| to the STRONG/MODERATE/BORDERLINE/WEAK verdict (green-light at 0.2)."""
    if not np.isfinite(iid_abs):
        return "N/A     -- no finite IID correlation (insufficient paired samples)"
    if iid_abs >= 0.5:
        return "STRONG     -- proceed with planner as designed"
    if iid_abs >= 0.3:
        return "MODERATE   -- proceed, plan for Stretch B online refinement"
    if iid_abs >= 0.2:
        return "BORDERLINE -- meets the 0.2 green-light; proceed but confirm on the transect run"
    return "WEAK       -- proceed to the richer transect sim run before building the planner"


def print_stdout_summary(
    corr_by_resp: dict[str, dict], counters: dict, saturation_pct: float, runtime_s: float
) -> None:
    """Print the human-readable final summary. Matched features are PRIMARY; n_features auxiliary."""
    print("\n" + "=" * 72)
    print(" Phase 0.5 v2 Validation Summary (shadow-aware rho_full + matched features)")
    print("=" * 72)
    print(f"Frames input            : {counters['n_frames_input']}")
    print(f"Frames processed        : {counters['n_frames_processed']}")
    print(f"Skipped (no FrontLeft)  : {counters['n_skipped_no_image']}")
    print(f"Skipped (no FrontRight) : {counters.get('n_skipped_no_image_right', 0)}")
    print(f"OOB pose (logged)       : {counters['n_skipped_oob_pose']}")
    print(f"KP saturation @{SUPERPOINT_MAX_KP:<4d}   : {saturation_pct:.2f}%")
    if saturation_pct > SATURATION_PCT_WARN:
        print(f"  WARNING: rerun with higher SUPERPOINT_MAX_KP and rename output dir for parity.")
    if counters.get("mask_lander"):
        print(f"Lander masking          : ON  (mean {counters['mean_lander_kp_frac'] * 100:.1f}% of "
              f"keypoints/frame dropped)")
        print(f"  mean n_features raw->masked: {counters['mean_n_features_raw']:.0f} -> "
              f"{counters['mean_n_features_masked']:.0f}")
    else:
        print(f"Lander masking          : OFF (PHASE0_MASK_LANDER=0)")
    print(f"Sun azimuth             : {SUN_AZIMUTH_DEG:.3f}° (CCW from +X), alt {SUN_ALTITUDE_DEG:.3f}°")
    print()

    def _row(s: dict) -> str:
        return (
            f"Pearson_raw={s['pearson_raw']:+.3f} [{s['ci_low']:+.3f},{s['ci_high']:+.3f}]  "
            f"Pearson_iid(n={s['n_iid']:>3d})={s['pearson_iid']:+.3f}  Spearman={s['spearman']:+.3f}"
        )

    # Per-response H4b detail.
    for resp, corr in corr_by_resp.items():
        tag = "PRIMARY" if resp.startswith("n_matched") else "auxiliary"
        print(f"[{resp}]  ({tag})  H4b rho_full@lookahead -> {resp}, per distance:")
        for d in sorted(corr["per_d"].keys()):
            print(f"  d={d:>3.1f}m | {_row(corr['per_d'][d]['h4b'])}")
        bd = corr["best_d"]
        print(f"  -> best d (H4b) = {bd:.1f}m | H4a@pose: {_row(corr['h4a'])}")
        print()

    # Headline comparison table at each response's own best_d.
    print("-" * 72)
    print("H4b @ best_d comparison (|Pearson_iid| drives the verdict):")
    print(f"  {'response':<20s} {'best_d':>6s} {'Pearson_raw':>12s} {'Pearson_iid':>12s} {'Spearman':>9s}")
    for resp, corr in corr_by_resp.items():
        bd = corr["best_d"]
        s = corr["per_d"][bd]["h4b"]
        print(f"  {resp:<20s} {bd:>5.1f}m {s['pearson_raw']:>+12.3f} {s['pearson_iid']:>+12.3f} {s['spearman']:>+9.3f}")
    print()

    # Headline verdict: best of the two matched-feature variables (PRIMARY).
    matched = {r: c for r, c in corr_by_resp.items() if r.startswith("n_matched")}
    best_resp, best_iid = None, -np.inf
    for r, c in matched.items():
        v = abs(c["per_d"][c["best_d"]]["h4b"]["pearson_iid"])
        if np.isfinite(v) and v > best_iid:
            best_resp, best_iid = r, v
    print(f"HEADLINE VERDICT (matched-feature |H4b Pearson_iid|): {_verdict(best_iid)}")
    if best_resp is not None:
        print(f"  driven by {best_resp} (|Pearson_iid|={best_iid:.3f} at best_d="
              f"{matched[best_resp]['best_d']:.1f}m)")
    print()
    print(f"NOTE: dataset = {DATA_DIR.name}. Before trusting/pivoting on these numbers, visually verify:")
    print("  (1) shadow direction in dem_overlays.png (bottom-left) matches frame_examples.png;")
    print("  (2) the lander overlay in frame_examples.png covers the lander and only the lander")
    print("      (over-masking would suppress real terrain features, under-masking leaves the confound);")
    print("  (3) the raw-vs-masked gap in response_comparison.png isolates the lander's effect.")
    print()
    print(f"Total runtime: {runtime_s:.1f} s")
    print(f"Outputs: {OUT_DIR}")


# ============================================================================
# main
# ============================================================================


def main() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    _test_shadow_pole()  # fail fast if shadow geometry is broken

    print(f"Loading DEM from {DEM_PATH}")
    z, rock = load_dem(DEM_PATH)
    roughness = compute_roughness_field(z, ROUGHNESS_WINDOW_CELLS)
    rock_density = compute_rock_density_field(rock, ROUGHNESS_WINDOW_CELLS)

    print(f"Computing shadow mask (sun az={SUN_AZIMUTH_DEG:.3f}°, alt={SUN_ALTITUDE_DEG:.3f}°)")
    t_shadow = time.time()
    shadow_mask = compute_shadow_mask(z, SUN_AZIMUTH_DEG, SUN_ALTITUDE_DEG, CELL_WIDTH, SHADOW_RAY_EPS)
    print(f"  shadow mask: {shadow_mask.mean() * 100:.1f}% cells in shadow ({time.time() - t_shadow:.2f}s)")
    rho_full = compute_rho_full(roughness, shadow_mask)

    fields = DEMFields(
        z=z, rock=rock, roughness=roughness, rock_density=rock_density,
        shadow_mask=shadow_mask, rho_full=rho_full,
    )

    print(f"Loading frames from {LOG_PATH}")
    frames = load_frames(LOG_PATH)
    print(f"  {len(frames)} frames")

    print("Initializing SuperPoint + LightGlue")
    extractor = init_superpoint(SUPERPOINT_MAX_KP)
    matcher = init_matcher()

    seg = init_segmentation()
    print(f"Lander masking: {'ON (UNet LANDER class)' if seg is not None else 'OFF'}")

    print("Processing frames (SuperPoint + LightGlue temporal/stereo matches + DEM lookups)")
    df, counters = process_all_frames(frames, extractor, matcher, fields, LOOKAHEAD_DISTANCES_M, seg)

    # Saturation is a property of the raw SuperPoint extraction (before masking drops keypoints),
    # so it must be measured on n_features_raw, not the masked n_features.
    saturation_pct = 100.0 * (df["n_features_raw"] == SUPERPOINT_MAX_KP).sum() / max(len(df), 1)

    # Response variables: matched features are PRIMARY; masked n_features is the auxiliary baseline.
    # n_features_raw (pre-mask) is added afterwards so the comparison plot/summary quantify how much
    # the lander confound moved the correlation. It gets correlations but not its own plot set.
    response_vars = ("n_features", "n_matched_temporal", "n_matched_stereo")
    suffix = {"n_features": "nfeat", "n_matched_temporal": "matched_temporal", "n_matched_stereo": "matched_stereo"}

    print("Computing correlations (per response variable)")
    corr_by_resp = {
        resp: compute_all_correlations(df, LOOKAHEAD_DISTANCES_M, rng, response_col=resp)
        for resp in response_vars
    }
    if seg is not None:
        corr_by_resp["n_features_raw"] = compute_all_correlations(
            df, LOOKAHEAD_DISTANCES_M, rng, response_col="n_features_raw"
        )

    print("Generating plots")
    traj_xy = df[["x", "y"]].to_numpy()
    plot_dem_overlays(fields, traj_xy, OUT_DIR / "dem_overlays.png")  # response-independent
    for resp in response_vars:
        corr = corr_by_resp[resp]
        bd = corr["best_d"]
        sfx = suffix[resp]
        plot_hypothesis_scatter(df, corr, OUT_DIR / f"hypothesis_scatter_{sfx}.png")
        plot_r_vs_lookahead(corr, OUT_DIR / f"r_vs_lookahead_{sfx}.png")
        plot_roughness_quartile_bars(df, bd, OUT_DIR / f"roughness_quartile_bars_{sfx}.png", response_col=resp)
        plot_timeline(df, bd, OUT_DIR / f"timeline_{sfx}.png", response_col=resp)
    plot_response_comparison(corr_by_resp, OUT_DIR / "response_comparison.png")  # the money plot

    # Example frames keyed off the primary temporal-matched best_d (falls back to n_features).
    primary = corr_by_resp.get("n_matched_temporal", corr_by_resp["n_features"])
    best_d = primary["best_d"]
    example_idxs = select_example_frames(df, best_d, EXAMPLE_FRAME_COUNT, EXAMPLE_MIN_STEP_GAP)
    plot_frame_examples(df, example_idxs, extractor, fields, best_d, OUT_DIR / "frame_examples.png", seg)

    runtime = time.time() - t0
    write_summary(df, corr_by_resp, counters, saturation_pct, runtime, OUT_DIR)
    print_stdout_summary(corr_by_resp, counters, saturation_pct, runtime)


if __name__ == "__main__":
    main()
