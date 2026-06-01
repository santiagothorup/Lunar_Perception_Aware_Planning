"""PerceptionMap: DEM-derived visual-feature-density predictor for the perception-aware planner.

Wraps a preset's ground-truth heightmap + sun angle into a normalized feature-density field
``rho_norm`` in [0, 1] and an uncertainty cost ``1 - rho_norm``. The planner uses the latter as the
perception term in its A* edge cost, biasing paths toward feature-rich terrain where SLAM localizes
well.

The field math (``compute_roughness_field`` / ``compute_shadow_mask`` / ``compute_rho_full``) is the
exact code validated in Phase 0 (``scripts/phase0_validation.py`` imports it from here), where
``rho_full = roughness * (1 - shadow_mask)`` was shown to correlate with SLAM-usable feature density.
Two things that matter:

* **Orientation.** All grids use the LAC-native ``(axis0 = x, axis1 = y)`` layout — the orientation
  the validation used and the one ``compute_shadow_mask``'s sun-direction math assumes. The DEM class
  in ``lac/planning/dem.py`` uses the opposite ``(rows = y, cols = x)`` layout; the two never index
  each other — callers query :meth:`PerceptionMap.get_feature_density` with **world** ``(x, y)``.
* **Validated formula only.** ``rho_full = roughness * (1 - shadow_mask)``. The ``sun_factor`` term
  from the original design is dropped (constant at this sun altitude) and rocks are excluded
  (uncorrelated with features in Phase 0). Do not re-add them.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.ndimage import maximum_filter, uniform_filter

from lac.params import CELL_WIDTH, MAP_EXTENT, MAP_SIZE

# Phase-0 sun, fixed for every preset (leaderboard/missionmanager/mission_weather.py hardcodes
# lat=-90, lon=0, date 2023-01-15). Only the DEM changes per preset.
DEFAULT_SUN_AZIMUTH_DEG = 263.575  # CCW from world +X
DEFAULT_SUN_ALTITUDE_DEG = 1.488
SHADOW_RAY_EPS = 1e-3  # vertical lift (m) so a flat plane doesn't self-shadow
ROUGHNESS_WINDOW_CELLS = 5


# ============================================================================
# Validated field functions (canonical home; imported by scripts/phase0_validation.py)
# ============================================================================


def load_lac_dem(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a LAC ``.dat`` DEM, returning ``(height_z, rock_bool)`` as ``(MAP_SIZE, MAP_SIZE)``.

    The file is a ``(MAP_SIZE, MAP_SIZE, 4)`` array of ``[x, y, z, rock_bool]`` with axis 0 = x and
    axis 1 = y (per ``lac/mapping/mapper.py``). Both returned grids keep that ``(x, y)`` orientation.
    """
    arr = np.load(Path(path), allow_pickle=True)
    if arr.shape != (MAP_SIZE, MAP_SIZE, 4):
        raise ValueError(f"unexpected DEM shape {arr.shape}, want ({MAP_SIZE},{MAP_SIZE},4)")
    return arr[:, :, 2].astype(np.float64), arr[:, :, 3].astype(np.float64)


def compute_roughness_field(z: np.ndarray, window: int) -> np.ndarray:
    """Per-cell std of heights in a ``window x window`` neighborhood.

    Uses the variance-via-uniform-filter trick (~500x faster than ``generic_filter``).
    Returns NaN at any cell whose window touches a non-finite input.
    """
    valid = np.isfinite(z)
    z_clean = np.where(valid, z, 0.0)
    mean = uniform_filter(z_clean, size=window, mode="nearest")
    mean_sq = uniform_filter(z_clean * z_clean, size=window, mode="nearest")
    var = np.clip(mean_sq - mean * mean, 0.0, None)  # clip away tiny negative float errors
    rough = np.sqrt(var)
    # Propagate NaN if any window cell was invalid.
    valid_count = uniform_filter(valid.astype(np.float64), size=window, mode="nearest")
    rough[valid_count < 1.0 - 1e-9] = np.nan
    return rough


def compute_rock_density_field(rock: np.ndarray, window: int) -> np.ndarray:
    """Window-mean of the binary rock indicator -> continuous density in [0, 1].

    Not used by ``PerceptionMap`` (rocks were uncorrelated with features in Phase 0); kept here as
    the shared home for the validation script's rock-hypothesis check.
    """
    return uniform_filter(rock, size=window, mode="nearest")


def compute_shadow_mask(
    z: np.ndarray, sun_az_deg: float, sun_alt_deg: float, cell_width: float, eps: float
) -> np.ndarray:
    """Cast a ray from each cell toward the sun; return True where blocked.

    Convention: ``sun_az_deg`` is CCW from world +X in the XY plane; the grid axis 0 is x, axis 1 is
    y. The ray rises by ``cell_width * tan(alt) * hypot(di, dj)`` m per DDA step (one cell along the
    dominant horizontal axis). ``eps`` lifts the ray off the source surface so a flat plane doesn't
    self-shadow.
    """
    az = np.radians(sun_az_deg)
    alt = np.radians(sun_alt_deg)
    if alt <= 0:
        return np.ones_like(z, dtype=bool)
    sx = float(np.cos(alt) * np.cos(az))
    sy = float(np.cos(alt) * np.sin(az))
    n = max(abs(sx), abs(sy))
    if n < 1e-9:
        return np.zeros_like(z, dtype=bool)  # sun directly overhead
    di, dj = sx / n, sy / n
    dh = cell_width * float(np.hypot(di, dj))
    dz_step = dh * float(np.tan(alt))

    H, W = z.shape
    k_geom = int(np.ceil(np.sqrt(2.0) * max(H, W))) + 5
    if dz_step > 0:
        k_height = int(np.ceil((float(np.nanmax(z)) - float(np.nanmin(z)) + 0.5) / dz_step))
        k_max = min(k_geom, k_height)
    else:
        k_max = k_geom

    shadow = np.zeros_like(z, dtype=bool)
    z_src = z + eps
    ii, jj = np.indices(z.shape)

    for k in range(1, k_max + 1):
        oi = int(round(k * di))
        oj = int(round(k * dj))
        si = ii + oi
        sj = jj + oj
        in_bounds = (si >= 0) & (si < H) & (sj >= 0) & (sj < W)
        if not in_bounds.any():
            break
        terrain = np.full_like(z, -np.inf)
        terrain[in_bounds] = z[si[in_bounds], sj[in_bounds]]
        shadow |= terrain > (z_src + k * dz_step)
    return shadow


def compute_rho_full(roughness: np.ndarray, shadow_mask: np.ndarray) -> np.ndarray:
    """rho_full = roughness * (1 - shadow_mask). NaN propagates from ``roughness``."""
    return roughness * (1.0 - shadow_mask.astype(np.float64))


def world_to_grid(x: float, y: float) -> tuple[int, int] | None:
    """World ``(x, y)`` metres -> grid ``(i = x-index, j = y-index)``, or None if out of map."""
    if abs(x) > MAP_EXTENT or abs(y) > MAP_EXTENT:
        return None
    i = min(MAP_SIZE - 1, max(0, int((x + MAP_EXTENT) / CELL_WIDTH)))
    j = min(MAP_SIZE - 1, max(0, int((y + MAP_EXTENT) / CELL_WIDTH)))
    return i, j


def grid_to_world(i: int, j: int) -> tuple[float, float]:
    """Inverse of :func:`world_to_grid` at cell centers: ``(i, j) -> (x, y)`` in metres."""
    x = (i + 0.5) * CELL_WIDTH - MAP_EXTENT
    y = (j + 0.5) * CELL_WIDTH - MAP_EXTENT
    return float(x), float(y)


def _test_shadow_pole() -> None:
    """Self-test: a 1 m pole on a flat plane casts a long shadow in the anti-sun direction.

    At sun_az=180 deg, sun_alt=1 deg the sun is in -x with a near-horizontal ray, so cells with
    i > pole_i look toward the pole (in -x) and ARE shadowed; cells with i < pole_i are NOT.
    Shadow length ~ 1/tan(1 deg) ~ 57 m ~ 381 cells at CELL_WIDTH=0.15.
    """
    z = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.float64)
    pi, pj = 90, 90
    z[pi, pj] = 1.0
    mask = compute_shadow_mask(z, sun_az_deg=180.0, sun_alt_deg=1.0, cell_width=0.15, eps=1e-3)
    assert mask[pi + 1, pj], "cell directly +x of pole should be shadowed"
    assert mask[pi + 80, pj], "cell +80 in shadow column should be shadowed (within 381-cell shadow)"
    assert not mask[pi - 1, pj], "sun-facing side cell must not be shadowed"
    assert not mask[pi, pj], "pole's own cell must not be shadowed (eps lifts ray)"
    assert not mask[pi + 1, pj + 5], "off-axis cell must not be shadowed for purely-x sun"
    print("[ok] shadow self-test passed")


# ============================================================================
# PerceptionMap
# ============================================================================


class PerceptionMap:
    """Normalized feature-density map rho(x, y) in [0, 1] from a heightmap + sun angle.

    Fields (all ``(MAP_SIZE, MAP_SIZE)``, ``(axis0 = x, axis1 = y)`` orientation), computed once at
    construction:
      - ``roughness``   : per-cell height std (window).
      - ``shadow_mask`` : True where the cell is sun-shadowed.
      - ``rho_full``    : raw validated predictor ``roughness * (1 - shadow_mask)`` (NaN where invalid).
      - ``rho_norm``    : ``rho_full`` scaled to [0, 1] by a per-map percentile (NaN/out-of-map -> 0).
    """

    def __init__(
        self,
        z_grid: np.ndarray,
        *,
        rock_grid: np.ndarray | None = None,
        cell_width: float = CELL_WIDTH,
        sun_azimuth_deg: float = DEFAULT_SUN_AZIMUTH_DEG,
        sun_altitude_deg: float = DEFAULT_SUN_ALTITUDE_DEG,
        roughness_window: int = ROUGHNESS_WINDOW_CELLS,
        rho_scale_pct: float = 95.0,
    ):
        self.z = np.asarray(z_grid, dtype=np.float64)  # (MAP_SIZE, MAP_SIZE), (x, y)
        # Optional rock indicator (MAP_SIZE, MAP_SIZE) in (x, y) orientation, used by v1b anchor
        # extraction's rock-density filter. None -> filter is skipped.
        self.rock_grid = (
            np.asarray(rock_grid, dtype=np.float64) if rock_grid is not None else None
        )
        # v1b: populated lazily by compute_anchors(); planner reads self.anchors_xy directly.
        self.anchors_xy: np.ndarray | None = None
        self.anchors_score: np.ndarray | None = None
        self.cell_width = cell_width
        self.sun_azimuth_deg = sun_azimuth_deg
        self.sun_altitude_deg = sun_altitude_deg

        self.roughness = compute_roughness_field(self.z, roughness_window)
        self.shadow_mask = compute_shadow_mask(
            self.z, sun_azimuth_deg, sun_altitude_deg, cell_width, SHADOW_RAY_EPS
        )
        self.rho_full = compute_rho_full(self.roughness, self.shadow_mask)

        # Normalize to [0, 1] for the planner's (1 - rho) cost term. A per-map high percentile is
        # robust to rare roughness spikes that a plain max would let dominate. NaN / out-of-map -> 0
        # (treated as maximally uncertain so the planner avoids unknown terrain).
        finite = self.rho_full[np.isfinite(self.rho_full)]
        scale = float(np.percentile(finite, rho_scale_pct)) if finite.size else 1.0
        self.rho_scale = scale if scale > 1e-12 else 1.0
        self.rho_norm = np.clip(np.nan_to_num(self.rho_full / self.rho_scale, nan=0.0), 0.0, 1.0)

        # Normalized roughness for the planner's traversability penalty (high = rough/rocky = avoid).
        # Same per-map percentile scaling as rho, but NaN/invalid -> 1.0 (max penalty: steer clear of
        # unknown/edge terrain rather than treating it as smooth).
        rfinite = self.roughness[np.isfinite(self.roughness)]
        rscale = float(np.percentile(rfinite, rho_scale_pct)) if rfinite.size else 1.0
        self.roughness_scale = rscale if rscale > 1e-12 else 1.0
        self.roughness_norm = np.clip(
            np.nan_to_num(self.roughness / self.roughness_scale, nan=1.0), 0.0, 1.0
        )

    @classmethod
    def from_lac_dat(
        cls,
        path: str | Path,
        *,
        sun_azimuth_deg: float = DEFAULT_SUN_AZIMUTH_DEG,
        sun_altitude_deg: float = DEFAULT_SUN_ALTITUDE_DEG,
        roughness_window: int = ROUGHNESS_WINDOW_CELLS,
        rho_scale_pct: float = 95.0,
    ) -> "PerceptionMap":
        """Build from a LAC ``.dat`` ground-truth DEM (the primary constructor)."""
        z, rock = load_lac_dem(path)
        return cls(
            z,
            rock_grid=rock,
            sun_azimuth_deg=sun_azimuth_deg,
            sun_altitude_deg=sun_altitude_deg,
            roughness_window=roughness_window,
            rho_scale_pct=rho_scale_pct,
        )

    def get_density_grid(self) -> np.ndarray:
        """The full normalized rho grid in [0, 1], ``(MAP_SIZE, MAP_SIZE)`` ``(x, y)`` orientation."""
        return self.rho_norm

    def get_feature_density(self, x: float, y: float) -> float:
        """Normalized rho at world ``(x, y)`` (nearest cell). 0.0 if out of map."""
        ij = world_to_grid(x, y)
        if ij is None:
            return 0.0
        return float(self.rho_norm[ij])

    def get_uncertainty_cost(self, x: float, y: float) -> float:
        """``1 - rho`` at world ``(x, y)`` in [0, 1] — high where features are sparse."""
        return 1.0 - self.get_feature_density(x, y)

    def get_traversability_cost(self, x: float, y: float) -> float:
        """Normalized roughness at world ``(x, y)`` in [0, 1] -- 0 = smooth, 1 = rough/rocky.

        The planner's traversability penalty: routing onto rough/rocky cells is discouraged (they
        stall the local ArcPlanner and degrade SLAM). Out-of-map -> 1.0 (max penalty).
        """
        ij = world_to_grid(x, y)
        return 1.0 if ij is None else float(self.roughness_norm[ij])

    def get_shadow_value(self, x: float, y: float) -> float:
        """Binary shadow at world ``(x, y)``: 1.0 = in shadow, 0.0 = sunlit. OOB -> 1.0.

        The planner's shadow-avoidance penalty: cells in shadow are unobservable to the camera
        (no features). Pairs with shadow_weight in the A* edge cost to route the rover through
        sunlit terrain -- the simplest principled extension of slope-only A* with perception intent
        (no roughness reward to push the rover into rocky cells).
        """
        ij = world_to_grid(x, y)
        return 1.0 if ij is None else float(self.shadow_mask[ij])

    def get_lookahead_density(self, x: float, y: float, heading_xy, distances) -> float:
        """Mean rho the camera SEES looking from ``(x, y)`` along ``heading_xy``, sampled at ``distances``.

        This is the planner's perception reward: it credits a position for the feature density
        VISIBLE ahead of it -- matching Phase 0's look-ahead predictor (pose-rho anti-correlates with
        features; only look-ahead rho correlates) -- rather than the roughness underfoot. So the
        planner routes the rover PAST feature-rich/rocky terrain instead of driving ONTO it.
        Out-of-map look-ahead points contribute 0.0 (a camera looking off-map sees no features).
        Falls back to the at-position density if the heading is ~zero.
        """
        hx, hy = float(heading_xy[0]), float(heading_xy[1])
        norm = np.hypot(hx, hy)
        if norm < 1e-9:
            return self.get_feature_density(x, y)
        hx, hy = hx / norm, hy / norm
        return float(np.mean([self.get_feature_density(x + d * hx, y + d * hy) for d in distances]))

    def compute_anchors(
        self,
        dem,
        *,
        min_separation_m: float = 4.0,
        max_count: int = 12,
        slope_max_deg: float = 20.0,
        min_rho: float = 0.4,
        rock_density_max: float = 0.05,
        rock_window_m: float = 1.0,
        roughness_max: float = 1.0,
        local_max_footprint: int = 5,
    ) -> np.ndarray:
        """Extract high-rho "anchor" hubs for the loop-closure-aware planner (v1b).

        Returns an ``(M, 2)`` array of world ``(x, y)`` anchor positions and stores it on
        ``self.anchors_xy`` (plus ``self.anchors_score`` = ``rho_norm`` at each). Anchors are local
        maxima of ``rho_norm`` filtered by slope (DEM-gradient-based, matching the planner's
        ``_cell_slope_deg``), rho threshold, optional rock density, then greedy non-max-suppression.

        Orientation note: ``rho_norm`` is ``(axis0=x, axis1=y)``; DEM gx/gy are ``(rows=y, cols=x)``.
        Anchor ``(i, j)`` indices are converted to world via :func:`grid_to_world`, then to DEM
        ``(r, c)`` via :meth:`DEM.xy_to_rc` for the slope lookup -- the two grids never index each other.
        """
        # 1) local maxima of rho_norm.
        peak_mask = self.rho_norm == maximum_filter(
            self.rho_norm, size=local_max_footprint, mode="nearest"
        )
        peak_mask &= self.rho_norm >= float(min_rho)

        ii, jj = np.where(peak_mask)
        if ii.size == 0:
            self.anchors_xy = np.empty((0, 2), dtype=np.float64)
            self.anchors_score = np.empty((0,), dtype=np.float64)
            return self.anchors_xy

        # 2) world (x, y) for each candidate; 3) slope filter; 4) rock-density filter;
        # 5) (v1e) roughness ceiling -- anchor cell itself must be SMOOTH even if it overlooks
        # feature-rich terrain (matches what made the perc_trav cost stay mobile).
        kept_xy: list[tuple[float, float]] = []
        kept_score: list[float] = []
        order = np.argsort(self.rho_norm[ii, jj])[::-1]  # descending rho
        rock_radius_cells = max(1, int(np.ceil(float(rock_window_m) / self.cell_width)))
        for k in order:
            i, j = int(ii[k]), int(jj[k])
            x, y = grid_to_world(i, j)
            # slope via DEM gradients (same form as PerceptionAwarePlanner._cell_slope_deg).
            r, c = dem.xy_to_rc(x, y)
            if not dem.rc_in_bounds(r, c):
                continue
            gx, gy = float(dem.gx[r, c]), float(dem.gy[r, c])
            if not (np.isfinite(gx) and np.isfinite(gy)):
                continue
            slope_deg = float(np.degrees(np.arctan(np.hypot(gx, gy))))
            if slope_deg > slope_max_deg:
                continue
            # v1e: anchor cell itself must be smooth (roughness_norm low). Skips the "rho peak ==
            # rocky cell" trap that broke v1b/v1c.
            if float(self.roughness_norm[i, j]) > roughness_max:
                continue
            # rock-density filter (skip if no rock grid).
            if self.rock_grid is not None:
                i0, i1 = max(0, i - rock_radius_cells), min(MAP_SIZE, i + rock_radius_cells + 1)
                j0, j1 = max(0, j - rock_radius_cells), min(MAP_SIZE, j + rock_radius_cells + 1)
                local_rock = float(self.rock_grid[i0:i1, j0:j1].mean())
                if local_rock > rock_density_max:
                    continue
            kept_xy.append((x, y))
            kept_score.append(float(self.rho_norm[i, j]))

        if not kept_xy:
            self.anchors_xy = np.empty((0, 2), dtype=np.float64)
            self.anchors_score = np.empty((0,), dtype=np.float64)
            return self.anchors_xy

        # 5) greedy non-max-suppression at min_separation_m. kept_xy is already sorted by rho desc.
        chosen_xy: list[tuple[float, float]] = []
        chosen_score: list[float] = []
        for xy, sc in zip(kept_xy, kept_score):
            if any(
                (xy[0] - cx) ** 2 + (xy[1] - cy) ** 2 < min_separation_m ** 2
                for cx, cy in chosen_xy
            ):
                continue
            chosen_xy.append(xy)
            chosen_score.append(sc)
            if len(chosen_xy) >= max_count:
                break

        self.anchors_xy = np.asarray(chosen_xy, dtype=np.float64).reshape(-1, 2)
        self.anchors_score = np.asarray(chosen_score, dtype=np.float64)
        return self.anchors_xy

    def visualize(self, save_path: str | Path | None = None):
        """Plot elevation, roughness, shadow mask, and normalized rho side by side."""
        import matplotlib.pyplot as plt  # lazy: keep import-time deps light for the planner path

        extent = [-MAP_EXTENT, MAP_EXTENT, -MAP_EXTENT, MAP_EXTENT]
        panels = [
            (self.z, "terrain", "elevation z (m)"),
            (self.roughness, "viridis", "roughness (m)"),
            (self.shadow_mask.astype(float), "gray", "shadow (1 = shadowed)"),
            (self.rho_norm, "magma", "rho_norm in [0, 1]"),
        ]
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        for ax, (field, cmap, title) in zip(axes, panels):
            # Transpose so x is horizontal and y vertical with origin at lower-left.
            im = ax.imshow(field.T, origin="lower", extent=extent, cmap=cmap)
            ax.set_title(title)
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle(
            f"PerceptionMap (sun az={self.sun_azimuth_deg:.1f} deg, alt={self.sun_altitude_deg:.2f} deg)"
        )
        fig.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
        return fig
