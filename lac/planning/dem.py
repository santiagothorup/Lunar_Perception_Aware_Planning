"""
DEM (Digital Elevation Model) class for the LAC perception-aware planner.

Pulled from data/Example_Implementations/HW3_Final/supplemental/dem.py
for Phase 1 of the project implementation plan. No logic changes — only
the `surface_plot()` method is removed (it depended on a supplemental
plotting module that is not available here).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator


@dataclass
class DEM:
    """Digital Elevation Model with interpolation and gradient support.

    Attributes
    ----------
    x_axis : np.ndarray  shape (W,)  — x coordinates (columns), must be sorted ascending
    y_axis : np.ndarray  shape (H,)  — y coordinates (rows),    must be sorted ascending
    z_grid : np.ndarray  shape (H, W) — elevation values in metres
    metadata : dict | None           — optional provenance metadata
    """

    x_axis: np.ndarray
    y_axis: np.ndarray
    z_grid: np.ndarray
    metadata: dict | None = None

    def __post_init__(self):
        self.x_axis = np.asarray(self.x_axis, dtype=np.float64)
        self.y_axis = np.asarray(self.y_axis, dtype=np.float64)
        self.z_grid = np.asarray(self.z_grid, dtype=np.float64)
        if self.z_grid.shape != (len(self.y_axis), len(self.x_axis)):
            raise ValueError(
                f"Expected z_grid shape {(len(self.y_axis), len(self.x_axis))}, "
                f"got {self.z_grid.shape}"
            )
        self.res_x = float(np.mean(np.diff(self.x_axis)))
        self.res_y = float(np.mean(np.diff(self.y_axis)))
        self.gy, self.gx = np.gradient(self.z_grid, self.y_axis, self.x_axis, edge_order=2)
        self._interp_z = RegularGridInterpolator(
            (self.y_axis, self.x_axis),
            self.z_grid,
            bounds_error=False,
            fill_value=np.nan,
            method="linear",
        )
        self._interp_gx = RegularGridInterpolator(
            (self.y_axis, self.x_axis),
            self.gx,
            bounds_error=False,
            fill_value=np.nan,
            method="linear",
        )
        self._interp_gy = RegularGridInterpolator(
            (self.y_axis, self.x_axis),
            self.gy,
            bounds_error=False,
            fill_value=np.nan,
            method="linear",
        )

    # ------------------------------------------------------------------ #
    #  Constructors / serialisation                                        #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_npz(cls, path: str | Path) -> "DEM":
        """Load a DEM saved with `to_npz()`."""
        data = np.load(Path(path), allow_pickle=True)
        metadata = {}
        if "metadata" in data:
            raw = data["metadata"]
            if isinstance(raw, np.ndarray) and raw.shape == ():
                metadata = raw.item()
            else:
                metadata = dict(raw)
        return cls(data["x_axis"], data["y_axis"], data["z_grid"], metadata=metadata)

    @classmethod
    def from_lac_dat(cls, path: str | Path) -> "DEM":
        """Load a DEM from the LAC .dat format (shape (H, W, 4): [x, y, z, rock_bool]).

        The LAC map array has first axis = x and second axis = y (confirmed by
        bin_points_to_grid in lac/mapping/mapper.py). This loader transposes
        appropriately so DEM follows the standard (rows=y, cols=x) convention.
        """
        arr = np.load(Path(path), allow_pickle=True)  # shape (MAP_SIZE, MAP_SIZE, 4)
        # arr[i, j] = [x_center, y_center, z, rock_bool]
        # First axis → x, second axis → y  →  transpose to (y, x)
        x_coords = arr[:, 0, 0]   # unique x values along first axis
        y_coords = arr[0, :, 1]   # unique y values along second axis
        z_grid   = arr[:, :, 2].T  # shape (H=y, W=x) after transpose
        return cls(x_coords, y_coords, z_grid, metadata={"source": str(path)})

    def to_npz(self, path: str | Path):
        """Save the DEM to a compressed .npz file."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            save_path,
            x_axis=self.x_axis,
            y_axis=self.y_axis,
            z_grid=self.z_grid,
            metadata=np.array(self.metadata or {}, dtype=object),
        )

    # ------------------------------------------------------------------ #
    #  Properties                                                          #
    # ------------------------------------------------------------------ #

    @property
    def shape(self) -> tuple[int, int]:
        """(n_rows, n_cols) == (len(y_axis), len(x_axis))."""
        return self.z_grid.shape

    # ------------------------------------------------------------------ #
    #  Bounds checks                                                       #
    # ------------------------------------------------------------------ #

    def xy_in_bounds(self, x: float, y: float) -> bool:
        return (self.x_axis[0] <= x <= self.x_axis[-1]) and (
            self.y_axis[0] <= y <= self.y_axis[-1]
        )

    def rc_in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < len(self.y_axis) and 0 <= c < len(self.x_axis)

    # ------------------------------------------------------------------ #
    #  Queries                                                             #
    # ------------------------------------------------------------------ #

    def query(self, x, y) -> float | np.ndarray:
        """Bilinearly interpolate elevation at world position (x, y).

        Accepts scalars or arrays.  Returns NaN for out-of-bounds queries.
        """
        pts = np.column_stack([np.asarray(y).reshape(-1), np.asarray(x).reshape(-1)])
        z = self._interp_z(pts)
        if np.isscalar(x) and np.isscalar(y):
            return float(z[0])
        return z.reshape(np.shape(x))

    def grad(self, x, y) -> tuple:
        """Return (dz/dx, dz/dy) at world position (x, y) via bilinear interpolation."""
        pts = np.column_stack([np.asarray(y).reshape(-1), np.asarray(x).reshape(-1)])
        gx = self._interp_gx(pts)
        gy = self._interp_gy(pts)
        if np.isscalar(x) and np.isscalar(y):
            return float(gx[0]), float(gy[0])
        return gx.reshape(np.shape(x)), gy.reshape(np.shape(x))

    def xy_to_rc(self, x: float, y: float) -> tuple[int, int]:
        """Nearest-neighbour world → grid index conversion."""
        c = int(np.argmin(np.abs(self.x_axis - x)))
        r = int(np.argmin(np.abs(self.y_axis - y)))
        return r, c

    def rc_to_xy(self, r: int, c: int) -> tuple[float, float]:
        """Grid index → world coordinate conversion."""
        return float(self.x_axis[c]), float(self.y_axis[r])

    # ------------------------------------------------------------------ #
    #  Derived grids                                                       #
    # ------------------------------------------------------------------ #

    def slope_deg_grid(self) -> np.ndarray:
        """Return (H, W) array of terrain slope in degrees."""
        return np.degrees(np.arctan(np.hypot(self.gx, self.gy)))

    # ------------------------------------------------------------------ #
    #  Resampling                                                          #
    # ------------------------------------------------------------------ #

    def downsample(self, factor: int = 2, method: str = "average") -> "DEM":
        """Return a coarser DEM by block-averaging."""
        if factor < 1 or int(factor) != factor:
            raise ValueError("factor must be a positive integer")
        factor = int(factor)
        if factor == 1:
            meta = dict(self.metadata or {})
            meta.update({"downsample_factor": 1, "downsample_method": method})
            return DEM(self.x_axis.copy(), self.y_axis.copy(), self.z_grid.copy(), metadata=meta)
        if method != "average":
            raise ValueError(f"Unsupported downsample method: {method!r}")

        rows = (len(self.y_axis) // factor) * factor
        cols = (len(self.x_axis) // factor) * factor
        if rows == 0 or cols == 0:
            raise ValueError("factor is too large for the DEM dimensions")

        x_trim = self.x_axis[:cols]
        y_trim = self.y_axis[:rows]
        z_trim = self.z_grid[:rows, :cols]

        z_blocks = z_trim.reshape(rows // factor, factor, cols // factor, factor)
        z_down   = z_blocks.mean(axis=(1, 3))
        x_down   = x_trim.reshape(cols // factor, factor).mean(axis=1)
        y_down   = y_trim.reshape(rows // factor, factor).mean(axis=1)

        meta = dict(self.metadata or {})
        meta.update(
            {
                "downsample_factor": factor,
                "downsample_method": method,
                "source_shape": tuple(int(v) for v in self.z_grid.shape),
            }
        )
        return DEM(x_down, y_down, z_down, metadata=meta)
