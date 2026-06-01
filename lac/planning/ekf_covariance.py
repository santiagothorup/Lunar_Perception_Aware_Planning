"""EKFCovariancePropagator: predicted localization-uncertainty score for a candidate path.

Propagates a simplified 2-D position covariance along a world-frame path. Each step dead-reckons
(adds process noise) and then gains information scaled by the perception map's predicted feature
density at that point (feature-rich terrain -> good SLAM measurement; feature-poor -> dead-reckon).
Returns the D-optimality score ``log det(Lambda_N)`` where ``Lambda = P^-1``; higher = more certain =
better localization. D-optimality is used because Carrillo et al. (2018) show it is the criterion
that stays monotonic as uncertainty grows during dead-reckoning.

This is an independent shadow filter for OFFLINE path ranking and the paper's predicted-uncertainty
metric. It is not the SLAM backend and is not part of the planner's edge cost.

Design notes (2-D, per PROJECT_TURNOVER step 7):
* State is position ``[x, y]`` only — that is the uncertainty we score; velocity/attitude add
  machinery the metric does not use.
* ``P0`` defaults to a neutral measurement-scale prior (``meas_sigma**2 I``), not the SLAM's
  lander-anchored ``EKF_INIT_R`` (0.001 m), which is so confident it would swamp the per-step
  feature-scaled measurement and make the score nearly path-independent.
* Constants come from ``lac/params.py`` (``EKF_Q_SIGMA_A``, ``EKF_R_SIGMAS``); all are
  constructor-overridable. Absolute scale is irrelevant to *ranking* — the feature-density
  dependence is.
"""
from __future__ import annotations

import numpy as np

from lac.params import CELL_WIDTH, EKF_Q_SIGMA_A, EKF_R_SIGMAS, TARGET_SPEED


class EKFCovariancePropagator:
    def __init__(
        self,
        P0: np.ndarray | None = None,
        *,
        speed: float = TARGET_SPEED,
        accel_sigma: float = EKF_Q_SIGMA_A,
        meas_sigma: float = float(EKF_R_SIGMAS[0]),
    ):
        self.speed = float(speed)
        self.accel_sigma = float(accel_sigma)
        self.meas_sigma = float(meas_sigma)
        self.P0 = np.eye(2) * (self.meas_sigma ** 2) if P0 is None else np.asarray(P0, dtype=np.float64)

    @staticmethod
    def _resample(path_xy: np.ndarray, step_size: float) -> np.ndarray:
        """Resample a polyline to ~uniform ``step_size`` spacing (so each EKF step is one increment)."""
        path = np.asarray(path_xy, dtype=np.float64)
        if len(path) < 2:
            return path
        seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        total = float(s[-1])
        if total < 1e-9:
            return path[:1]
        n = max(1, int(np.ceil(total / step_size)))
        s_new = np.linspace(0.0, total, n + 1)
        return np.column_stack([np.interp(s_new, s, path[:, 0]), np.interp(s_new, s, path[:, 1])])

    def _run(self, path_xy: np.ndarray, density_fn, step_size: float) -> np.ndarray:
        """Run the filter along the path; return the final 2x2 position covariance ``P_N``."""
        pts = self._resample(path_xy, step_size)
        dt = step_size / self.speed if self.speed > 0 else step_size
        q = (0.5 * self.accel_sigma * dt * dt) ** 2          # position process variance per step
        Q = np.eye(2) * q
        r2 = self.meas_sigma ** 2
        eye = np.eye(2)
        P = self.P0.copy()
        for x, y in pts:
            P = P + Q                                        # predict: dead-reckoning random walk
            rho = max(float(density_fn(x, y)), 0.0)          # predicted feature density in [0, 1]
            info = rho / r2                                  # measurement information, feature-scaled
            P = np.linalg.inv(np.linalg.inv(P) + info * eye)  # information-form measurement update
        return P

    def propagate_path(self, path_xy: np.ndarray, density_fn, step_size: float = CELL_WIDTH) -> float:
        """D-optimality score ``log det(Lambda_N) = -log det(P_N)``; higher = better localization."""
        P = self._run(path_xy, density_fn, step_size)
        _sign, logdet_P = np.linalg.slogdet(P)
        return float(-logdet_P)

    def get_final_covariance(
        self, path_xy: np.ndarray, density_fn, step_size: float = CELL_WIDTH
    ) -> np.ndarray:
        """Final 2x2 position covariance ``P_N`` along the path."""
        return self._run(path_xy, density_fn, step_size)
