"""
A* search base class and path-metric helpers for the LAC perception-aware planner.

Pulled from data/Example_Implementations/HW3_Final/supplemental/util.py
for Phase 1 of the project implementation plan.

Changes vs the original:
- Only AStar, path_length, path_max_slope_deg, GridCoord, and T are kept.
- All HW3-specific utilities (VO loaders, camera helpers, pose-graph optimiser,
  LoopClosureMeasurement, etc.) are dropped — they are not needed here.
- Relative import `from .dem import DEM` is updated to an absolute package import.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from heapq import heappop, heappush
from typing import Generic, Iterable, TypeVar

import numpy as np
from tqdm.auto import tqdm

from lac.planning.dem import DEM

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

#: A 2-D grid coordinate (row, col).  Used as the node type for grid-based A*.
GridCoord = tuple[int, int]

T = TypeVar("T")


# ---------------------------------------------------------------------------
# A* base class
# ---------------------------------------------------------------------------

class AStar(ABC, Generic[T]):
    """Generic A* search.

    Subclass and implement:
      - ``heuristic_cost_estimate(current, goal) -> float``
      - ``distance_between(n1, n2) -> float``
      - ``neighbors(node) -> Iterable[T]``

    Optionally override ``is_goal_reached(current, goal)`` for non-exact goals.

    Then call ``astar(start, goal)`` to get the path (list of nodes) or None if
    no path exists.
    """

    class SearchNode:
        __slots__ = ("data", "gscore", "fscore", "closed", "came_from", "out_openset")

        def __init__(self, data: T, gscore: float = float("inf"), fscore: float = float("inf")):
            self.data = data
            self.gscore = gscore
            self.fscore = fscore
            self.closed = False
            self.came_from = None
            self.out_openset = True

        def __lt__(self, other: "AStar.SearchNode"):
            return self.fscore < other.fscore

    class SearchNodeDict(dict):
        def __missing__(self, key):
            value = AStar.SearchNode(key)
            self[key] = value
            return value

    @abstractmethod
    def heuristic_cost_estimate(self, current: T, goal: T) -> float:
        raise NotImplementedError

    @abstractmethod
    def distance_between(self, n1: T, n2: T) -> float:
        raise NotImplementedError

    @abstractmethod
    def neighbors(self, node: T) -> Iterable[T]:
        raise NotImplementedError

    def is_goal_reached(self, current: T, goal: T) -> bool:
        return current == goal

    def reconstruct_path(self, last: "AStar.SearchNode") -> list[T]:
        path = []
        current = last
        while current is not None:
            path.append(current.data)
            current = current.came_from
        return list(reversed(path))

    def astar(
        self,
        start: T,
        goal: T,
        *,
        progress: bool = False,
        update_every: int = 1000,
    ) -> list[T] | None:
        """Run A* from *start* to *goal*.

        Parameters
        ----------
        start, goal : T
            Start and goal nodes.
        progress : bool
            Show a tqdm progress bar (counts expanded nodes).
        update_every : int
            tqdm update interval.

        Returns
        -------
        list[T] | None
            Ordered list of nodes from start to goal, or ``None`` if unreachable.
        """
        if self.is_goal_reached(start, goal):
            return [start]

        search_nodes = AStar.SearchNodeDict()
        start_node = search_nodes[start] = AStar.SearchNode(
            start,
            gscore=0.0,
            fscore=self.heuristic_cost_estimate(start, goal),
        )
        open_set = [start_node]
        pbar = tqdm(desc="A* planning", dynamic_ncols=True) if progress else None
        expanded = 0

        while open_set:
            current = heappop(open_set)
            expanded += 1
            if pbar is not None and expanded % update_every == 0:
                pbar.update(update_every)

            if self.is_goal_reached(current.data, goal):
                if pbar is not None:
                    pbar.update(expanded % update_every)
                    pbar.close()
                return self.reconstruct_path(current)

            current.out_openset = True
            current.closed = True

            for neighbor in map(lambda n: search_nodes[n], self.neighbors(current.data)):
                if neighbor.closed:
                    continue

                tentative_g = current.gscore + self.distance_between(current.data, neighbor.data)
                if tentative_g >= neighbor.gscore:
                    continue

                neighbor.came_from = current
                neighbor.gscore = tentative_g
                neighbor.fscore = tentative_g + self.heuristic_cost_estimate(
                    neighbor.data, goal
                )
                if neighbor.out_openset:
                    neighbor.out_openset = False
                    heappush(open_set, neighbor)
                else:
                    open_set.remove(neighbor)
                    heappush(open_set, neighbor)

        if pbar is not None:
            pbar.close()
        return None


# ---------------------------------------------------------------------------
# Path metric helpers
# ---------------------------------------------------------------------------

def path_length(path_xy: np.ndarray) -> float:
    """Total Euclidean arc length of a path.

    Parameters
    ----------
    path_xy : np.ndarray  shape (N, 2)
        Sequence of [x, y] world-frame positions.

    Returns
    -------
    float
        Sum of Euclidean segment lengths in metres.
    """
    if len(path_xy) < 2:
        return 0.0
    diffs = np.diff(path_xy, axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


def path_max_slope_deg(path_xy: np.ndarray, dem: DEM) -> float:
    """Maximum terrain slope (degrees) encountered along a path.

    Parameters
    ----------
    path_xy : np.ndarray  shape (N, 2)
        Sequence of [x, y] world-frame positions.
    dem : DEM
        The elevation model to query gradients from.

    Returns
    -------
    float
        Maximum slope angle in degrees along the path.
    """
    if len(path_xy) == 0:
        return 0.0
    gx, gy = dem.grad(path_xy[:, 0], path_xy[:, 1])
    slopes = np.degrees(np.arctan(np.hypot(gx, gy)))
    return float(np.nanmax(slopes))
