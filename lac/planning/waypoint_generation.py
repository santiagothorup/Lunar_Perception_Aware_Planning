"""Functions for waypoint generation."""

import numpy as np

# Clockwise order: top-left, top-right, bottom-right, bottom-left
DEFAULT_ORDER = np.array([[-1, 1], [1, 1], [1, -1], [-1, -1]])


def get_starting_direction_order(initial_pose):
    signs = np.sign(initial_pose[:2, 3])
    start_index = np.argwhere((DEFAULT_ORDER == signs).all(axis=1)).flatten()[0]
    return np.roll(DEFAULT_ORDER, -start_index, axis=0), start_index


def gen_spiral(
    initial_pose: np.ndarray,
    min_val: float,
    max_val: float,
    step: float,
    repeat: int = 0,
):
    """
    Generate an Nx2 numpy array of 2D coordinates following a square spiral,
    ensuring that the last waypoint is repeated, and the next ring starts at
    the next diagonal position.

    Parameters:
      initial_pose (np.array): The initial pose of the rover.
      max_val (float): The half-side length of the outermost square.
      min_val (float): The half-side length of the innermost square.
      step (float): The decrement between successive squares.

    Returns:
      np.array: An (N x 2) numpy array containing the 2D coordinates.
    """
    points = []
    r = min_val
    direction_order, _ = get_starting_direction_order(initial_pose)

    while r <= max_val + 1e-8:
        # Compute the four corners of the current square ring
        corners = r * direction_order

        # Add corners in order
        points.extend(corners)

        # Repeat the last `repeat` waypoints
        if repeat > 0:
            points.extend(corners[:repeat])
            direction_order = np.roll(direction_order, -repeat, axis=0)

        r += step

    return np.array(points)


def gen_five_loops(initial_pose: np.ndarray, loop_width: float = 7.0, extra_closure: bool = False):
    """
    Generate an Nx2 numpy array of 2D coordinates following a flower pattern.

    Parameters:
      initial_pose (np.array): The initial pose of the rover.
      max_val (float): The half-side length of the outermost square.
      min_val (float): The half-side length of the innermost square.
      step (float): The decrement between successive squares.

    Returns:
      np.array: An (N x 2) numpy array containing the 2D coordinates.
    """
    W = loop_width / 2  # half-width of a square loop

    points = []

    # Add center loop around lander with order based on initial pose
    direction_order, start_index = get_starting_direction_order(initial_pose)
    center_loop = W * direction_order
    points.append(center_loop)

    if extra_closure:
        points.append(center_loop[:2])

    # Corner loops
    # Top-left
    petal_1 = np.array([[-W, W], [-W, 3 * W], [-3 * W, 3 * W], [-3 * W, W], [-W, W]])
    # Top-right
    petal_2 = np.array([[W, W], [3 * W, W], [3 * W, 3 * W], [W, 3 * W], [W, W]])
    # Bottom-right
    petal_3 = np.array([[W, -W], [W, -3 * W], [3 * W, -3 * W], [3 * W, -W], [W, -W]])
    # Bottom-left
    petal_4 = np.array([[-W, -W], [-3 * W, -W], [-3 * W, -3 * W], [-W, -3 * W], [-W, -W]])
    # Concatenate petals

    petals = np.array([petal_1, petal_2, petal_3, petal_4])
    shift = -start_index
    if extra_closure:
        shift -= 2
    petals = np.roll(petals, shift, axis=0)
    petals = np.concatenate(petals, axis=0)
    points.append(petals)

    if extra_closure:
        points.append(center_loop[2])

    points = np.vstack(points)
    return points


def gen_nine_loops(initial_pose: np.ndarray, loop_width: float = 7.0):
    """ """
    points = []
    W = loop_width / 2  # half-width of a square loop

    # Generate the waypoints assuming top-left start

    # Center loop
    center_loop = W * DEFAULT_ORDER
    points.append(center_loop)
    points.append(center_loop[:2])

    # Side loops
    points.append(np.array([[3 * W, W], [3 * W, -W], [W, -W], [-W, -W]]))  # Right
    points.append(np.array([[-3 * W, -W], [-3 * W, W], [-W, W], [W, W], [W, -W]]))  # Left
    points.append(np.array([[W, -3 * W], [-W, -3 * W], [-W, -W], [-W, W]]))  # Bottom
    points.append(np.array([[-W, 3 * W], [W, 3 * W], [W, W], [W, -W]]))  # Top

    # Corner loops
    # - Bottom-right
    points.append(np.array([[W, -3 * W], [3 * W, -3 * W], [3 * W, -W], [W, -W], [-W, -W]]))
    # - Bottom-left
    points.append(np.array([[-3 * W, -W], [-3 * W, -3 * W], [-W, -3 * W], [-W, -W], [-W, W]]))
    # - Top-left
    points.append(np.array([[-W, 3 * W], [-3 * W, 3 * W], [-3 * W, W], [-W, W], [W, W]]))
    # - Top-right
    points.append(np.array([[3 * W, W], [3 * W, 3 * W], [W, 3 * W], [W, W], [W, -W]]))

    points = np.vstack(points)

    # Rotate the whole trajectory based on the starting index
    _, start_index = get_starting_direction_order(initial_pose)
    R = np.array([[0, 1], [-1, 0]])  # 90-deg clockwise rotation
    for j in range(start_index):
        points = points @ R.T

    return points


def gen_triangle_loops(
    initial_pose: np.ndarray, loop_width: float = 7.0, additional_loops: bool = False
):
    W = loop_width

    points = []

    # Add center loop around lander with order based on initial pose
    direction_order, start_index = get_starting_direction_order(initial_pose)
    center_loop = (W / 2) * direction_order
    points.append(center_loop)  # [0, 1, 2, 3]
    points.append(center_loop[:2])  # [0, 1]

    # Side points for (+,+) quadrant (top-right), which is the first side to be added if start index is 0
    if additional_loops:
        side_points = W * np.array([[0, -1], [1, -1], [0, 0], [1, 0], [1, -1], [0, 0], [0, -1]])
    else:
        side_points = W * np.array([[0, -1], [1, -1], [1, 0], [0, -1]])
    R = np.array([[0, 1], [-1, 0]])  # 90-deg clockwise rotation
    # Rotate the side points based on start index
    for j in range(start_index):
        side_points = side_points @ R.T

    for i in range(4):
        # Get the corner point in the current quadrant (offset by 1)
        corner = center_loop[(i + 1) % 4]
        x, y = np.sign(corner)
        # Corner
        corner_points = W * np.array([[x, 0], [x, y], [0, 0], [0, y], [x, y], [0, 0]])
        points.append(corner + corner_points)

        # Side
        points.append(corner + side_points)
        side_points = side_points @ R.T

    points = np.vstack(points)
    return points


def gen_phase0_transect(
    initial_pose: np.ndarray,
    sweep_xs: tuple = (-10.5, -7.5, -4.5, -2.5, 2.5, 4.5, 7.5, 10.5),
    y_extent: float = 11.0,
    lander_keepout: float = 2.5,
    waypoint_spacing: float = 4.0,
    peak_probe_x: float | None = 0.0,
):
    """Serpentine N-S raster for Phase 0 data collection.

    Drives the rover across the whole map so its look-ahead samples the full range of DEM
    roughness, sun-shadow, and rock density (the existing preset-2 run only covered a 7x15 m box,
    starving the ρ_full correlation of signal). Returns an (N, 2) array of world-frame [x, y]
    waypoints, like the other generators here.

    Design constraints (verified against the sim/leaderboard):
      - All `sweep_xs` keep |x| >= `lander_keepout`, so full-height sweeps never hit the 3x3 m
        lander at the origin. (The central column is covered by the peak probe instead.)
      - Waypoints are densified to <= `waypoint_spacing` m apart so no leg exceeds WAYPOINT_TIMEOUT
        (2000 steps = 100 s at 0.2 m/s -> 20 m), which keeps a momentarily-stuck rover from
        tripping the 300 s "blocked" termination.
      - `initial_pose` only selects the nearest starting x-extreme and y-end, minimizing the
        initial transit. No quadrant rotation (which would crash for a start on the x or y axis).

    The peak probe is a collision-free excursion north of the lander (y >= `lander_keepout`) that
    ends with a northward ascent at `peak_probe_x`, putting the ~3 m look-ahead on the dominant
    preset-2 roughness peak at (0.22, 5.18). Set `peak_probe_x=None` to disable.
    """
    sweep_xs = np.asarray(sweep_xs, dtype=float)
    if np.any(np.abs(sweep_xs) < lander_keepout):
        raise ValueError(
            f"sweep_xs must all satisfy |x| >= lander_keepout ({lander_keepout} m) to clear the "
            "lander at the origin; use peak_probe_x for central coverage instead."
        )

    start_x, start_y = float(initial_pose[0, 3]), float(initial_pose[1, 3])

    # Serpentine must begin at an x-extreme; pick the one nearest the rover.
    xs = np.sort(sweep_xs)
    if abs(start_x - xs[0]) > abs(start_x - xs[-1]):
        xs = xs[::-1]
    # First sweep heads from the y-end nearest the rover toward the far end.
    a, b = (-y_extent, y_extent) if start_y <= 0 else (y_extent, -y_extent)

    vertices = []
    for x in xs:
        vertices.append((x, a))
        vertices.append((x, b))
        a, b = b, a  # serpentine: reverse the next sweep so connectors stay short

    if peak_probe_x is not None:
        x_last = vertices[-1][0]
        vertices.append((x_last, lander_keepout))      # up the last column to north of the lander
        vertices.append((peak_probe_x, lander_keepout))  # west along y = keepout (clears the lander)
        vertices.append((peak_probe_x, y_extent))      # northward ascent -> look-ahead hits the peak

    # Densify each leg to <= waypoint_spacing.
    waypoints = [np.asarray(vertices[0], dtype=float)]
    for p0, p1 in zip(vertices[:-1], vertices[1:]):
        p0, p1 = np.asarray(p0, dtype=float), np.asarray(p1, dtype=float)
        dist = np.linalg.norm(p1 - p0)
        if dist < 1e-9:
            continue
        n = int(np.ceil(dist / waypoint_spacing))
        for k in range(1, n + 1):
            waypoints.append(p0 + (p1 - p0) * (k / n))
    return np.array(waypoints)


def gen_phase0_probe(initial_pose: np.ndarray, drive_dist: float = 3.0):
    """Minimal trajectory whose only purpose is to end the mission quickly so the leaderboard
    writes the preset's ground-truth DEM (statistics_manager saves the sim terrain map at mission
    end -> results/Moon_Map_01_<mission_id>_rep0.dat). Used to scan presets for terrain richness
    before committing a full gen_phase0_transect run; NOT for feature data collection.

    Returns a single waypoint `drive_dist` m radially OUTWARD from the start (away from the lander at
    the origin), so the short drive can't collide with the 3x3 m lander. The rover reaches it in
    ~15 s; WAYPOINT_TIMEOUT (100 s) is the safety net if a preset's spawn happens to be obstructed.
    """
    p = np.asarray(initial_pose[:2, 3], dtype=float)
    r = float(np.linalg.norm(p))
    direction = p / r if r > 1e-6 else np.array([1.0, 0.0])  # outward = away from lander/origin
    return np.array([p + drive_dist * direction])


def gen_loops_lander_lc(initial_pose):
    """
    Same as 5 loops above, but with stop and turn to loop at the lander at each corner.
    """
    pass
