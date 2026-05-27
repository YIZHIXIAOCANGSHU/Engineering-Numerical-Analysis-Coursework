from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline

from arm_planning.core.math_utils import joint_path_length
from arm_planning.sim.mujoco_world import MujocoWorld


def shortcut_smooth(path: list[np.ndarray], context: MujocoWorld, iterations: int, rng: np.random.Generator) -> list[np.ndarray]:
    if len(path) <= 2:
        return [p.copy() for p in path]
    smoothed = [p.copy() for p in path]
    for _ in range(int(iterations)):
        if len(smoothed) <= 2:
            break
        i, j = sorted(rng.choice(len(smoothed), size=2, replace=False).tolist())
        if j <= i + 1:
            continue
        if context.is_edge_valid(smoothed[i], smoothed[j]):
            smoothed = smoothed[: i + 1] + smoothed[j:]
    return smoothed


def cubic_resample(path: list[np.ndarray], samples: int = 80) -> list[np.ndarray]:
    if len(path) == 0:
        return []
    if len(path) == 1:
        return [path[0].copy()]
    samples = max(int(samples), len(path))
    arr = np.asarray(path, dtype=float)
    distances = np.zeros(len(arr), dtype=float)
    for i in range(1, len(arr)):
        distances[i] = distances[i - 1] + float(np.linalg.norm(arr[i] - arr[i - 1]))
    if distances[-1] <= 1e-12:
        return [arr[0].copy() for _ in range(samples)]
    t = distances / distances[-1]
    # Remove duplicate support points after aggressive shortcutting.
    keep = np.r_[True, np.diff(t) > 1e-9]
    t = t[keep]
    arr = arr[keep]
    if len(arr) < 3:
        new_t = np.linspace(0.0, 1.0, samples)
        return [arr[0] + (arr[-1] - arr[0]) * alpha for alpha in new_t]
    spline = CubicSpline(t, arr, axis=0, bc_type="natural")
    return [q.copy() for q in spline(np.linspace(0.0, 1.0, samples))]


def process_path(path: list[np.ndarray], context: MujocoWorld, iterations: int, samples: int, rng: np.random.Generator) -> list[np.ndarray]:
    smoothed = shortcut_smooth(path, context, iterations, rng)
    resampled = cubic_resample(smoothed, samples)
    # Cubic splines can overshoot; keep the original safe path if resampling invalidates it.
    if len(resampled) >= 2 and all(context.is_edge_valid(resampled[i], resampled[i + 1]) for i in range(len(resampled) - 1)):
        return resampled
    return smoothed if joint_path_length(smoothed) <= joint_path_length(path) else [p.copy() for p in path]
