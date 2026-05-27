from __future__ import annotations

import numpy as np


def wrap_to_pi(q: np.ndarray) -> np.ndarray:
    return (np.asarray(q, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi


def clip_to_limits(q: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(np.asarray(q, dtype=float), lower), upper)


def interpolate_joint_path(q1: np.ndarray, q2: np.ndarray, resolution: float) -> list[np.ndarray]:
    q1 = np.asarray(q1, dtype=float)
    q2 = np.asarray(q2, dtype=float)
    distance = float(np.linalg.norm(q2 - q1))
    steps = max(2, int(np.ceil(distance / max(resolution, 1e-6))) + 1)
    return [q1 + (q2 - q1) * alpha for alpha in np.linspace(0.0, 1.0, steps)]


def joint_path_length(path: list[np.ndarray]) -> float:
    if len(path) < 2:
        return 0.0
    return float(sum(np.linalg.norm(path[i + 1] - path[i]) for i in range(len(path) - 1)))


def nearest_index(points: list[np.ndarray], sample: np.ndarray) -> int:
    dists = [float(np.linalg.norm(point - sample)) for point in points]
    return int(np.argmin(dists))
