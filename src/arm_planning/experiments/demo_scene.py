from __future__ import annotations

import time
from dataclasses import replace
from typing import Any

import numpy as np

from arm_planning.core.types import ObstacleSpec, SceneSpec


def _as_range(values: Any, default: tuple[float, float]) -> tuple[float, float]:
    if values is None:
        return default
    vals = list(values)
    if len(vals) != 2:
        raise ValueError(f"Expected a two-value range, got: {values}")
    return float(vals[0]), float(vals[1])


def _as_vector(values: Any, default: tuple[float, float, float]) -> np.ndarray:
    if values is None:
        return np.asarray(default, dtype=float)
    vals = list(values)
    if len(vals) != 3:
        raise ValueError(f"Expected a three-value vector, got: {values}")
    return np.asarray(vals, dtype=float)


def _quat_from_yaw(yaw: float) -> np.ndarray:
    half = 0.5 * float(yaw)
    return np.asarray([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=float)


def _append_frame_parts(
    obstacles: list[ObstacleSpec],
    prefix: str,
    center: np.ndarray,
    lateral: np.ndarray,
    frame_quat: np.ndarray,
    aperture_width: float,
    aperture_height: float,
    frame_depth: float,
    post_width: float,
    beam_height: float,
    bottom_height: float,
    bottom_z: float,
) -> None:
    half_width = aperture_width / 2.0 + post_width / 2.0
    post_z = bottom_z + bottom_height + aperture_height / 2.0
    top_z = bottom_z + bottom_height + aperture_height + beam_height / 2.0
    bottom_beam_z = bottom_z + bottom_height / 2.0
    for name, offset, z, size in [
        ("left", -half_width, post_z, [frame_depth, post_width, aperture_height + 2.0 * beam_height]),
        ("right", half_width, post_z, [frame_depth, post_width, aperture_height + 2.0 * beam_height]),
        ("top", 0.0, top_z, [frame_depth, aperture_width + 2.0 * post_width, beam_height]),
        ("bottom", 0.0, bottom_beam_z, [frame_depth, aperture_width + 2.0 * post_width, bottom_height]),
    ]:
        pos = center.copy()
        pos[:2] += lateral * offset
        pos[2] = z
        obstacles.append(
            ObstacleSpec(
                type="box",
                name=f"{prefix}_{name}",
                position=pos,
                size=np.asarray(size, dtype=float),
                quat_wxyz=frame_quat.copy(),
            )
        )


def _append_floating_obstacles(
    obstacles: list[ObstacleSpec],
    rng: np.random.Generator,
    start: np.ndarray,
    line: np.ndarray,
    direction: np.ndarray,
    lateral: np.ndarray,
    frame_quat: np.ndarray,
    params: dict[str, Any],
    target: np.ndarray | None = None,
) -> None:
    floating_count = int(params.get("floating_count", 0))
    if floating_count <= 0:
        return
    float_t_range = _as_range(params.get("floating_fraction_range"), (0.42, 0.88))
    if "floating_lateral_abs_range" in params:
        lat_range = _as_range(params.get("floating_lateral_abs_range"), (0.10, 0.32))
        min_lat = min(abs(lat_range[0]), abs(lat_range[1]))
        max_lat = max(abs(lat_range[0]), abs(lat_range[1]))
    else:
        lat_range = _as_range(params.get("floating_lateral_range"), (-0.32, 0.32))
        min_lat = min(abs(lat_range[0]), abs(lat_range[1])) * 0.55
        max_lat = max(abs(lat_range[0]), abs(lat_range[1]))
    float_z_range = _as_range(params.get("floating_z_range"), (0.30, 0.58))
    float_size_min = _as_vector(params.get("floating_box_size_min"), (0.08, 0.08, 0.08))
    float_size_max = _as_vector(params.get("floating_box_size_max"), (0.15, 0.15, 0.15))
    radius_range = _as_range(params.get("floating_sphere_radius_range"), (0.03, 0.048))
    forward_jitter = float(params.get("floating_forward_jitter", 0.025))
    target_keepout = float(params.get("floating_target_keepout", 0.0))
    for idx, fraction in enumerate(np.linspace(float_t_range[0], float_t_range[1], floating_count)):
        center = start + float(fraction) * line
        side = -1.0 if idx % 2 == 0 else 1.0
        center[:2] += lateral * (side * rng.uniform(min_lat, max_lat))
        center[:2] += direction * rng.uniform(-forward_jitter, forward_jitter)
        center[2] = rng.uniform(float_z_range[0], float_z_range[1])
        if target is not None and target_keepout > 0.0:
            delta = center[:2] - target[:2]
            distance = float(np.linalg.norm(delta))
            if distance < target_keepout:
                push = delta / distance if distance > 1e-9 else side * lateral
                center[:2] = target[:2] + push * target_keepout
        if idx % 2 == 0:
            obstacles.append(
                ObstacleSpec(
                    type="sphere",
                    name=f"maze_float_{idx:02d}_sphere",
                    position=center,
                    radius=float(rng.uniform(radius_range[0], radius_range[1])),
                )
            )
        else:
            obstacles.append(
                ObstacleSpec(
                    type="box",
                    name=f"maze_float_{idx:02d}_box",
                    position=center,
                    size=rng.uniform(float_size_min, float_size_max),
                    quat_wxyz=frame_quat.copy(),
                )
            )


def random_blocking_scene(scene: SceneSpec, ee_start: np.ndarray, params: dict[str, Any]) -> SceneSpec:
    if not params.get("enabled", False):
        return scene

    seed = params.get("seed")
    if seed is None:
        seed = int(time.time_ns() % (2**32 - 1))
    seed = int(seed)
    rng = np.random.default_rng(seed)

    count = max(1, int(params.get("count", 5)))
    t_range = _as_range(params.get("path_fraction_range"), (0.25, 0.78))
    lateral_range = _as_range(params.get("lateral_offset_range"), (-0.11, 0.11))
    z_range = _as_range(params.get("z_range"), (0.24, 0.48))
    min_size = _as_vector(params.get("box_size_min"), (0.16, 0.16, 0.34))
    max_size = _as_vector(params.get("box_size_max"), (0.28, 0.28, 0.58))

    start = np.asarray(ee_start, dtype=float)
    target = np.asarray(scene.target_position, dtype=float)
    line = target - start
    line_xy = line[:2]
    line_xy_norm = float(np.linalg.norm(line_xy))
    if line_xy_norm <= 1e-9:
        lateral = np.asarray([1.0, 0.0], dtype=float)
    else:
        lateral = np.asarray([-line_xy[1], line_xy[0]], dtype=float) / line_xy_norm

    fractions = np.linspace(t_range[0], t_range[1], count)
    obstacles: list[ObstacleSpec] = []
    for idx, fraction in enumerate(fractions):
        fraction = float(np.clip(fraction + rng.uniform(-0.04, 0.04), 0.15, 0.88))
        center = start + fraction * line
        center[:2] += lateral * rng.uniform(lateral_range[0], lateral_range[1])
        center[2] = rng.uniform(z_range[0], z_range[1])
        size = rng.uniform(min_size, max_size)
        obstacles.append(
            ObstacleSpec(
                type="box",
                name=f"obs_block_{idx:02d}",
                position=center,
                size=size,
            )
        )

    if not bool(params.get("replace_existing", True)):
        obstacles = list(scene.obstacles) + obstacles

    return replace(
        scene,
        id=f"{scene.id}_random_blocking",
        seed=seed,
        obstacles=obstacles,
    )


def mixed_maze_scene(scene: SceneSpec, ee_start: np.ndarray, params: dict[str, Any]) -> SceneSpec:
    if not params.get("enabled", True):
        return scene

    seed = int(params.get("seed", scene.seed))
    rng = np.random.default_rng(seed)
    gate_count = max(3, int(params.get("gate_count", 4)))
    corridor_width = float(params.get("corridor_width", 0.24))
    gate_depth = float(params.get("gate_depth", 0.16))
    wall_thickness = float(params.get("wall_thickness", 0.10))
    wall_height = float(params.get("wall_height", 0.38))
    z_center = float(params.get("z_center", 0.30))
    curve_amplitude = float(params.get("curve_amplitude", 0.10))
    t_range = _as_range(params.get("path_fraction_range"), (0.20, 0.82))
    offset_wall_count = int(params.get("offset_wall_count", max(1, gate_count - 2)))

    start = np.asarray(ee_start, dtype=float)
    target = np.asarray(scene.target_position, dtype=float)
    line = target - start
    line_xy = line[:2]
    line_xy_norm = float(np.linalg.norm(line_xy))
    if line_xy_norm <= 1e-9:
        direction = np.asarray([1.0, 0.0], dtype=float)
    else:
        direction = line_xy / line_xy_norm
    lateral = np.asarray([-direction[1], direction[0]], dtype=float)
    frame_quat = _quat_from_yaw(float(np.arctan2(direction[1], direction[0])))

    obstacles: list[ObstacleSpec] = []
    fractions = np.linspace(t_range[0], t_range[1], gate_count)
    if bool(params.get("frame_mode", False)):
        aperture_width = float(params.get("aperture_width", params.get("corridor_width", 0.90)))
        aperture_height = float(params.get("aperture_height", 0.54))
        post_width = float(params.get("post_width", wall_thickness))
        beam_height = float(params.get("beam_height", post_width))
        bottom_height = float(params.get("bottom_height", beam_height * 0.75))
        frame_depth = float(params.get("frame_depth", gate_depth))
        bottom_z = float(params.get("bottom_z", z_center))
        if params.get("frame_layout") == "insert_target":
            target_depth = float(params.get("target_depth", 0.06))
            frame_lateral_offset = float(params.get("frame_lateral_offset", 0.0))
            short_reach_threshold = float(params.get("short_reach_distance_threshold", 0.0))
            if short_reach_threshold > 0.0 and line_xy_norm < short_reach_threshold:
                target_depth = float(params.get("short_reach_target_depth", target_depth))
                frame_lateral_offset = float(params.get("short_reach_frame_lateral_offset", frame_lateral_offset))
            frame_center = target.copy()
            frame_center[:2] -= direction * target_depth
            frame_center[:2] += lateral * frame_lateral_offset
            _append_frame_parts(
                obstacles,
                "maze_insert_frame",
                frame_center,
                lateral,
                frame_quat,
                aperture_width,
                aperture_height,
                frame_depth,
                post_width,
                beam_height,
                bottom_height,
                bottom_z,
            )
            guide_count = int(params.get("guide_obstacle_count", 2))
            guide_t_range = _as_range(params.get("guide_fraction_range"), (0.52, 0.72))
            guide_width = float(params.get("guide_post_width", post_width))
            guide_height = float(params.get("guide_height", aperture_height * 0.70))
            guide_z = bottom_z + bottom_height + guide_height / 2.0
            guide_offset = aperture_width * 0.48 + guide_width / 2.0
            for idx, fraction in enumerate(np.linspace(guide_t_range[0], guide_t_range[1], guide_count)):
                center = start + float(fraction) * line
                center[:2] += lateral * rng.uniform(-0.025, 0.025)
                center[:2] += lateral * (-1.0 if idx % 2 == 0 else 1.0) * guide_offset
                center[2] = guide_z
                obstacles.append(
                    ObstacleSpec(
                        type="box",
                        name=f"maze_insert_guide_{idx:02d}",
                        position=center,
                        size=np.asarray([frame_depth * 0.75, guide_width, guide_height], dtype=float),
                        quat_wxyz=frame_quat.copy(),
                    )
                )
        else:
            for idx, fraction in enumerate(fractions):
                curve = curve_amplitude * np.sin(idx * np.pi / max(gate_count - 1, 1))
                curve += rng.uniform(-0.012, 0.012)
                center = start + float(fraction) * line
                center[:2] += lateral * curve
                _append_frame_parts(
                    obstacles,
                    f"maze_frame_{idx:02d}",
                    center,
                    lateral,
                    frame_quat,
                    aperture_width,
                    aperture_height,
                    frame_depth,
                    post_width,
                    beam_height,
                    bottom_height,
                    bottom_z,
                )
        _append_floating_obstacles(obstacles, rng, start, line, direction, lateral, frame_quat, params, target)
        if not bool(params.get("replace_existing", True)):
            obstacles = list(scene.obstacles) + obstacles
        return replace(scene, seed=seed, obstacles=obstacles)

    for idx, fraction in enumerate(fractions):
        curve = curve_amplitude * np.sin(idx * np.pi / max(gate_count - 1, 1))
        curve += rng.uniform(-0.025, 0.025)
        center = start + float(fraction) * line
        center[:2] += lateral * curve
        center[2] = z_center + rng.uniform(-0.025, 0.025)

        side_gap = corridor_width / 2.0 + wall_thickness / 2.0
        for side_name, side in [("left", -1.0), ("right", 1.0)]:
            wall_center = center.copy()
            wall_center[:2] += lateral * side * side_gap
            size = np.asarray(
                [
                    gate_depth * rng.uniform(0.85, 1.15),
                    wall_thickness * rng.uniform(0.9, 1.2),
                    wall_height * rng.uniform(0.92, 1.08),
                ],
                dtype=float,
            )
            obstacles.append(
                ObstacleSpec(
                    type="box",
                    name=f"maze_gate_{idx:02d}_{side_name}",
                    position=wall_center,
                    size=size,
                    quat_wxyz=frame_quat.copy(),
                )
            )

    if offset_wall_count > 0:
        for idx, fraction in enumerate(np.linspace(t_range[0] + 0.08, t_range[1] - 0.08, offset_wall_count)):
            blocker = start + float(fraction) * line
            side = -1.0 if idx % 2 == 0 else 1.0
            blocker[:2] += lateral * side * (corridor_width * 0.9 + rng.uniform(0.02, 0.06))
            blocker[2] = z_center + rng.uniform(-0.02, 0.03)
            obstacles.append(
                ObstacleSpec(
                    type="box",
                    name=f"maze_offset_wall_{idx:02d}",
                    position=blocker,
                    size=np.asarray([0.14, 0.12, wall_height * 0.9], dtype=float),
                    quat_wxyz=frame_quat.copy(),
                )
            )

    if not bool(params.get("replace_existing", True)):
        obstacles = list(scene.obstacles) + obstacles

    return replace(scene, seed=seed, obstacles=obstacles)
