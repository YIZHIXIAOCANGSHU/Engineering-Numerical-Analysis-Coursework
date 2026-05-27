import numpy as np

from arm_planning.experiments.demo_scene import mixed_maze_scene, random_blocking_scene
from arm_planning.utils.config import load_project_config


def test_random_blocking_scene_places_large_obstacles_near_start_target_line():
    cfg = load_project_config(config_path="run_config.yaml")
    scene = next(s for s in cfg.scenes if s.id == cfg.experiment["demo_scene"])
    ee_start = np.asarray([-0.134, 0.492, 0.488], dtype=float)
    blocking = random_blocking_scene(
        scene,
        ee_start,
        {
            "enabled": True,
            "seed": 123,
            "count": 4,
            "box_size_min": [0.16, 0.16, 0.34],
            "box_size_max": [0.28, 0.28, 0.58],
            "lateral_offset_range": [-0.08, 0.08],
        },
    )

    assert blocking.id.endswith("_random_blocking")
    assert blocking.seed == 123
    assert len(blocking.obstacles) == 4
    line = scene.target_position[:2] - ee_start[:2]
    line_norm = np.linalg.norm(line)
    for obstacle in blocking.obstacles:
        assert obstacle.type == "box"
        assert obstacle.size is not None
        assert np.all(obstacle.size >= [0.16, 0.16, 0.34])
        rel = obstacle.position[:2] - ee_start[:2]
        distance_to_line = abs((line[0] * rel[1] - line[1] * rel[0]) / line_norm)
        assert distance_to_line <= 0.12


def test_mixed_maze_scene_is_reproducible_and_forms_gate_pairs():
    cfg = load_project_config(config_path="run_config.yaml")
    scene = next(s for s in cfg.scenes if s.id == "scene_05_mixed_maze_passage")
    ee_start = np.asarray([-0.134, 0.492, 0.488], dtype=float)
    params = {
        "enabled": True,
        "seed": 20260601,
        "gate_count": 4,
        "corridor_width": 0.24,
    }

    first = mixed_maze_scene(scene, ee_start, params)
    second = mixed_maze_scene(scene, ee_start, params)

    assert first.id == "scene_05_mixed_maze_passage"
    assert len(first.obstacles) >= 8
    for obs_a, obs_b in zip(first.obstacles, second.obstacles):
        assert obs_a.name == obs_b.name
        assert np.allclose(obs_a.position, obs_b.position)
        assert np.allclose(obs_a.size, obs_b.size)

    gate_indices = sorted({obs.name.split("_")[2] for obs in first.obstacles if obs.name.startswith("maze_gate_")})
    assert len(gate_indices) >= 3
    for gate in gate_indices:
        pair = [obs for obs in first.obstacles if f"maze_gate_{gate}_" in obs.name]
        assert len(pair) == 2
        assert np.linalg.norm(pair[0].position[:2] - pair[1].position[:2]) > 0.24


def test_pass_through_maze_uses_auto_end_effector_start():
    cfg = load_project_config(config_path="run_config.yaml")
    scene = next(s for s in cfg.scenes if s.id == "scene_pass_through_maze")

    assert scene.target_quat_wxyz is not None
    assert np.isclose(np.linalg.norm(scene.target_quat_wxyz), 1.0, atol=1e-4)
    assert len(scene.obstacles) >= 12
    frame = [obs for obs in scene.obstacles if obs.name.startswith("maze_insert_frame_")]
    floating = [obs for obs in scene.obstacles if obs.name.startswith("maze_float_")]
    assert len(frame) == 4
    assert len(floating) >= 8
    assert all(obs.quat_wxyz is not None for obs in frame)
    left = next(obs for obs in frame if obs.name.endswith("_left"))
    right = next(obs for obs in frame if obs.name.endswith("_right"))
    top = next(obs for obs in frame if obs.name.endswith("_top"))
    bottom = next(obs for obs in frame if obs.name.endswith("_bottom"))
    center = (left.position + right.position) / 2.0
    direction = scene.target_position[:2] - np.asarray([-0.134, 0.492])
    direction = direction / np.linalg.norm(direction)
    target_depth = float((scene.target_position[:2] - center[:2]) @ direction)
    assert 0.01 <= target_depth <= 0.04
    assert 0.54 <= np.linalg.norm(left.position[:2] - right.position[:2]) <= 0.56
    assert top.position[2] > bottom.position[2]
    assert bottom.position[2] < scene.target_position[2] < top.position[2]

    for obs in floating:
        assert obs.position[2] >= 0.24
