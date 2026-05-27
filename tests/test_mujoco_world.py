import numpy as np

from arm_planning.ik.solvers import create_ik_solver
from arm_planning.planners.factory import create_planner
from arm_planning.sim.mujoco_world import world_for_scene
from arm_planning.trajectory.smoothing import process_path
from arm_planning.utils.config import load_project_config
from arm_planning.utils.reproducibility import create_rng


def _gate_center_hits(scene, ee_points, threshold: float) -> int:
    frame_ids = sorted({obs.name.split("_")[2] for obs in scene.obstacles if obs.name.startswith("maze_frame_")})
    gate_ids = frame_ids or sorted({obs.name.split("_")[2] for obs in scene.obstacles if obs.name.startswith("maze_gate_")})
    direction = scene.target_position[:2] - ee_points[0, :2]
    direction_norm = np.linalg.norm(direction)
    assert direction_norm > 1e-9
    direction = direction / direction_norm
    hits = 0
    for gate_id in gate_ids:
        if frame_ids:
            pair = [obs for obs in scene.obstacles if f"maze_frame_{gate_id}_" in obs.name and (obs.name.endswith("_left") or obs.name.endswith("_right"))]
        else:
            pair = [obs for obs in scene.obstacles if f"maze_gate_{gate_id}_" in obs.name]
        center = (pair[0].position + pair[1].position) / 2.0
        lateral = pair[1].position[:2] - pair[0].position[:2]
        lateral = lateral / np.linalg.norm(lateral)
        idx = int(np.argmin(np.abs((ee_points[:, :2] - center[:2]) @ direction)))
        offset = abs((ee_points[idx, :2] - center[:2]) @ lateral)
        if offset <= threshold:
            hits += 1
    return hits


def _insert_frame_entry_hits(scene, ee_points) -> int:
    frame = [obs for obs in scene.obstacles if obs.name.startswith("maze_insert_frame_")]
    if not frame:
        return 0
    left = next(obs for obs in frame if obs.name.endswith("_left"))
    right = next(obs for obs in frame if obs.name.endswith("_right"))
    top = next(obs for obs in frame if obs.name.endswith("_top"))
    bottom = next(obs for obs in frame if obs.name.endswith("_bottom"))
    center = (left.position + right.position) / 2.0
    lateral = right.position[:2] - left.position[:2]
    lateral = lateral / np.linalg.norm(lateral)
    aperture_width = np.linalg.norm(right.position[:2] - left.position[:2]) - float(left.size[1])
    lower_z = bottom.position[2] + float(bottom.size[2]) / 2.0
    upper_z = top.position[2] - float(top.size[2]) / 2.0
    offsets = np.abs((ee_points[:, :2] - center[:2]) @ lateral)
    inside = (offsets < aperture_width / 2.0) & (ee_points[:, 2] > lower_z) & (ee_points[:, 2] < upper_z)
    return int(np.count_nonzero(inside))


def test_default_scene_start_and_goal_are_valid():
    cfg = load_project_config()
    scene = cfg.scenes[0]
    world = world_for_scene(cfg.robot, scene, cfg.experiment.get("edge_resolution", 0.06))
    assert world.is_state_valid(scene.q_start)
    result = create_ik_solver("scipy_baseline", cfg.ik).solve(scene.target_position, scene.q_start, world)
    assert result.success
    assert result.position_error < 0.02
    assert world.is_state_valid(result.q)


def test_edge_collision_check_rejects_invalid_limit_state():
    cfg = load_project_config()
    scene = cfg.scenes[0]
    world = world_for_scene(cfg.robot, scene, cfg.experiment.get("edge_resolution", 0.06))
    invalid = np.full(world.dof, 99.0)
    assert not world.is_state_valid(invalid)


def test_ground_geoms_are_detected_and_start_above_ground():
    for config_path in [None, "run_config.yaml"]:
        cfg = load_project_config(config_path=config_path)
        scene = cfg.scenes[0]
        world = world_for_scene(cfg.robot, scene, cfg.experiment.get("edge_resolution", 0.06))
        assert world.ground_geom_ids
        assert np.isfinite(world.ground_clearance(scene.q_start))
        assert world.ground_clearance(scene.q_start) > -1e-4


def test_ground_penetration_invalidates_state():
    cfg = load_project_config()
    scene = cfg.scenes[0]
    world = world_for_scene(cfg.robot, scene, cfg.experiment.get("edge_resolution", 0.06))
    original = world.ground_clearance_current
    world.ground_clearance_current = lambda: -1e-3
    try:
        assert not world.is_state_valid(scene.q_start)
    finally:
        world.ground_clearance_current = original



def test_all_configured_scenes_have_valid_start_and_goal():
    cfg = load_project_config()
    for scene in cfg.scenes:
        world = world_for_scene(cfg.robot, scene, cfg.experiment.get("edge_resolution", 0.06))
        assert world.is_state_valid(scene.q_start), scene.id
        result = create_ik_solver("scipy_baseline", cfg.ik).solve(scene.target_position, scene.q_start, world, scene.target_quat_wxyz)
        if scene.id == "scene_pass_through_maze" and cfg.robot.name == "simple_ur5e":
            assert result.position_error < 0.02, scene.id
            assert result.orientation_error < cfg.ik["orientation_tolerance_rad"], scene.id
        else:
            assert result.success, scene.id
        assert world.is_state_valid(result.q), scene.id


def test_full_ur5e_scene_loads_with_collision_geoms_and_ik_goal():
    cfg = load_project_config(config_path="run_config.yaml")
    scene = next(s for s in cfg.scenes if s.id == "scene_01_single_box")
    world = world_for_scene(cfg.robot, scene, cfg.experiment.get("edge_resolution", 0.06))
    assert world.robot_geom_ids
    assert world.obstacle_geom_ids
    assert world.min_obstacle_distance(scene.q_start) < float("inf")
    assert world.is_state_valid(scene.q_start)
    result = create_ik_solver("scipy_baseline", cfg.ik).solve(scene.target_position, scene.q_start, world)
    assert result.success
    assert result.position_error < cfg.ik["tolerance"]


def test_full_ur5e_all_scenes_have_valid_start_and_ik_goal():
    cfg = load_project_config(config_path="run_config.yaml")
    for scene in cfg.scenes:
        world = world_for_scene(cfg.robot, scene, cfg.experiment.get("edge_resolution", 0.06))
        assert world.is_state_valid(scene.q_start), scene.id
        result = create_ik_solver("scipy_baseline", cfg.ik).solve(
            scene.target_position,
            scene.q_start,
            world,
            scene.target_quat_wxyz,
        )
        assert result.success, scene.id
        assert world.is_state_valid(result.q), scene.id


def test_full_ur5e_mixed_maze_rrt_connect_finds_path():
    cfg = load_project_config(config_path="run_config.yaml")
    scene = next(s for s in cfg.scenes if s.id == "scene_05_mixed_maze_passage")
    world = world_for_scene(cfg.robot, scene, cfg.experiment.get("edge_resolution", 0.06))
    ik = create_ik_solver("scipy_baseline", cfg.ik).solve(scene.target_position, scene.q_start, world, scene.target_quat_wxyz)
    assert ik.success
    assert world.is_state_valid(scene.q_start)
    assert world.is_state_valid(ik.q)
    result = create_planner("rrt_connect", cfg.planners).plan(scene.q_start, ik.q, world, create_rng(42))
    assert result.success
    ee_end = world.forward_kinematics(result.path[-1])
    assert np.linalg.norm(ee_end - scene.target_position) < cfg.ik["tolerance"]


def test_full_ur5e_pass_through_maze_rrt_connect_inserts_into_small_frame():
    cfg = load_project_config(config_path="run_config.yaml")
    scene = next(s for s in cfg.scenes if s.id == "scene_pass_through_maze")
    world = world_for_scene(cfg.robot, scene, cfg.experiment.get("edge_resolution", 0.06))
    assert scene.target_quat_wxyz is not None
    assert not np.allclose(world.forward_quat(scene.q_start), scene.target_quat_wxyz, atol=1e-2)
    ik = create_ik_solver("scipy_baseline", cfg.ik).solve(scene.target_position, scene.q_start, world, scene.target_quat_wxyz)
    assert ik.success
    assert ik.orientation_error < cfg.ik["orientation_tolerance_rad"]
    assert world.is_state_valid(scene.q_start)
    assert world.is_state_valid(ik.q)

    result = create_planner("rrt_connect", cfg.planners).plan(scene.q_start, ik.q, world, create_rng(42))
    assert result.success

    path = process_path(result.path, world, 20, cfg.trajectory.get("samples", 80), create_rng(1041))
    ee_points = world.sample_end_effector_path(path)
    assert np.linalg.norm(ee_points[-1] - scene.target_position) < cfg.ik["tolerance"]
    assert _insert_frame_entry_hits(scene, ee_points) > 0
