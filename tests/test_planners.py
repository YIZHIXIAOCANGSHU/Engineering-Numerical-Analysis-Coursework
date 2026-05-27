from arm_planning.ik.solvers import create_ik_solver
from arm_planning.planners.factory import create_planner
from arm_planning.sim.mujoco_world import world_for_scene
from arm_planning.utils.config import load_project_config
from arm_planning.utils.reproducibility import create_rng


def test_rrt_connect_finds_default_path():
    cfg = load_project_config()
    scene = cfg.scenes[0]
    world = world_for_scene(cfg.robot, scene, cfg.experiment.get("edge_resolution", 0.06))
    ik = create_ik_solver("scipy_baseline", cfg.ik).solve(scene.target_position, scene.q_start, world)
    planner = create_planner("rrt_connect", cfg.planners)
    result = planner.plan(scene.q_start, ik.q, world, create_rng(42))
    assert result.success
    assert len(result.path) >= 2
    assert world.is_edge_valid(result.path[-2], result.path[-1])
