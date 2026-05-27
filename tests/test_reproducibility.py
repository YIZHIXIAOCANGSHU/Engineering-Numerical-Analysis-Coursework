import numpy as np

from arm_planning.utils.config import load_project_config
from arm_planning.utils.reproducibility import create_rng, make_trial_seed


def test_trial_seed_formula_is_stable():
    assert make_trial_seed(42, 2, 3, 7) == 23049


def test_rng_is_repeatable():
    a = create_rng(123).normal(size=5)
    b = create_rng(123).normal(size=5)
    assert np.allclose(a, b)


def test_config_loads_fixed_scenes():
    cfg = load_project_config()
    assert cfg.global_seed == 42
    assert len(cfg.scenes) >= 2
    assert cfg.robot.ee_site == "ee_site"



def test_config_has_fixed_scenes_including_mixed_maze():
    cfg = load_project_config()
    scene_ids = [scene.id for scene in cfg.scenes]
    assert scene_ids == [
        "scene_00_no_obstacle",
        "scene_01_single_box",
        "scene_02_multi_obstacles",
        "scene_03_narrow_passage",
        "scene_04_near_singularity",
        "scene_05_mixed_maze_passage",
        "scene_pass_through_maze",
    ]
    assert cfg.experiment["trials"] == 50


def test_root_run_config_overrides_robot_and_demo_defaults():
    cfg = load_project_config(config_path="run_config.yaml")
    assert cfg.robot.name == "ur5e_full"
    assert cfg.robot.model_xml == "third_party/mujoco_menagerie/universal_robots_ur5e/scene.xml"
    assert cfg.robot.ee_site == "attachment_site"
    assert cfg.robot.joint_names == [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]
    assert np.allclose(cfg.robot.q_start, [-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])
    assert cfg.experiment["demo_scene"] == "scene_pass_through_maze"
    assert cfg.ik["demo_method"] == "scipy_baseline"
    assert cfg.planners["demo_method"] == "rrt_connect"
