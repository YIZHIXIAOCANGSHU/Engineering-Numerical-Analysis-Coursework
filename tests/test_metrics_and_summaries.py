import csv

import numpy as np
import pandas as pd

from arm_planning.analysis.metrics import trajectory_kinematic_metrics
from arm_planning.experiments.failure import failure_category
from arm_planning.experiments.result_store import ResultStore
from arm_planning.core.types import PlanResult, SceneSpec
from arm_planning.experiments.runner import _failure_snapshot_row, _path_is_valid, write_summary_tables


class DummyWorld:
    def sample_end_effector_path(self, path):
        return np.asarray([[q[0], 0.0, 0.0] for q in path], dtype=float)

    def forward_kinematics(self, q):
        return np.asarray([float(q[0]), 0.0, 0.0], dtype=float)

    def forward_quat(self, q):
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=float)

    def sample_end_effector_quats(self, path):
        return np.asarray([[1.0, 0.0, 0.0, 0.0] for _ in path], dtype=float)

    def min_obstacle_distance(self, q):
        return 0.25 + float(q[0]) * 0.01

    def ground_clearance(self, q):
        return 0.50 + float(q[0]) * 0.01


class EdgeInvalidWorld:
    def is_state_valid(self, q):
        return True

    def is_edge_valid(self, q1, q2):
        return False


def test_path_validation_checks_interpolated_edges():
    path = [np.asarray([0.0]), np.asarray([1.0])]

    assert not _path_is_valid(path, EdgeInvalidWorld())


def test_trajectory_metrics_include_jerk_norms():
    path = [
        np.asarray([0.0, 0.0]),
        np.asarray([1.0, 0.0]),
        np.asarray([3.0, 0.0]),
        np.asarray([8.0, 0.0]),
        np.asarray([18.0, 0.0]),
    ]

    metrics = trajectory_kinematic_metrics(path, DummyWorld(), np.asarray([18.0, 0.0, 0.0]))

    assert metrics["max_joint_jerk_norm"] == 2.0
    assert metrics["mean_joint_jerk_norm"] == 2.0
    assert metrics["smoothness"] > 0.0
    assert metrics["min_ground_clearance"] > 0.0
    assert metrics["safety_margin"] > 0.0
    assert metrics["final_orientation_error"] == 0.0


def test_trajectory_rows_include_ground_clearance():
    from arm_planning.experiments.runner import trajectory_rows

    path = [np.asarray([0.0]), np.asarray([1.0])]
    rows = trajectory_rows(
        "scene_a",
        "rrt",
        0,
        path,
        np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float),
        DummyWorld(),
        np.asarray([1.0, 0.0, 0.0]),
    )

    assert "min_ground_clearance" in rows[0]
    assert rows[0]["min_ground_clearance"] == 0.50
    assert rows[1]["min_ground_clearance"] == 0.51


def test_failure_category_maps_messages_to_paper_labels():
    assert failure_category(False, "max iterations reached", "planner") == ("iteration_limit", "迭代上限")
    assert failure_category(False, "no roadmap path", "planner") == ("roadmap_disconnected", "采样图未连通")
    assert failure_category(False, "local minimum", "planner") == ("local_minimum", "局部极小值")
    assert failure_category(False, "ground penetration", "validation") == ("ground_penetration", "地面穿透")
    assert failure_category(False, "blocked by obstacle", "planner") == ("obstacle_collision", "障碍碰撞")
    assert failure_category(False, "max iterations reached", "ik") == ("ik_failed", "IK 未收敛")
    assert failure_category(True, "success", "planner") == ("success", "成功")


def test_failure_snapshot_row_uses_failure_q_metadata():
    scene = SceneSpec(
        id="scene_a",
        seed=7,
        q_start=np.asarray([0.0]),
        target_position=np.asarray([1.0, 0.0, 0.0]),
        obstacles=[],
    )
    result = PlanResult(
        success=False,
        path=[],
        planning_time=0.2,
        path_length=0.0,
        collision_checks=12,
        planner_name="rrt",
        message="max iterations reached",
        metadata={"failure_q": np.asarray([0.4])},
    )

    row = _failure_snapshot_row(
        scene,
        "rrt",
        0,
        123,
        "iteration_limit",
        "迭代上限",
        result.message,
        result,
        [],
        DummyWorld(),
    )

    assert row is not None
    assert row["q_snapshot"] == "[0.4]"
    assert row["ee_error"] == 0.6
    assert row["failure_category_cn"] == "迭代上限"


def test_write_summary_tables_exports_extended_statistics(tmp_path):
    store = ResultStore(tmp_path)
    planner_rows = [
        {
            "scene_id": "scene_a",
            "planner": "rrt",
            "success": 1,
            "planning_time": 0.10,
            "path_length_joint": 1.0,
            "path_length_task": 0.9,
            "smoothness": 0.20,
            "min_obstacle_distance": 0.12,
            "min_ground_clearance": 0.15,
            "collision_checks": 10,
            "final_error": 0.01,
            "final_orientation_error": 0.02,
            "message": "success",
            "failure_category": "success",
            "failure_category_cn": "成功",
        },
        {
            "scene_id": "scene_a",
            "planner": "rrt",
            "success": 0,
            "planning_time": 0.30,
            "path_length_joint": 2.0,
            "path_length_task": 1.8,
            "smoothness": 0.50,
            "min_obstacle_distance": 0.02,
            "min_ground_clearance": 0.10,
            "collision_checks": 30,
            "final_error": 0.30,
            "final_orientation_error": 0.40,
            "message": "max iterations reached",
            "failure_category": "iteration_limit",
            "failure_category_cn": "迭代上限",
        },
    ]
    with store.planner_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(planner_rows[0].keys()))
        writer.writeheader()
        writer.writerows(planner_rows)

    ik_rows = [
        {
            "scene_id": "scene_a",
            "ik_method": "dls",
            "success": 1,
            "position_error": 0.01,
            "solve_time": 0.02,
            "iterations": 5,
            "condition_number": 20.0,
        },
        {
            "scene_id": "scene_a",
            "ik_method": "dls",
            "success": 1,
            "position_error": 0.03,
            "solve_time": 0.04,
            "iterations": 9,
            "condition_number": 60.0,
        },
    ]
    with store.ik_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(ik_rows[0].keys()))
        writer.writeheader()
        writer.writerows(ik_rows)

    write_summary_tables(store)

    planner_summary = pd.read_csv(store.planner_summary_path)
    ik_summary = pd.read_csv(store.ik_summary_path)

    assert planner_summary.loc[0, "success_ci95"] > 0.0
    assert planner_summary.loc[0, "planning_time_p50"] == 0.20
    assert planner_summary.loc[0, "failure_rate"] == 0.5
    assert planner_summary.loc[0, "final_orientation_error_mean"] == 0.21
    assert "min_ground_clearance_p10" in planner_summary.columns
    assert "composite_score" in planner_summary.columns
    assert "message" in planner_summary.columns
    assert "failure_category_cn" in planner_summary.columns
    assert ik_summary.loc[0, "condition_number_p90"] > ik_summary.loc[0, "condition_number_p50"]
    assert ik_summary.loc[0, "error_p90"] > ik_summary.loc[0, "error_p50"]

    failure_summary = pd.read_csv(store.planner_failure_summary_path)
    assert {"failure_category", "failure_category_cn", "count", "share"}.issubset(failure_summary.columns)
