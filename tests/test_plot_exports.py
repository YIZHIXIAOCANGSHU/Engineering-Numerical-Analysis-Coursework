import csv

from arm_planning.analysis.plots import _failure_snapshot_candidates, _short_label, export_paper_trajectory_figures, export_trajectory_metric_charts
from arm_planning.experiments.result_store import array_to_string


def test_paper_trajectory_export_writes_png_pdf_svg(tmp_path):
    csv_path = tmp_path / "trajectory_samples.csv"
    rows = []
    for i in range(5):
        rows.append(
            {
                "scene_id": "scene_05_mixed_maze_passage",
                "planner": "rrt_connect",
                "trial_id": 0,
                "sample_index": i,
                "q": array_to_string([0.1 * i, -0.2, 0.3, 0.1, -0.1, 0.0]),
                "ee_x": 0.20 + 0.05 * i,
                "ee_y": 0.02 * i,
                "ee_z": 0.35 + 0.01 * i,
                "ee_error": 0.1 / (i + 1),
                "ee_orientation_error": 0.2 / (i + 1),
                "cumulative_task_length": 0.05 * i,
                "min_obstacle_distance": 0.05 + 0.01 * i,
                "min_ground_clearance": 0.10 + 0.005 * i,
                "joint_speed_norm": 0.02 * i,
                "joint_acc_norm": 0.01 * i,
                "joint_jerk_norm": 0.005 * i,
            }
        )
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    export_paper_trajectory_figures(csv_path, tmp_path)

    for suffix in ["png", "pdf", "svg"]:
        assert (tmp_path / f"paper_mixed_maze_ee_path.{suffix}").exists()
        assert (tmp_path / f"paper_error_and_safety_curves.{suffix}").exists()
        for name in [
            "paper_ee_path_3d_large",
            "paper_ee_path_top_view_large",
            "paper_ee_position_x",
            "paper_ee_position_y",
            "paper_ee_position_z",
            "paper_ee_position_error",
            "paper_ee_orientation_error",
            "paper_cumulative_task_length",
            "paper_min_obstacle_distance",
            "paper_min_ground_clearance",
            "paper_joint_angles_large",
            "paper_joint_speed_norm",
            "paper_joint_acc_norm",
            "paper_joint_jerk_norm",
        ]:
            assert (tmp_path / f"{name}.{suffix}").exists()


def test_metric_export_writes_large_paper_figures(tmp_path):
    csv_path = tmp_path / "trajectory_metrics.csv"
    rows = [
        {
            "scene_id": "scene_pass_through_maze",
            "trial_id": 0,
            "planner": "rrt",
            "path_length_task": 0.90,
            "smoothness": 0.03,
            "final_error": 0.012,
            "min_obstacle_distance": 0.04,
            "mean_joint_jerk_norm": 0.10,
        },
        {
            "scene_id": "scene_pass_through_maze",
            "trial_id": 0,
            "planner": "rrt_connect",
            "path_length_task": 0.82,
            "smoothness": 0.02,
            "final_error": 0.006,
            "min_obstacle_distance": 0.05,
            "mean_joint_jerk_norm": 0.08,
        },
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    export_trajectory_metric_charts(csv_path, tmp_path)

    for suffix in ["png", "pdf", "svg"]:
        for name in [
            "paper_metric_path_length",
            "paper_metric_smoothness",
            "paper_metric_final_error",
        ]:
            assert (tmp_path / f"{name}.{suffix}").exists()


def test_failure_snapshot_candidates_keep_one_per_category():
    import pandas as pd

    df = pd.DataFrame(
        [
            {"failure_category": "iteration_limit", "failure_category_cn": "迭代上限", "ee_error": 0.2},
            {"failure_category": "iteration_limit", "failure_category_cn": "迭代上限", "ee_error": 0.5},
            {"failure_category": "local_minimum", "failure_category_cn": "局部极小值", "ee_error": 0.1},
            {"failure_category": "roadmap_disconnected", "failure_category_cn": "采样图未连通", "ee_error": 0.4},
        ]
    )

    selected = _failure_snapshot_candidates(df, limit=2)

    assert list(selected["failure_category"]) == ["local_minimum", "iteration_limit"]
    assert float(selected.iloc[1]["ee_error"]) == 0.5


def test_short_labels_keep_legends_compact():
    assert _short_label("rrt_connect") == "RRT-C"
    assert _short_label("scipy_baseline") == "SciPy"
    assert _short_label("采样图未连通") == "图未连通"
