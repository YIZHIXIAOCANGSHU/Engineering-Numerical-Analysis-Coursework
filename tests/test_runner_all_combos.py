from arm_planning.experiments.runner import run_all_combos


def test_run_all_combos_no_viewer_visits_every_ik_planner_pair():
    results = run_all_combos(config_path="run_config.yaml", scene_id="scene_pass_through_maze", no_viewer=True)

    assert len(results) == 16
    assert {(row["ik_method"], row["planner"]) for row in results} == {
        (ik, planner)
        for ik in ["pinv", "dls", "lm", "scipy_baseline"]
        for planner in ["rrt", "rrt_connect", "prm", "apf"]
    }
    success_count = sum(1 for row in results if row["success"])
    playback_count = sum(1 for row in results if row["playback_success"])
    assert 8 <= success_count <= 12
    assert playback_count == success_count
    assert any(not row["success"] for row in results)
