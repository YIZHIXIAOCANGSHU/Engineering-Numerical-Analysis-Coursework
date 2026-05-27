from __future__ import annotations

import argparse

from arm_planning.experiments.runner import run_all_combos, run_demo, run_experiments
from arm_planning.utils.config import ensure_output_dirs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="arm-planning")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("run-demo", help="Run the configured planning demo")
    demo.add_argument("--config", default=None, help="Optional root YAML config overlay")
    demo.add_argument("--ik-method", default=None, help="IK method to use for the demo goal")
    demo.add_argument("--planner", default=None, help="Planner/obstacle-avoidance method to use for the demo")
    demo.add_argument("--scene", default=None, help="Scene id to run")
    demo.add_argument("--no-viewer", action="store_true", help="Do not open the MuJoCo viewer")

    all_combos = sub.add_parser("run-all-combos", help="Run every configured IK/planner pair once")
    all_combos.add_argument("--config", default=None, help="Optional root YAML config overlay")
    all_combos.add_argument("--scene", default=None, help="Scene id to run")
    all_combos.add_argument("--no-viewer", action="store_true", help="Do not open the MuJoCo viewer")

    exp = sub.add_parser("run-experiments", help="Run IK and planner comparison experiments")
    exp.add_argument("--config", default=None, help="Optional root YAML config overlay")
    exp.add_argument("--scene", default=None, help="Scene id to run")
    exp.add_argument("--trials", type=int, default=None, help="Override trials per scene/algorithm")
    exp.add_argument("--no-rerun", action="store_true", help="Disable Rerun recording")
    exp.add_argument("--append", action="store_true", help="Append to existing CSV files")

    plots = sub.add_parser("export-plots", help="Export Matplotlib figures for LaTeX")
    plots.add_argument("--config", default=None, help="Optional root YAML config overlay")
    return parser


def main(argv: list[str] | None = None) -> None:
    ensure_output_dirs()
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run-demo":
        run_demo(
            no_viewer=args.no_viewer,
            config_path=args.config,
            ik_method=args.ik_method,
            planner_name=args.planner,
            scene_id=args.scene,
        )
    elif args.command == "run-all-combos":
        run_all_combos(config_path=args.config, scene_id=args.scene, no_viewer=args.no_viewer)
    elif args.command == "run-experiments":
        run_experiments(
            trials=args.trials,
            reset=not args.append,
            log_rerun=not args.no_rerun,
            config_path=args.config,
            scene_id=args.scene,
        )
    elif args.command == "export-plots":
        from arm_planning.analysis.plots import export_all_plots

        export_all_plots(config_path=args.config)
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
