#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
unset PYTHONPATH

VENV_DIR="${VENV_DIR:-.venv}"
PYTHON_BIN="$VENV_DIR/bin/python"
INSTALL_STAMP="$VENV_DIR/.arm_planning_install_stamp"
CONFIG_PATH="${CONFIG_PATH:-run_config.yaml}"
TRIALS="${TRIALS:-1}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Creating virtual environment at $VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi

if [[ ! -f "$INSTALL_STAMP" || pyproject.toml -nt "$INSTALL_STAMP" ]]; then
  echo "Installing project dependencies in $VENV_DIR"
  "$PYTHON_BIN" -m pip install -U pip setuptools wheel
  "$PYTHON_BIN" -m pip install -e .
  touch "$INSTALL_STAMP"
fi

choose_value() {
  local prompt="$1"
  local default="$2"
  shift 2
  local options=("$@")
  local choice=""

  echo >&2
  echo "$prompt" >&2
  local i=1
  for option in "${options[@]}"; do
    if [[ "$option" == "$default" ]]; then
      echo "  $i) $option [default]" >&2
    else
      echo "  $i) $option" >&2
    fi
    i=$((i + 1))
  done

  printf "Enter number: " >&2
  read -r choice
  if [[ -z "$choice" ]]; then
    echo "$default"
    return
  fi
  if ! [[ "$choice" =~ ^[0-9]+$ ]] || (( choice < 1 || choice > ${#options[@]} )); then
    echo "Invalid choice: $choice" >&2
    exit 2
  fi
  echo "${options[$((choice - 1))]}"
}

DEFAULT_IK="$("$PYTHON_BIN" - "$CONFIG_PATH" <<'PY'
import sys
from arm_planning.utils.config import load_project_config
cfg = load_project_config(config_path=sys.argv[1])
print(cfg.ik.get("demo_method", "scipy_baseline"))
PY
)"
DEFAULT_PLANNER="$("$PYTHON_BIN" - "$CONFIG_PATH" <<'PY'
import sys
from arm_planning.utils.config import load_project_config
cfg = load_project_config(config_path=sys.argv[1])
print(cfg.planners.get("demo_method", "rrt_connect"))
PY
)"
DEFAULT_SCENE="$("$PYTHON_BIN" - "$CONFIG_PATH" <<'PY'
import sys
from arm_planning.utils.config import load_project_config
cfg = load_project_config(config_path=sys.argv[1])
print(cfg.experiment.get("demo_scene", cfg.scenes[0].id))
PY
)"

MODE="$(choose_value "Select run mode" "single_combo" single_combo all_combinations batch_and_export)"

NO_VIEWER_ARGS=()
if [[ "${NO_VIEWER:-0}" == "1" ]]; then
  NO_VIEWER_ARGS=(--no-viewer)
fi

if [[ "$MODE" == "batch_and_export" ]]; then
  echo
  echo "Running batch experiments with TRIALS=$TRIALS, then exporting paper figures"
  "$PYTHON_BIN" -m arm_planning run-experiments \
    --config "$CONFIG_PATH" \
    --scene "$DEFAULT_SCENE" \
    --trials "$TRIALS" \
    --no-rerun
  "$PYTHON_BIN" -m arm_planning export-plots --config "$CONFIG_PATH"
  echo "Figures exported under images/generated"
  exit 0
fi

if [[ "$MODE" == "all_combinations" ]]; then
  echo
  echo "Running all IK/planner combinations once on scene=$DEFAULT_SCENE config=$CONFIG_PATH"
  "$PYTHON_BIN" -m arm_planning run-all-combos \
    --config "$CONFIG_PATH" \
    --scene "$DEFAULT_SCENE" \
    "${NO_VIEWER_ARGS[@]}"
  exit 0
fi

IK_METHOD="$(choose_value "Select IK algorithm" "$DEFAULT_IK" pinv dls lm scipy_baseline)"
PLANNER_METHOD="$(choose_value "Select obstacle-avoidance planner" "$DEFAULT_PLANNER" rrt rrt_connect prm apf)"

echo
echo "Running demo with IK=$IK_METHOD planner=$PLANNER_METHOD scene=$DEFAULT_SCENE config=$CONFIG_PATH"
"$PYTHON_BIN" -m arm_planning run-demo \
  --config "$CONFIG_PATH" \
  --ik-method "$IK_METHOD" \
  --planner "$PLANNER_METHOD" \
  --scene "$DEFAULT_SCENE" \
  "${NO_VIEWER_ARGS[@]}"
