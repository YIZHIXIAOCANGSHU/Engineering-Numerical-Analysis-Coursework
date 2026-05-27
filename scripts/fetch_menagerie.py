#!/usr/bin/env python3
from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "third_party" / "mujoco_menagerie"
URL = "https://github.com/google-deepmind/mujoco_menagerie.git"


def main() -> None:
    if TARGET.exists():
        subprocess.run(["git", "-C", str(TARGET), "pull", "--ff-only"], check=True)
    else:
        TARGET.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", URL, str(TARGET)], check=True)
    print(f"MuJoCo Menagerie is available at {TARGET}")


if __name__ == "__main__":
    main()
