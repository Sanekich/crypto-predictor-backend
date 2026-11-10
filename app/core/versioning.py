"""Code and environment versions recorded with every trained model."""

import platform
import subprocess
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def git_revision(root: Path = PROJECT_ROOT) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True).stdout.strip()

    try:
        return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def environment_info() -> dict[str, str]:
    import sklearn
    import xgboost

    return {
        "python": platform.python_version(),
        "xgboost": xgboost.__version__,
        "scikit-learn": sklearn.__version__,
    }
