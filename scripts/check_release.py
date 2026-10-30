"""Install the built wheel in a temporary target and smoke-test public APIs."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import tomllib


ROOT = Path(__file__).resolve().parents[1]
VERSION = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
WHEEL = ROOT / "dist" / f"brinsons-{VERSION}-py3-none-any.whl"
SDIST = ROOT / "dist" / f"brinsons-{VERSION}.tar.gz"


def main() -> None:
    if not WHEEL.exists():
        raise FileNotFoundError(WHEEL)
    with tempfile.TemporaryDirectory(prefix="brinsons-release-") as target:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-deps", "--target", target, str(WHEEL)],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        script = "\n".join([
            "from pathlib import Path",
            "import brinsons",
            "from brinsons import calculate_single_period, calculate_multiple_period",
            "from brinsons import baf_change_table, chart_baf_change_table",
            f"assert Path(brinsons.__file__).resolve().is_relative_to(Path({target!r}).resolve())",
            "print('Wheel import and public API: OK')",
        ])
        environment = os.environ.copy()
        environment["PYTHONPATH"] = target
        subprocess.run([sys.executable, "-c", script], cwd=target,
                       env=environment, check=True)
    if not SDIST.exists():
        raise FileNotFoundError(SDIST)
    with tempfile.TemporaryDirectory(prefix="brinsons-sdist-") as target:
        with tarfile.open(SDIST) as archive:
            archive.extractall(target, filter="data")
        source = Path(target) / f"brinsons-{VERSION}"
        subprocess.run([sys.executable, "-m", "pytest", "-q"],
                       cwd=source, check=True)


if __name__ == "__main__":
    main()
