"""Export selected executed tutorial figures for GitHub README previews."""

from __future__ import annotations

import base64
from pathlib import Path
import shutil

import nbformat


ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "assets"

TUTORIAL_FIGURES = {
    "single_period_waterfall.png": ("single_period_attribution.ipynb", "fig, ax = chart_single_period_excess_return_waterfall"),
    "multiple_period_comparison.png": ("multiple_period_attribution.ipynb", "fig, axes = plt.subplots(2, 2"),
    "multi_account_comparison.png": ("multi_account_attribution.ipynb", "fig, axes = plt.subplots(1, 2"),
    "dow30_interactive.png": ("dow30_brinson_interactive.ipynb", 'category_summary = category.group_by("segment").agg('),
}


def extract_image(notebook_path: Path, code_marker: str) -> bytes:
    notebook = nbformat.read(notebook_path, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type != "code" or code_marker not in cell.source:
            continue
        for output in cell.get("outputs", []):
            encoded = output.get("data", {}).get("image/png")
            if encoded:
                return base64.b64decode(encoded)
    raise ValueError(f"No executed PNG for {code_marker!r} in {notebook_path}")


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    for output_name, (notebook_name, marker) in TUTORIAL_FIGURES.items():
        source = ROOT / "tutorials" / "en" / notebook_name
        target = DESTINATION / output_name
        target.write_bytes(extract_image(source, marker))
        print(target)
    for name in ("synthetic_frontier.png", "synthetic_peer_changes.png"):
        source = ROOT / "outputs" / "baf" / name
        target = DESTINATION / name
        shutil.copyfile(source, target)
        print(target)
    source = ROOT / "outputs" / "dow30_brinson" / "sector_attribution.png"
    target = DESTINATION / "dow30_sector_attribution.png"
    shutil.copyfile(source, target)
    print(target)


if __name__ == "__main__":
    main()
